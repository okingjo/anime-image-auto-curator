"""Pluggable aesthetic-scoring backends.

All backends share the same contract (see AestheticBackend). Each is
import-safe: if its optional dependency is missing, `available=False` with a
hint, and the UI guides the user to enable it. Models are downloaded lazily
on first `load()` (via the HF mirror) and cached under `.models/`.

Score convention:
  - `score()` returns the model's RAW score (its native scale).
  - `normalize(raw)` maps it to [0, 1] for ranking / the composite score.
The raw value is surfaced in the UI so you can A/B compare models on your
own images — the only reliable way to pick the one matching your taste.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

_CACHE = Path(__file__).resolve().parents[2] / ".models"

# 国内优先的 HF 下载源。huggingface.co 在境内通常不可达，hf-mirror.com 是社区镜像。
# 显式多源 + 重试 + 断点续传，避免 transformers 内部单次下载失败后抛出
# “找不到 preprocessor_config.json / 当成目录”之类误导性报错。
_HF_ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]

# Ordered = UI display order. The first available one is the default fallback.
BACKEND_IDS = ["shadow-v2", "skytnt-anime", "aesthetic-predictor-v2-5", "improved-clip"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _set_hf_endpoint(url: str):
    """Point HuggingFace downloads at a given endpoint via HF_ENDPOINT.

    huggingface_hub reads HF_ENDPOINT from the environment at call time, so
    setting it here reliably redirects subsequent downloads. huggingface.co is
    unreachable from China; the CN mirror is hf-mirror.com.
    """
    os.environ["HF_ENDPOINT"] = url


def _load_with_endpoint_fallback(do_load, label: str):
    """Ensure HF_ENDPOINT points at the CN mirror, then run do_load().

    The proven mechanism is the HF_ENDPOINT environment variable (read by
    huggingface_hub at call time). huggingface.co is unreachable from China,
    so if HF_ENDPOINT is unset — or explicitly the default huggingface.co —
    we repoint it at the mirror first. Respects any custom mirror the user set.
    """
    ep = os.environ.get("HF_ENDPOINT", "")
    if (not ep) or ("huggingface.co" in ep):
        _set_hf_endpoint("https://hf-mirror.com")
        print(f"[{label}] HF_ENDPOINT 未设置或为 huggingface.co，已改用 hf-mirror.com", flush=True)
    return do_load()


def _hf_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    def _do():
        return hf_hub_download(repo_id, filename)

    return _load_with_endpoint_fallback(_do, f"download {filename}")


def _raw_download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bool:
    """纯 HTTP 断点续传下载（不依赖 huggingface_hub）。成功返回 True。"""
    for attempt in range(retries):
        resume = dest.stat().st_size if dest.exists() else 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if resume:
                req.add_header("Range", f"bytes={resume}-")
            mode = "ab" if resume else "wb"
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, mode) as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            return True
        except Exception as e:
            print(f"  [raw] attempt {attempt + 1} failed: {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    return False


def _download_repo_files(repo_id: str, filenames: list, label: str) -> Path:
    """逐文件下载一个 HF 仓库的指定文件，返回可直接 from_pretrained 的本地目录。

    策略：
      1. 已在缓存 -> 直接用。
      2. hf_hub_download（显式 endpoint，逐个尝试国内镜像→官方，重试，自带断点续传）。
      3. 裸 resolve URL 纯 HTTP 断点续传（最后兑底）。
    全部失败才抛异常，报错包含已成功/失败明细，方便排查。
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    _ensure_endpoint()  # 保证 HF_ENDPOINT 指向可用镜像
    local_dir = _CACHE / "hub-local" / repo_id.replace("/", "--")
    snapshot_dir: Optional[Path] = None
    failed = []

    for fn in filenames:
        # 1) 缓存命中
        cached = try_to_load_from_cache(repo_id, fn)
        if isinstance(cached, str) and os.path.exists(cached) and os.path.getsize(cached) > 0:
            snapshot_dir = snapshot_dir or Path(cached).parent
            continue

        got = None
        # 2) hf_hub_download 多源重试
        for ep in _HF_ENDPOINTS:
            for attempt in range(2):
                try:
                    p = hf_hub_download(repo_id, fn, endpoint=ep)
                    snapshot_dir = snapshot_dir or Path(p).parent
                    got = p
                    break
                except Exception as e:
                    print(f"[{label}] hub {ep} {fn} try{attempt + 1}: "
                          f"{type(e).__name__} {str(e)[:100]}", flush=True)
            if got:
                break

        # 3) 裸 URL 兑底
        if not got:
            local_dir.mkdir(parents=True, exist_ok=True)
            dest = local_dir / fn
            for ep in _HF_ENDPOINTS:
                if _raw_download(f"{ep}/{repo_id}/resolve/main/{fn}", dest):
                    got = str(dest)
                    snapshot_dir = local_dir
                    break

        if not got:
            failed.append(fn)

    if failed:
        raise RuntimeError(
            f"下载失败的文件: {failed}。请检查网络/代理后重试；"
            f"也可手动下载 https://hf-mirror.com/{repo_id} 的对应文件。"
        )
    assert snapshot_dir is not None
    return snapshot_dir


def _ensure_endpoint():
    ep = os.environ.get("HF_ENDPOINT", "")
    if (not ep) or ("huggingface.co" in ep):
        _set_hf_endpoint("https://hf-mirror.com")


def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class AestheticBackend:
    id = "base"
    label = "base"
    domain = ""        # 动漫 / 通用
    desc = ""
    scale = ""         # human-readable native scale
    approx_size = "?"
    available = True
    reason = ""

    def __init__(self):
        self._ready = False
        self._failed = False

    def load(self) -> bool:
        raise NotImplementedError

    def score(self, image: Image.Image) -> tuple:
        """Return (raw_score: float, notes: str)."""
        raise NotImplementedError

    def normalize(self, raw: float) -> float:
        return _clamp01(raw)

    def prewarm(self):
        if self.available and not self._ready and not self._failed:
            try:
                self.load()
            except Exception as e:
                self._failed = True
                self.available = False
                self.reason = f"加载失败：{e}"


# --------------------------------------------------------------------------- #
# improved-clip (current default / light baseline)
# --------------------------------------------------------------------------- #
class ImprovedClipBackend(AestheticBackend):
    id = "improved-clip"
    label = "Improved Aesthetic (CLIP L/14)"
    domain = "通用(照片/AI图)"
    desc = "CLIP ViT-L/14 + improved-aesthetic-predictor MLP。SAC+AVA 训练，照片与AI写真为主，动漫偏弱。轻量基线。"
    scale = "约 1–10"
    approx_size = "~1GB (CLIP) + 3.7MB (head)"

    _HEAD_URLS = [
        "https://hf-mirror.com/camenduru/improved-aesthetic-predictor/resolve/main/sac+logos+ava1-l14-linearMSE.pth",
        "https://huggingface.co/camenduru/improved-aesthetic-predictor/resolve/main/sac+logos+ava1-l14-linearMSE.pth",
        "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
    ]

    def __init__(self):
        super().__init__()
        try:
            import torch  # noqa: F401
            import open_clip  # noqa: F401
            self.available = True
        except Exception as e:
            self.available = False
            self.reason = f"缺少 torch/open_clip：uv sync --extra aesthetic ({e})"
        self._model = self._preprocess = self._head = None
        self._device = "cpu"

    @staticmethod
    def _build_head():
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
        )

    def load(self) -> bool:
        if self._ready:
            return True
        import open_clip
        import torch

        self._device = _device()

        def _do():
            return open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")

        model, _, preprocess = _load_with_endpoint_fallback(_do, "improved-clip")
        model.to(self._device).eval()
        self._model, self._preprocess = model, preprocess
        self._head = self._load_head()
        self._ready = True
        return True

    def _load_head(self):
        import torch

        _CACHE.mkdir(parents=True, exist_ok=True)
        path = _CACHE / "aesthetic-l14-linearMSE.pth"
        if not path.exists():
            for url in self._HEAD_URLS:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                    with urllib.request.urlopen(req, timeout=30) as resp, open(path, "wb") as fh:
                        fh.write(resp.read())
                    break
                except Exception:
                    path.unlink(missing_ok=True)
        if path.exists() and path.stat().st_size > 1000:
            try:
                head = self._build_head()
                state = torch.load(path, map_location="cpu", weights_only=True)
                state = {k[len("layers."):]: v for k, v in state.items()
                         if k.startswith("layers.")} or state
                head.load_state_dict(state)
                head.to(self._device).eval()
                return head
            except Exception as e:
                print(f"[improved-clip] head load failed: {e}", flush=True)
        return None

    def score(self, image: Image.Image) -> tuple:
        import torch

        if not self._ready:
            self.load()
        with torch.no_grad():
            x = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
            feat = self._model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            if self._head is None:
                return float(feat.norm().item()), "缺少权重文件，使用占位分"
            return float(self._head(feat).item()), ""

    def normalize(self, raw: float) -> float:
        # Good anime/AI outputs land ~4.5–7.5.
        return _clamp01((raw - 3.5) / 4.0)


# --------------------------------------------------------------------------- #
# shadow-v2 (anime-specific, default)
# --------------------------------------------------------------------------- #
class ShadowV2Backend(AestheticBackend):
    id = "shadow-v2"
    label = "Aesthetic Shadow V2"
    domain = "动漫专用"
    desc = "1.1B ViT，1024×1024 输入，danbooru 训练，专为评估动漫图质量设计。社区公认质量最高的动漫美观度模型。原作者已下架，使用备份仓库。"
    scale = "hq 概率 0–1"
    approx_size = "~4.4GB"
    _REPO = "NeoChen1024/aesthetic-shadow-v2-backup"

    def __init__(self):
        super().__init__()
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            self.available = True
        except Exception as e:
            self.available = False
            self.reason = f"缺少 torch/transformers：uv sync --extra aesthetic ({e})"
        self._model = self._proc = None
        self._device = "cpu"

    def load(self) -> bool:
        if self._ready:
            return True
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._device = _device()

        # 先把必要文件（含 4.4GB 权重）用多源+断点续传下到本地，再从本地目录加载。
        # 这样即便 transformers 在线探测抽风，也不会报“找不到 preprocessor_config.json”。
        local = _download_repo_files(
            self._REPO,
            ["preprocessor_config.json", "config.json", "model.safetensors"],
            "shadow-v2",
        )
        self._proc = AutoImageProcessor.from_pretrained(str(local))
        model = AutoModelForImageClassification.from_pretrained(
            str(local), low_cpu_mem_usage=True
        )
        self._model = model.to(self._device).eval()
        self._ready = True
        return True

    def score(self, image: Image.Image) -> tuple:
        import torch

        if not self._ready:
            self.load()
        with torch.no_grad():
            inputs = self._proc(images=image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits.float()
            prob = torch.softmax(logits, dim=-1)
            hq = float(prob[0, 0].item())  # label 0 == 'hq'
        return hq, ""

    def normalize(self, raw: float) -> float:
        # hq-probability typically spans ~0.5–0.98; stretch for contrast.
        return _clamp01((raw - 0.45) / 0.5)


# --------------------------------------------------------------------------- #
# skytnt-anime (anime-specific, ONNX, light & fast)
# --------------------------------------------------------------------------- #
class SkytntBackend(AestheticBackend):
    id = "skytnt-anime"
    label = "Skytnt Anime Aesthetic"
    domain = "动漫专用"
    desc = "ConvNeXtV2，ONNX 仅 112MB，danbooru 训练。轻快，CPU/GPU 都快，适合大批量。"
    scale = "score (约 0–1)"
    approx_size = "~112MB"
    _REPO = "skytnt/anime-aesthetic"
    _SIZE = 768

    def __init__(self):
        super().__init__()
        try:
            import onnxruntime  # noqa: F401
            self.available = True
        except Exception as e:
            self.available = False
            self.reason = f"缺少 onnxruntime：uv sync --extra aesthetic ({e})"
        self._sess = None
        self._in_name = "img"

    def load(self) -> bool:
        if self._ready:
            return True
        import onnxruntime as ort

        path = _hf_download(self._REPO, "model.onnx")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(path, providers=providers)
        self._in_name = self._sess.get_inputs()[0].name
        self._ready = True
        return True

    def _preprocess(self, image: Image.Image):
        import numpy as np

        img = image.convert("RGB")
        img.thumbnail((self._SIZE, self._SIZE), Image.LANCZOS)  # aspect-preserving downscale
        w, h = img.size
        canvas = Image.new("RGB", (self._SIZE, self._SIZE), (0, 0, 0))
        canvas.paste(img, ((self._SIZE - w) // 2, (self._SIZE - h) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5                      # [-1, 1]
        arr = arr.transpose(2, 0, 1)[None, ...]      # [1, 3, 768, 768]
        return arr

    def score(self, image: Image.Image) -> tuple:
        if not self._ready:
            self.load()
        out = self._sess.run(None, {self._in_name: self._preprocess(image)})
        return float(out[0][0, 0]), ""

    def normalize(self, raw: float) -> float:
        return _clamp01(raw)


# --------------------------------------------------------------------------- #
# aesthetic-predictor-v2-5 (SigLIP, general + illustrations)
# --------------------------------------------------------------------------- #
class PredictorV25Backend(AestheticBackend):
    id = "aesthetic-predictor-v2-5"
    label = "Aesthetic Predictor V2.5 (SigLIP)"
    domain = "通用+插画"
    desc = "SigLIP，1–10 分，明确支持插画。社区生态最好（pip 包/ComfyUI 节点），维护活跃。5.5+ 算优秀。"
    scale = "1–10 (5.5+ 优秀)"
    approx_size = "~3.6GB (SigLIP)"

    def __init__(self):
        super().__init__()
        try:
            import aesthetic_predictor_v2_5  # noqa: F401
            import torch  # noqa: F401
            self.available = True
        except Exception as e:
            self.available = False
            self.reason = f"缺少 aesthetic-predictor-v2-5/torch：uv sync --extra aesthetic ({e})"
        self._model = self._preprocessor = None
        self._device = "cpu"

    def load(self) -> bool:
        if self._ready:
            return True
        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        self._device = _device()

        def _do():
            return convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)

        model, preprocessor = _load_with_endpoint_fallback(_do, "aesthetic-predictor-v2-5")
        self._model = model.to(self._device).eval()
        self._preprocessor = preprocessor
        self._ready = True
        return True

    def score(self, image: Image.Image) -> tuple:
        import torch

        if not self._ready:
            self.load()
        inputs = self._preprocessor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            score = float(self._model(pixel_values).logits.squeeze().cpu().numpy())
        return score, ""

    def normalize(self, raw: float) -> float:
        return _clamp01((raw - 1.0) / 9.0)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_backends() -> dict:
    backends = {}
    for cls in (ShadowV2Backend, SkytntBackend, PredictorV25Backend, ImprovedClipBackend):
        try:
            b = cls()
        except Exception as e:  # backstop
            b = cls.__new__(cls)
            AestheticBackend.__init__(b)
            b.available = False
            b.reason = f"初始化失败：{e}"
        backends[b.id] = b
    return backends
