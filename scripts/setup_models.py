"""One-shot model bootstrap — everything local, nothing system-wide.

Downloads all perception models into the HuggingFace cache (reuse if present)
or the project's ``.models/`` directory (gitignored). No installer, no service,
no PATH changes: deleting ``.models/`` + the HF cache entries fully cleans up.

Usage (from repo root, with the project venv active)::

    python scripts/setup_models.py                 # everything
    python scripts/setup_models.py --only vlm      # just the VLM runtime+weights
    python scripts/setup_models.py --only detector,parts,ccip,pose
    python scripts/setup_models.py --quant Q8_0    # heavier VLM quant (A/B)

Components:
    pose      DWPose (yolox_l COCO + dw-ll_ucoco_384)       ~330 MB
    detector  anime person YOLOv8m v1.1_m + HumanArt YOLOX-L ~310 MB
    parts     ADetailer hand_yolov9c + face_yolov8n_v2       ~58 MB (+ONNX export)
    ccip      CCIP caformer_b36-24 feature model             ~384 MB
    vlm       portable llama-server (Vulkan) + Qwen3-VL-8B   ~6.3 GB
              abliterated GGUF (Q4_K_M) + mmproj-f16

Network: HF downloads honour HF_ENDPOINT (default https://hf-mirror.com for
CN users; set HF_ENDPOINT=https://huggingface.co elsewhere). GitHub /
openmmlab downloads are direct.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / ".models"
UA = {"User-Agent": "Mozilla/5.0 (model-bootstrap)"}

# Pinned llama.cpp release (portable Vulkan build, ~38 MB, zero dependencies).
LLAMA_TAG = "b12322"
LLAMA_ZIP = f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{LLAMA_ZIP}"

VLM_REPO = "mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF"
VLM_BASE = "Huihui-Qwen3-VL-8B-Instruct-abliterated"
HUMANART_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a62.zip")


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def hf_file(repo: str, filename: str) -> Path:
    """Cache-aware HF download. Prefers huggingface_hub (resumable, shares the
    user's existing cache); falls back to plain urllib into .models/hub."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo, filename=filename, endpoint=_hf_endpoint())
        return Path(p)
    except ImportError:
        pass
    except Exception as e:
        print(f"  hub failed ({type(e).__name__}: {str(e)[:80]}), raw fallback…", flush=True)
    dest = MODELS / "hub" / repo.replace("/", "--") / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for ep in (_hf_endpoint(), "https://huggingface.co"):
        url = f"{ep}/{repo}/resolve/main/{filename}"
        print(f"  GET {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:
            print(f"  failed: {type(e).__name__} {str(e)[:80]}", flush=True)
    raise RuntimeError(f"download failed: {repo}/{filename}")


def http_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers=UA)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
def setup_pose() -> None:
    print("[pose] DWPose", flush=True)
    for fn in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        p = hf_file("yzd-v/DWPose", fn)
        print(f"  ok {p.name} ({p.stat().st_size/1e6:.0f} MB)", flush=True)


def setup_detector() -> None:
    print("[detector] anime person YOLOv8m v1.1_m (primary)", flush=True)
    for fn in ("model.onnx", "labels.json", "threshold.json"):
        p = hf_file("deepghs/anime_person_detection", f"person_detect_v1.1_m/{fn}")
        print(f"  ok {fn}", flush=True)
    print("[detector] HumanArt YOLOX-L (A/B shadow)", flush=True)
    out = MODELS / "humanart" / "yolox_l_humanart.onnx"
    if not out.exists():
        zp = http_file(HUMANART_URL, MODELS / "humanart" / "yolox_l_humanart.zip")
        with zipfile.ZipFile(zp) as z:
            onnx = [n for n in z.namelist() if n.endswith(".onnx")]
            if not onnx:
                raise RuntimeError(f"no .onnx in {zp}: {z.namelist()}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(onnx[0]))
        print(f"  extracted {onnx[0]} -> {out}", flush=True)
    print(f"  ok {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)


def setup_parts() -> None:
    print("[parts] ADetailer hand/face detectors", flush=True)
    pts = {}
    for fn in ("hand_yolov9c.pt", "face_yolov8n_v2.pt"):
        pts[fn] = hf_file("Bingsu/adetailer", fn)
        print(f"  ok {fn}", flush=True)
    # One-time ONNX export so runtime needs only onnxruntime (GPU via DirectML).
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [skip] ultralytics not installed — runtime will use .pt directly", flush=True)
        return
    onnx_dir = MODELS / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for fn, pt in pts.items():
        out = onnx_dir / (Path(fn).stem + ".onnx")
        if out.exists():
            print(f"  ok {out.name} (cached)", flush=True)
            continue
        print(f"  exporting {fn} -> ONNX ...", flush=True)
        exported = YOLO(str(pt)).export(format="onnx", imgsz=640, opset=12, device="cpu")
        Path(exported).replace(out)
        print(f"  ok {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)


def setup_ccip() -> None:
    print("[ccip] caformer_b36-24 feature model", flush=True)
    for fn in ("model_feat.onnx", "metrics.json", "cluster.json"):
        hf_file("deepghs/ccip_onnx", f"ccip-caformer_b36-24/{fn}")
    print("  ok", flush=True)


def setup_vlm(quant: str) -> None:
    print(f"[vlm] portable llama-server ({LLAMA_TAG}, Vulkan)", flush=True)
    bin_dir = MODELS / "bin" / "llama"
    exe = bin_dir / "llama-server.exe"
    if not exe.exists():
        zp = http_file(LLAMA_URL, MODELS / "bin" / LLAMA_ZIP)
        with zipfile.ZipFile(zp) as z:
            z.extractall(bin_dir)
        # some builds nest everything under a top-level folder
        if not exe.exists():
            nested = list(bin_dir.rglob("llama-server.exe"))
            if nested:
                for f in nested[0].parent.iterdir():
                    f.replace(bin_dir / f.name)
    if not exe.exists():
        raise RuntimeError(f"llama-server.exe not found after extracting {LLAMA_ZIP}")
    print(f"  ok {exe}", flush=True)

    print(f"[vlm] Qwen3-VL-8B abliterated GGUF ({quant} + mmproj-f16)", flush=True)
    gguf = hf_file(VLM_REPO, f"{VLM_BASE}.{quant}.gguf")
    mmproj = hf_file(VLM_REPO, f"{VLM_BASE}.mmproj-f16.gguf")
    print(f"  ok {gguf.name} ({gguf.stat().st_size/1e9:.2f} GB)", flush=True)
    print(f"  ok {mmproj.name} ({mmproj.stat().st_size/1e9:.2f} GB)", flush=True)
    # Record resolved paths for the runtime (avoids re-scanning the HF cache).
    import json
    (MODELS / "vlm_paths.json").write_text(json.dumps({
        "server_exe": str(exe), "gguf": str(gguf), "mmproj": str(mmproj),
        "quant": quant, "llama_tag": LLAMA_TAG,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {MODELS / 'vlm_paths.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="all",
                    help="comma list: pose,detector,parts,ccip,vlm")
    ap.add_argument("--quant", default="Q4_K_M",
                    help="VLM GGUF quant (Q4_K_M default; Q8_0 for A/B)")
    args = ap.parse_args()
    # Force the CN-friendly mirror: a pre-existing HF_ENDPOINT=huggingface.co
    # in the environment would otherwise win and time out behind the GFW.
    os.environ["HF_ENDPOINT"] = os.environ.get("AIC_HF_ENDPOINT") or "https://hf-mirror.com"
    want = set(args.only.split(",")) if args.only != "all" else {
        "pose", "detector", "parts", "ccip", "vlm"}
    MODELS.mkdir(exist_ok=True)
    if "pose" in want:
        setup_pose()
    if "detector" in want:
        setup_detector()
    if "parts" in want:
        setup_parts()
    if "ccip" in want:
        setup_ccip()
    if "vlm" in want:
        setup_vlm(args.quant)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
