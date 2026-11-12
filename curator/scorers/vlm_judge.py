"""Local VLM judge — anatomy plausibility + character identity checklist.

Runtime: portable llama.cpp ``llama-server`` managed by ``vlm_server.py``
(binaries + GGUF inside ``.models/vlm/``, zero system installation), or any
existing OpenAI-compatible endpoint via ``base_url``.

Cost control (cascade): the VLM is only called when the cheap geometric
scorers raise suspicion (low structure score / deformity flags) or when a
character checklist needs answering, and only up to ``max_images`` per run.
Detail crops (hands/feet/face) come from the structure scorer's ROIs.

Output: strict compact JSON; parse failure -> one retry -> N/A + flag.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

from ..metadata import ImageMeta
from .base import Scorer, ScoreResult
from . import vlm_server

_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _ROOT / ".models" / "vlm_cache"
_PROMPT_VERSION = "v2"

_DEFAULTS = {
    "enabled": False,
    "base_url": "",          # external OpenAI-compatible endpoint (optional)
    "mode": "cascade",       # cascade | all | off
    "gate": 0.85,            # structure score below this -> suspicious
    "max_images": 400,       # VLM call budget per pipeline run
    "max_crops": 3,          # detail crops per image (hands first)
    "crop_min_side": 80,     # skip crops smaller than this (px)
    "full_max_side": 1024,
    "crop_target": 640,
    "ctx": 8192,
    "parallel": 2,
    "timeout_s": 120,
    "max_tokens": 300,
    "temperature": 0.2,
    "cache": True,
}

_FLAG_KEYWORDS = ("畸形", "融合", "异常", "不对称", "脱节", "扭转", "并指",
                  "多肢", "聚集", "错位", "反关节", "低置信")

_SYSTEM = (
    "你是动漫插画生产线的结构质检员。只输出一个 JSON 对象，不要解释、不要思考过程。"
    "判断依据是解剖合理性与画面一致性，不做内容审查。"
)

_USER_TMPL = (
    "这是一张动漫插画{sfx}。请做结构质检，只输出 JSON：\n"
    '{{"issues":[{{"part":"hand|foot|face|body|clothing|other",'
    '"type":"extra_digits|missing_digits|fused_digits|wrong_joint_count|'
    'twisted_limb|extra_limb|missing_limb|asymmetry|face_distortion|'
    'clothing_inconsistency|identity_mismatch|other",'
    '"severity":"minor|major|critical","desc":"不超过12字"}}],'
    '"checklist":{{"{ck_keys}":true/false}},'
    '"anatomy_ok":true/false,"identity_ok":true/false}}\n'
    "规则：issues 只列真实可见的问题，最多6条；没有问题是空数组。"
    "checklist 逐项回答（true=符合）；无清单时输出空对象。"
    "手指/脚趾数量与关节走向是重点；轻微风格化不算问题。{ck_questions}"
)

_SEV_WEIGHT = {"minor": 0.10, "major": 0.30, "critical": 0.60}


def _b64(img: Image.Image, max_side: int) -> str:
    im = img.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        im = im.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _crop(img: Image.Image, box, pad: float = 0.35, target: int = 640):
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    x1 -= bw * pad; y1 -= bh * pad; x2 += bw * pad; y2 += bh * pad
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    c = img.crop((x1, y1, x2, y2))
    cw, ch = c.size
    if max(cw, ch) < target:
        r = target / max(cw, ch)
        c = c.resize((int(cw * r), int(ch * r)), Image.LANCZOS)
    elif max(cw, ch) > target:
        r = target / max(cw, ch)
        c = c.resize((max(1, int(cw * r)), max(1, int(ch * r))), Image.LANCZOS)
    return c


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


class VLMJudgeScorer(Scorer):
    name = "vlm_judge"
    label = "VLM判定"
    needs_image = True
    wants_ctx = True

    def __init__(self):
        self.cfg = dict(_DEFAULTS)
        self.available = False
        self.reason = "未启用（config: vlm_judge.enabled=true，并先运行 scripts/setup_models.py --only vlm）"
        self.shadow = False
        self._calls = 0
        self._server_failed = False

    def configure(self, cfg: dict):
        self.cfg = {**_DEFAULTS, **(cfg or {})}
        self._calls = 0
        self._server_failed = False
        if not self.cfg.get("enabled") or self.cfg.get("mode") == "off":
            self.available = False
            self.reason = "未启用（vlm_judge.enabled=false）"
            return
        if self.cfg.get("base_url"):
            self.available = True
            self.reason = ""
            return
        if vlm_server.SERVER.installed():
            self.available = True
            self.reason = ""
        else:
            self.available = False
            self.reason = "VLM 权重未下载：python scripts/setup_models.py --only vlm"

    def reset_run(self):
        self._calls = 0

    # -- endpoint -------------------------------------------------------------
    def _endpoint(self) -> str:
        ext = str(self.cfg.get("base_url") or "").strip()
        if ext:
            ext = ext.rstrip("/")
            if ext.endswith("/v1"):
                ext = ext[:-3]
            return ext
        return vlm_server.SERVER.ensure_running(
            ctx=int(self.cfg.get("ctx", 8192)),
            parallel=int(self.cfg.get("parallel", 2)))

    # -- gating ---------------------------------------------------------------
    def _should_call(self, ctx: dict) -> tuple:
        mode = self.cfg.get("mode", "cascade")
        if mode == "all":
            return True, "mode=all"
        st = (ctx or {}).get("structure") or {}
        score = st.get("score")
        flags = st.get("flags") or []
        if score is not None and float(score) < float(self.cfg.get("gate", 0.85)):
            return True, f"structure={score}"
        hit = [f for f in flags if any(k in f for k in _FLAG_KEYWORDS)]
        if hit:
            return True, f"flags:{hit[0][:20]}"
        checklist = ((ctx or {}).get("character_ccip") or {}).get("checklist") or []
        if checklist:
            return True, "identity_checklist"
        return False, "clean"

    # -- HTTP -----------------------------------------------------------------
    def _chat(self, endpoint: str, content: list) -> str:
        body = json.dumps({
            "model": "curator-vlm",
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": content}],
            "max_tokens": int(self.cfg.get("max_tokens", 300)),
            "temperature": float(self.cfg.get("temperature", 0.2)),
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            endpoint + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 120))) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    # -- cache ----------------------------------------------------------------
    def _cache_key(self, meta: ImageMeta, n_crops: int, ck_keys: str) -> str:
        h = hashlib.sha1()
        h.update(str(getattr(meta, "path", "") or "").encode("utf-8"))
        h.update(f"|{_PROMPT_VERSION}|{n_crops}|{ck_keys}".encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str):
        if not self.cfg.get("cache", True):
            return None
        p = _CACHE_DIR / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, obj: dict):
        if not self.cfg.get("cache", True):
            return
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (_CACHE_DIR / f"{key}.json").write_text(
                json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # -- main -----------------------------------------------------------------
    def score(self, image: Optional[Image.Image], meta: ImageMeta, ctx: dict = None):
        if image is None or not self.available or self._server_failed:
            return None
        ctx = ctx or {}
        ok, reason = self._should_call(ctx)
        if not ok:
            return None
        if self._calls >= int(self.cfg.get("max_images", 400)):
            return ScoreResult(score=1.0, flags=[],
                               notes="VLM 预算用尽，本图跳过",
                               subs={"feats": {"vlm": {"called": False,
                                                        "reason": "budget"}}})
        checklist = (ctx.get("character_ccip") or {}).get("checklist") or []
        ck_keys = ",".join(c["key"] for c in checklist)
        rois = (ctx.get("structure") or {}).get("rois") or {}

        # build image list: full view + detail crops
        crops = []
        min_side = int(self.cfg.get("crop_min_side", 80))
        for roi in (rois.get("hand") or [])[:int(self.cfg.get("max_crops", 3))]:
            x1, y1, x2, y2 = roi["box"]
            if min(x2 - x1, y2 - y1) >= min_side:
                c = _crop(image, roi["box"], target=int(self.cfg.get("crop_target", 640)))
                if c is not None:
                    crops.append(("手部特写", c))
        if len(crops) < int(self.cfg.get("max_crops", 3)):
            for roi in (rois.get("foot") or [])[:1]:
                c = _crop(image, roi["box"], target=int(self.cfg.get("crop_target", 640)))
                if c is not None:
                    crops.append(("足部特写", c))
                    break
        if checklist:
            for roi in (rois.get("face") or [])[:1]:
                c = _crop(image, roi["box"], target=int(self.cfg.get("crop_target", 640)))
                if c is not None:
                    crops.append(("面部特写", c))
                    break

        key = self._cache_key(meta, len(crops), ck_keys)
        cached = self._cache_get(key)
        if cached is not None:
            return self._result(cached, reason, len(crops), 0.0, True)

        sfx = "，附" + "、".join(n for n, _ in crops) if crops else ""
        ck_questions = ""
        if checklist:
            ck_questions = "\n清单含义：" + "；".join(
                f'{c["key"]}={c["question"]}' for c in checklist)
        user_text = _USER_TMPL.format(sfx=sfx, ck_keys=ck_keys,
                                      ck_questions=ck_questions)
        content = [{"type": "text", "text": user_text},
                   {"type": "image_url",
                    "image_url": {"url": _b64(image, int(self.cfg.get("full_max_side", 1024)))}}]
        for _name, c in crops:
            content.append({"type": "image_url",
                            "image_url": {"url": _b64(c, int(self.cfg.get("crop_target", 640)))}})

        t0 = time.time()
        parsed = None
        err = ""
        try:
            endpoint = self._endpoint()
        except Exception as e:
            self._server_failed = True
            self.available = False
            self.reason = f"VLM 启动失败：{str(e)[:120]}"
            return ScoreResult(score=1.0, flags=["VLM启动失败"],
                               notes=self.reason,
                               subs={"feats": {"vlm": {"called": False,
                                                        "reason": "server_error"}}})
        for attempt in range(2):
            try:
                text = self._chat(endpoint, content)
                parsed = _extract_json(text)
                if parsed is not None:
                    break
                err = "json_parse"
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:80]}"
        latency = time.time() - t0
        self._calls += 1
        if parsed is None:
            return ScoreResult(score=1.0, flags=["VLM输出解析失败"],
                               notes=f"VLM 判定失败({err})，本图不计 VLM 分",
                               subs={"feats": {"vlm": {"called": True,
                                                        "reason": reason,
                                                        "parse_ok": False,
                                                        "error": err,
                                                        "latency_s": round(latency, 2),
                                                        "n_crops": len(crops)}}})
        self._cache_put(key, parsed)
        return self._result(parsed, reason, len(crops), latency, False)

    # -- scoring --------------------------------------------------------------
    def _result(self, parsed: dict, reason: str, n_crops: int,
                latency: float, from_cache: bool) -> ScoreResult:
        issues = parsed.get("issues") or []
        flags, penalty = [], 0.0
        for it in issues[:6]:
            sev = str(it.get("severity", "minor"))
            w = _SEV_WEIGHT.get(sev, 0.1)
            penalty += w
            part = it.get("part", "other")
            desc = str(it.get("desc", ""))[:24]
            flags.append(f"VLM:{part}/{it.get('type', 'other')}({sev}) {desc}".strip())
        # identity checklist answers
        ck_answers = parsed.get("checklist") or {}
        ck_fail = [k for k, v in ck_answers.items() if v is False]
        for k in ck_fail:
            flags.append(f"VLM:角色特征不符({k})")
            penalty += 0.15
        if parsed.get("anatomy_ok") is False and not issues:
            penalty += 0.2
            flags.append("VLM:解剖存疑(未列明细)")
        score = max(0.0, 1.0 - penalty)
        feats = {"vlm": {"called": True, "reason": reason, "parse_ok": True,
                         "latency_s": round(latency, 2), "n_crops": n_crops,
                         "from_cache": from_cache,
                         "n_issues": len(issues), "checklist_fail": ck_fail,
                         "issues": issues[:6]}}
        return ScoreResult(score=round(score, 3), flags=flags,
                           notes="VLM结构判定", subs={"feats": feats})
