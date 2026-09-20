"""Structure scorer — anatomy sanity via DWPose keypoints + optional YOLO plate.

Design (see docs/STRUCTURE_SCORER.md):

  Stage 1 "checklist"  : which body parts are VISIBLE? (DWPose conf + YOLO dets)
  Stage 2 "conditions" : geometric rule checks run ONLY on visible parts.
                         Not detected -> N/A (no penalty, no bonus).

Three sub-scores, each may be N/A:
  - body  : COCO17 skeleton rules (fused joints, limb asymmetry, neck, ...)
  - hands : 21-keypoint geometry per visible hand (finger ratios, tip order,
            clustering) — the main "deformed hand" signal
  - yolo  : a PLATE of up to 4 YOLO models; the plate score is the plain
            average of per-model scores. Per-model score = min confidence
            over its top-k confident detections (worst visible instance of
            that part); no detection -> N/A (excluded from the average).

Overall structure score = weighted average of AVAILABLE parts only.
If nothing is available -> score() returns None and the pipeline skips this
scorer entirely (composite renormalizes over the remaining scorers).

SHADOW MODE (structure.shadow=true, default): the pipeline records the score
+ flags for display and feedback collection, but EXCLUDES it from the
composite until thresholds are calibrated on real user data.

All numeric features are persisted in `subs.feats` so the feedback JSONL
(docs/FEEDBACK_FORMAT.md) contains a complete feature row per image.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from ..metadata import ImageMeta
from .base import Scorer, ScoreResult
from . import dwpose_infer as dw

STRUCTURE_RULES_VERSION = "v1"

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".models"

# ---- skeleton topology for overlay rendering ------------------------------- #
BODY_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
              (13, 15), (12, 14), (14, 16)]
FEET_EDGES = [(15, 17), (15, 19), (16, 20), (16, 22)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

_DEFAULTS = {
    "shadow": True,
    "parts": {"body": 0.35, "hands": 0.35, "yolo": 0.30},
    "pose": {"enabled": True, "det_conf": 0.30, "max_persons": 3,
             "body_vis": 0.35, "hand_vis": 0.40, "hand_min_palm": 0.035},
    "yolo": {"vis_floor": 0.35, "top_k": 4, "models": []},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def _angle(a, b, c) -> float:
    """Angle at vertex b (degrees) formed by segments b->a and b->c."""
    v1 = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    v2 = np.asarray(c, np.float32) - np.asarray(b, np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Per-model YOLO wrapper
# --------------------------------------------------------------------------- #
class _YoloModel:
    """One entry of the YOLO plate. `source` is a local path or "repo:file"."""

    def __init__(self, cfg: dict):
        self.id = str(cfg.get("id") or "yolo")
        self.label = str(cfg.get("label") or self.id)
        self.source = str(cfg.get("source") or "")
        self.available = bool(self.source)
        self.reason = "" if self.available else "缺少 source"
        self._model = None
        self._lock = threading.Lock()

    def _resolve_path(self) -> str:
        p = Path(self.source)
        if p.exists():
            return str(p)
        if ":" in self.source:                  # "repo_id:filename" -> HF download
            repo, fn = self.source.split(":", 1)
            from .aesthetic_backends import _hf_download
            return _hf_download(repo, fn)
        raise RuntimeError(f"无法解析 YOLO 模型来源: {self.source}（支持本地路径或 repo:file）")

    def load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._resolve_path())
        return self._model

    def detect(self, img: Image.Image, vis_floor: float, top_k: int) -> Optional[dict]:
        """Returns {"n_det","max_conf","score"} or None (= part not visible)."""
        if not self.available:
            return None
        try:
            model = self.load()
            with self._lock:
                res = model.predict(img, imgsz=640,
                                    conf=max(0.15, vis_floor * 0.6), verbose=False)
            confs = []
            for r in res:
                if r.boxes is not None and len(r.boxes):
                    confs.extend(float(c) for c in r.boxes.conf.cpu().numpy())
        except Exception as e:
            self.available = False
            self.reason = f"运行失败: {type(e).__name__} {str(e)[:80]}"
            return None
        vis = sorted([c for c in confs if c >= vis_floor], reverse=True)
        if not vis:
            return None
        top = vis[:max(1, top_k)]
        return {"n_det": len(vis), "max_conf": round(vis[0], 3),
                "score": round(min(top), 3)}


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #
class StructureScorer(Scorer):
    name = "structure"
    label = "结构"
    needs_image = True

    def __init__(self):
        self.cfg = dict(_DEFAULTS)
        self.shadow = True
        self._runner = None
        self._runner_lock = threading.Lock()
        self._yolos: list = []
        self.reason = ""
        try:
            import onnxruntime  # noqa: F401
            self.available = True
        except Exception:
            self.available = False
            self.reason = "缺少 onnxruntime：uv sync --extra structure"

    # ---- config -------------------------------------------------------------
    def configure(self, cfg: dict):
        self.cfg = _deep_merge(_DEFAULTS, cfg or {})
        self.shadow = bool(self.cfg.get("shadow", True))
        models = (self.cfg.get("yolo") or {}).get("models") or []
        self._yolos = [_YoloModel(m) for m in models[:4]]   # hard cap: 4 models

    # ---- model lifecycle ----------------------------------------------------
    def _ensure_runner(self):
        with self._runner_lock:
            if self._runner is None:
                pcfg = self.cfg.get("pose") or {}
                det, pose = dw.ensure_models(_CACHE_DIR)
                self._runner = dw.DWPoseRunner(
                    det, pose,
                    det_conf=float(pcfg.get("det_conf", 0.30)),
                    max_persons=int(pcfg.get("max_persons", 3)),
                )
            return self._runner

    def prewarm(self):
        try:
            self._ensure_runner()
        except Exception as e:
            self.available = False
            self.reason = f"DWPose 加载失败：{e}"
            return
        for y in self._yolos:
            try:
                y.load()
            except Exception as e:
                y.available = False
                y.reason = f"加载失败: {type(e).__name__} {str(e)[:60]}"

    @property
    def pose_ready(self) -> bool:
        return self._runner is not None

    # ---- main entry ---------------------------------------------------------
    def score(self, image: Optional[Image.Image], meta: ImageMeta):
        if image is None or not self.available:
            return None
        pcfg = self.cfg.get("pose") or {}
        ycfg = self.cfg.get("yolo") or {}

        persons, pose_err = [], ""
        if pcfg.get("enabled", True):
            try:
                persons = self._ensure_runner().infer(image)
            except Exception as e:
                pose_err = f"{type(e).__name__} {str(e)[:80]}"
                self.available = False
                self.reason = f"DWPose 推理失败：{pose_err}"

        flags: list = []
        feats: dict = {"rules_version": STRUCTURE_RULES_VERSION,
                       "n_persons": len(persons), "pose_error": pose_err}
        subs: dict = {}

        # ---- body + hands from DWPose ---------------------------------------
        body_scores, hand_scores = [], []
        if persons:
            for pi, p in enumerate(persons):
                tag = f"人{pi+1}" if len(persons) > 1 else ""
                bs, bflags, _bf = self._check_body(p, feats, tag)
                body_scores.append(bs)
                flags.extend(bflags)
                for side in ("l", "r"):
                    hs, hflags, _hf = self._check_hand(p, side, pcfg, feats, tag)
                    if hs is not None:
                        hand_scores.append(hs)
                        flags.extend(hflags)
        subs["body"] = round(float(np.min(body_scores)), 3) if body_scores else None
        subs["hands"] = round(float(np.mean(hand_scores)), 3) if hand_scores else None

        # ---- YOLO plate (average of per-model scores; N/A models excluded) --
        plate_scores, yolo_info = [], []
        vis_floor = float(ycfg.get("vis_floor", 0.35))
        top_k = int(ycfg.get("top_k", 4))
        for y in self._yolos:
            r = y.detect(image, vis_floor, top_k)
            info = {"id": y.id, "label": y.label, "available": y.available}
            if not y.available:
                info["reason"] = y.reason
            elif r is None:
                info["detected"] = False
            else:
                info.update({"detected": True, **r})
                plate_scores.append(r["score"])
                if r["score"] < 0.5:
                    flags.append(f"{y.label}低置信({r['score']:.2f})")
            yolo_info.append(info)
        subs["yolo"] = round(float(np.mean(plate_scores)), 3) if plate_scores else None
        feats["yolo_models"] = yolo_info

        # ---- visibility checklist (for UI + feedback analysis) ---------------
        hands_f = feats.get("hands", [])
        body_f = feats.get("body", [])
        feats["visible"] = {
            "person": bool(persons),
            "hand_l": any(h.get("side") == "l" and h.get("visible") for h in hands_f),
            "hand_r": any(h.get("side") == "r" and h.get("visible") for h in hands_f),
            "feet": any(b.get("feet_conf", 0.0) >= 0.4 for b in body_f),
            "yolo_parts": [i["id"] for i in yolo_info if i.get("detected")],
        }

        # ---- aggregate over AVAILABLE parts only -----------------------------
        weights = self.cfg.get("parts") or {}
        num = den = 0.0
        for part in ("body", "hands", "yolo"):
            v = subs.get(part)
            if v is not None:
                w = float(weights.get(part, 0.0))
                num += w * float(v)
                den += w
        if den <= 0:
            return None           # nothing measurable -> N/A, skip entirely
        overall = num / den

        feats["parts"] = {k: subs.get(k) for k in ("body", "hands", "yolo")}
        subs["overall"] = round(overall, 3)
        subs["feats"] = feats
        subs["visible"] = feats["visible"]
        return ScoreResult(score=round(overall, 3), subs=subs, flags=flags,
                           notes="shadow" if self.shadow else "")

    # ---- body rules ---------------------------------------------------------
    def _check_body(self, p: dict, feats: dict, tag: str):
        """COCO17 skeleton sanity. Returns (score, flags, feats_sub)."""
        pcfg = self.cfg.get("pose") or {}
        vis = float(pcfg.get("body_vis", 0.35))
        k, ks = p["kpts"], p["kscores"]

        def seen(*idxs):
            return all(ks[i] >= vis for i in idxs)

        flags = []
        score = 1.0
        f = {}

        torso = None
        if seen(dw.L_SH, dw.R_SH, dw.L_HIP, dw.R_HIP):
            sh_mid = (k[dw.L_SH] + k[dw.R_SH]) / 2
            hip_mid = (k[dw.L_HIP] + k[dw.R_HIP]) / 2
            torso = _dist(sh_mid, hip_mid)
            sh_w = _dist(k[dw.L_SH], k[dw.R_SH])
            f["torso_px"] = round(torso, 1)
            f["shoulder_w_px"] = round(sh_w, 1)
            # neck ratio only meaningful when shoulder width is plausible
            if torso > 0 and sh_w > 0.25 * torso:
                f["neck_ratio"] = round(_dist(k[dw.NOSE], sh_mid) / sh_w, 2)

        f["body_conf_mean"] = round(float(ks[:17].mean()), 3)
        f["feet_conf"] = round(float(ks[dw.FEET[0]:dw.FEET[1]].mean()), 3)
        if float(ks[:17].mean()) < 0.25:
            flags.append(f"{tag}骨架整体低置信")

        # fused shoulders / wrists (multi-limb collapse signature)
        if torso and seen(dw.L_SH, dw.R_SH):
            if _dist(k[dw.L_SH], k[dw.R_SH]) < 0.15 * torso:
                flags.append(f"{tag}双肩融合(疑似多肢)")
                score -= 0.4
        if torso and seen(dw.L_WR, dw.R_WR):
            if _dist(k[dw.L_WR], k[dw.R_WR]) < 0.06 * torso:
                flags.append(f"{tag}双腕融合(疑似多肢)")
                score -= 0.3

        # limb asymmetry (upper arm + forearm; thigh + shin)
        for name, left, right in [
            ("手臂", (dw.L_SH, dw.L_EL, dw.L_WR), (dw.R_SH, dw.R_EL, dw.R_WR)),
            ("腿", (dw.L_HIP, dw.L_KN, dw.L_AN), (dw.R_HIP, dw.R_KN, dw.R_AN)),
        ]:
            ls, le, lw = left
            rs, re, rw = right
            if seen(ls, le, lw) and seen(rs, re, rw):
                llen = _dist(k[ls], k[le]) + _dist(k[le], k[lw])
                rlen = _dist(k[rs], k[re]) + _dist(k[re], k[rw])
                ratio = max(llen, rlen) / max(min(llen, rlen), 1e-3)
                f[f"{name}_asym"] = round(ratio, 2)
                if ratio > 1.35:
                    flags.append(f"{tag}{name}左右长度不对称({ratio:.2f})")
                    score -= 0.25

        # joint fusion (zero-length segments)
        if torso:
            for (a, b, nm) in [(dw.L_SH, dw.L_EL, "左"), (dw.R_SH, dw.R_EL, "右"),
                               (dw.L_HIP, dw.L_KN, "左"), (dw.R_HIP, dw.R_KN, "右")]:
                if seen(a, b) and _dist(k[a], k[b]) < 0.05 * torso:
                    flags.append(f"{tag}{nm}侧关节融合")
                    score -= 0.2

        # neck extremes (conservative; anime proportions vary a lot)
        if f.get("neck_ratio", 0) > 1.6:
            flags.append(f"{tag}颈部比例异常({f['neck_ratio']:.2f})")
            score -= 0.15

        score = max(0.0, score)
        f["body_score"] = round(score, 3)
        feats.setdefault("body", []).append(f)
        return score, flags, f

    # ---- hand rules ---------------------------------------------------------
    def _check_hand(self, p: dict, side: str, pcfg: dict, feats: dict, tag: str):
        """21-keypoint hand geometry. side: 'l'|'r'.

        Returns (score|None, flags, feats_sub). None = hand not visible or too
        small to judge -> N/A (no penalty, no bonus).
        """
        base = dw.HAND_L[0] if side == "l" else dw.HAND_R[0]
        k, ks = p["kpts"][base:base + 21], p["kscores"][base:base + 21]
        side_cn = "左手" if side == "l" else "右手"

        f = {"side": side, "conf_mean": round(float(ks.mean()), 3)}
        vis_conf = float(pcfg.get("hand_vis", 0.40))
        if float(ks.mean()) < vis_conf:
            f["visible"] = False
            feats.setdefault("hands", []).append(f)
            return None, [], f

        # scale-invariant size guard: palm (wrist->middle MCP) vs person box height
        palm = _dist(k[0], k[9])
        box = p.get("box") or [0, 0, 1, 1]
        person_h = max(float(box[3] - box[1]), 1.0)
        min_palm = float(pcfg.get("hand_min_palm", 0.035))
        if palm < min_palm * person_h:
            f["visible"] = True
            f["too_small"] = True
            feats.setdefault("hands", []).append(f)
            return None, [], f
        f["visible"] = True

        flags = []
        score = 1.0
        f["palm_px"] = round(palm, 1)

        # finger segment layout: (mcp, pip, dip, tip)
        fingers = {"拇指": (1, 2, 3, 4), "食指": (5, 6, 7, 8),
                   "中指": (9, 10, 11, 12), "无名指": (13, 14, 15, 16),
                   "小指": (17, 18, 19, 20)}

        # 1) fingertip order: middle finger should reach farthest from wrist
        tips = {n: _dist(k[0], k[i[3]]) for n, i in fingers.items() if n != "拇指"}
        f["tip_dists"] = {n: round(v, 1) for n, v in tips.items()}
        if tips and tips["小指"] > tips["中指"] * 1.02:
            flags.append(f"{tag}{side_cn}手指长度顺序异常(小指>中指)")
            score -= 0.3

        # 2) segment proportions: distal segment shouldn't exceed proximal
        bad_seg = 0
        for n, (m, pi, di, t) in fingers.items():
            if n == "拇指":
                continue
            prox = _dist(k[m], k[pi])
            dist_seg = _dist(k[di], k[t])
            if prox > 1e-3 and dist_seg > prox * 1.6:
                bad_seg += 1
        f["bad_segments"] = bad_seg
        if bad_seg >= 2:
            flags.append(f"{tag}{side_cn}{bad_seg}根手指指节比例异常")
            score -= 0.25 * min(bad_seg - 1, 2)

        # 3) fingertip clustering (webbed/fused fingers signature)
        tips_xy = [k[i[3]] for n, i in fingers.items() if n != "拇指"]
        clustered = 0
        for a in range(len(tips_xy)):
            for b in range(a + 1, len(tips_xy)):
                if _dist(tips_xy[a], tips_xy[b]) < 0.10 * palm:
                    clustered += 1
        f["tip_clusters"] = clustered
        if clustered >= 1:
            flags.append(f"{tag}{side_cn}指尖聚集(并指嫌疑)")
            score -= 0.25 * min(clustered, 2)

        # 4) extreme joint fold (angle at PIP below 30 degrees)
        for n, (m, pi, di, t) in fingers.items():
            if n == "拇指":
                continue
            ang = _angle(k[m], k[pi], k[di])
            if ang < 30:
                f[f"fold_angle_{n}"] = round(ang, 0)
                flags.append(f"{tag}{side_cn}{n}关节角度异常({ang:.0f}°)")
                score -= 0.25
                break

        score = max(0.0, score)
        f["hand_score"] = round(score, 3)
        feats.setdefault("hands", []).append(f)
        return score, flags, f

    # ---- overlay rendering ---------------------------------------------------
    def render_overlay(self, img: Image.Image) -> Image.Image:
        """Draw skeleton + hands + feet on a copy of the image (for the UI)."""
        img = img.convert("RGB").copy()
        persons = self._ensure_runner().infer(img)
        d = ImageDraw.Draw(img, "RGBA")
        lw = max(2, min(img.size) // 300)
        vis = float((self.cfg.get("pose") or {}).get("body_vis", 0.35))
        for p in persons:
            k, ks = p["kpts"], p["kscores"]
            d.rectangle(p["box"], outline=(255, 255, 255, 160), width=max(1, lw // 2))
            for a, b in BODY_EDGES:
                if ks[a] >= vis and ks[b] >= vis:
                    d.line([tuple(k[a]), tuple(k[b])], fill=(0, 255, 110, 220), width=lw)
            for a, b in FEET_EDGES:
                if ks[a] >= vis and ks[b] >= vis:
                    d.line([tuple(k[a]), tuple(k[b])], fill=(255, 165, 0, 220), width=lw)
            for base, col in ((dw.HAND_L[0], (0, 229, 255, 230)),
                              (dw.HAND_R[0], (255, 79, 216, 230))):
                for a, b in HAND_EDGES:
                    ia, ib = base + a, base + b
                    if ks[ia] >= vis and ks[ib] >= vis:
                        d.line([tuple(k[ia]), tuple(k[ib])], fill=col,
                               width=max(1, lw - 1))
                r = max(1, lw // 2)
                for i in range(base, base + 21):
                    if ks[i] >= vis:
                        d.ellipse([k[i][0] - r, k[i][1] - r, k[i][0] + r, k[i][1] + r],
                                  fill=col)
        return img
