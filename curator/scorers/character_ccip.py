"""Image-side character fidelity via CCIP (DeepGHS character-identity metric).

Three layers, all instrumented for later calibration:
  B. reference comparison — per-character reference images + outfit/hair
     variants (``characters.yaml``). Best-matching variant wins; hair-changed
     variants get a relaxed threshold. A dynamic anchor (first confident
     image of a group) covers costume changes with no references.
  A. group outlier — after the scan, members far from their group's feature
     centroid are flagged (catches wrong-character images even when the whole
     group drifted).
  C. semantic checklist — outfit tags the prompt claims but the image must
     confirm are handed to the VLM judge (ctx["character_ccip"]["checklist"]).

Runs on DirectML through the shared ONNX session; features are cached on
disk keyed by path+mtime so rescans are cheap. Shadow mode by default.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..characters import load_characters, match_character, match_outfit
from ..metadata import ImageMeta
from .base import Scorer, ScoreResult

_ROOT = Path(__file__).resolve().parents[2]
_FEAT_CACHE = _ROOT / ".models" / "ccip_cache"

_DEFAULTS = {
    "enabled": True,
    "shadow": True,
    "characters_file": "config/characters.yaml",
    "model": "ccip-caformer_b36-24",
    "threshold": 0.213,       # official CCIP same-character cutoff
    "hair_relax": 1.4,        # threshold multiplier for hair-changed variants
    "outfit_relax": 1.15,     # multiplier when prompt outfit overlap is low
    "group_outlier": 0.16,    # layer-A distance above group median
    "dynamic_anchor": True,
    "cache": True,
    "max_checklist": 4,
}


def _path_key(p: Path) -> str:
    try:
        st = p.stat()
        raw = f"{p}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        raw = str(p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class CharacterCCIPScorer(Scorer):
    name = "character_ccip"
    label = "角色还原"
    needs_image = True
    wants_ctx = True

    def __init__(self):
        self.cfg = dict(_DEFAULTS)
        self.available = False
        self.reason = "未配置"
        self.shadow = True
        self._chars: dict = {}
        self._extractor = None
        self._lock = threading.Lock()
        self._ref_feats: dict = {}     # (cid, variant) -> [feat, ...]
        self._anchors: dict = {}       # (cid, outfit) -> feat
        self._run_feats: dict = {}     # filename -> feat (for group pass)

    def configure(self, cfg: dict):
        self.cfg = {**_DEFAULTS, **(cfg or {})}
        self.shadow = bool(self.cfg.get("shadow", True))
        self._ref_feats = {}
        self._anchors = {}
        self._run_feats = {}
        cf = Path(self.cfg.get("characters_file", "config/characters.yaml"))
        if not cf.is_absolute():
            cf = _ROOT / cf
        self._chars = load_characters(cf)
        if not self.cfg.get("enabled", True):
            self.available = False
            self.reason = "未启用（character_ccip.enabled=false）"
            return
        if not self._chars:
            self.available = False
            self.reason = "characters.yaml 无角色定义"
            return
        if not any(c.reference_images or c.variants for c in self._chars.values()):
            # no references yet: dynamic anchor + group outlier still work,
            # but only for groups whose character matched via prompt.
            pass
        try:
            from .ccip_infer import CCIPExtractor
            self._extractor = CCIPExtractor(self.cfg.get("model"))
            self.available = True
            self.reason = ""
        except Exception as e:
            self.available = False
            self.reason = f"CCIP 模型不可用：{str(e)[:80]}（运行 scripts/setup_models.py --only ccip）"

    def reset_run(self):
        self._anchors = {}
        self._run_feats = {}

    # -- features -------------------------------------------------------------
    def _feat(self, img: Image.Image, path_key: str):
        if self.cfg.get("cache", True) and path_key:
            p = _FEAT_CACHE / f"{path_key}.npy"
            if p.exists():
                try:
                    return np.load(p)
                except Exception:
                    pass
        f = self._extractor.extract(img)
        if self.cfg.get("cache", True) and path_key:
            try:
                _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
                np.save(_FEAT_CACHE / f"{path_key}.npy", f)
            except OSError:
                pass
        return f

    def _ref_set(self, ch, variant: dict | None) -> list:
        key = (ch.id, (variant or {}).get("name", "_default"))
        with self._lock:
            if key in self._ref_feats:
                return self._ref_feats[key]
            paths = (variant or {}).get("reference_images") or ch.reference_images
            feats = []
            for rp in paths:
                p = Path(rp)
                if not p.is_absolute():
                    p = _ROOT / p
                if not p.exists():
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                    feats.append(self._feat(im, _path_key(p)))
                    im.close()
                except Exception:
                    continue
            self._ref_feats[key] = feats
            return feats

    # -- main -----------------------------------------------------------------
    def score(self, image: Optional[Image.Image], meta: ImageMeta, ctx: dict = None):
        if image is None or not self.available:
            return None
        ctx = ctx if ctx is not None else {}
        ch = match_character(meta, self._chars)
        if ch is None:
            return None

        # crop to the largest detected person when available
        crop = image
        boxes = ctx.get("person_boxes") or []
        if boxes:
            b = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            w, h = image.size
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(w, int(b[2])), min(h, int(b[3]))
            if x2 - x1 > 32 and y2 - y1 > 32:
                crop = image.crop((x1, y1, x2, y2))

        pk = _path_key(Path(str(getattr(meta, "path", "") or "")))
        try:
            feat = self._feat(crop, pk)
        except Exception as e:
            return ScoreResult(score=1.0, flags=[f"CCIP提取失败:{type(e).__name__}"],
                               notes="CCIP 特征提取失败")
        self._run_feats[getattr(meta, "filename", "") or pk] = feat

        outfit_name, outfit_ratio, _matched, missing = match_outfit(meta, ch)
        thr = float(self.cfg.get("threshold", 0.213))

        # layer B: references + variants
        best_diff, best_variant, hair_changed = None, "_default", False
        sets = [None] + list(ch.variants or [])
        for v in sets:
            refs = self._ref_set(ch, v)
            if not refs:
                continue
            d = min(float(self._extractor.difference(feat, r)) for r in refs)
            if best_diff is None or d < best_diff:
                best_diff = d
                best_variant = (v or {}).get("name", "_default")
                hair_changed = bool((v or {}).get("hair_changed", False))
        eff_thr = thr
        if hair_changed:
            eff_thr *= float(self.cfg.get("hair_relax", 1.4))
        if outfit_ratio < 0.5:
            eff_thr *= float(self.cfg.get("outfit_relax", 1.15))

        # dynamic anchor: first confident image of this (character, outfit)
        anchor_used = False
        akey = (ch.id, outfit_name or "_")
        if self.cfg.get("dynamic_anchor", True):
            anchor = self._anchors.get(akey)
            if anchor is not None:
                da = float(self._extractor.difference(feat, anchor))
                if best_diff is None or da < best_diff:
                    best_diff, best_variant, anchor_used = da, "_anchor", True
            elif best_diff is not None and best_diff <= eff_thr:
                self._anchors[akey] = feat

        flags = []
        if best_diff is None:
            # no references and no anchor yet -> nothing measurable
            subs = {"feats": {"ccip": {"character_id": ch.id, "measurable": False}}}
            ctx["character_ccip"] = {"character_id": ch.id, "checklist":
                                     self._checklist(ch, missing)}
            return ScoreResult(score=1.0, notes="无参考图，等待动态锚定", subs=subs)

        if best_diff <= eff_thr:
            score = 1.0 - 0.3 * (best_diff / max(eff_thr, 1e-6))
        else:
            score = max(0.0, 0.7 - 0.7 * (best_diff - eff_thr) / max(eff_thr, 1e-6))
            flags.append(f"角色还原度低(diff={best_diff:.3f}>{eff_thr:.3f})")

        checklist = self._checklist(ch, missing)
        ctx["character_ccip"] = {"character_id": ch.id, "checklist": checklist,
                                 "diff": round(best_diff, 4)}
        subs = {"overall": round(score, 3),
                "feats": {"ccip": {"character_id": ch.id, "measurable": True,
                                   "diff": round(best_diff, 4),
                                   "threshold": round(eff_thr, 4),
                                   "variant": best_variant,
                                   "hair_changed": hair_changed,
                                   "anchor_used": anchor_used,
                                   "outfit": outfit_name,
                                   "outfit_ratio": round(outfit_ratio, 3)}}}
        return ScoreResult(score=round(score, 3), flags=flags,
                           notes="shadow" if self.shadow else "CCIP角色还原",
                           subs=subs)

    def _checklist(self, ch, missing: list) -> list:
        """Layer C: outfit tags to verify visually (prompt claims them)."""
        out = []
        for i, tag in enumerate((missing or [])[:int(self.cfg.get("max_checklist", 4))]):
            out.append({"key": f"outfit_{i}", "question": f"画面中是否可见「{tag}」?"})
        return out

    # -- layer A: group outlier pass (called by the pipeline after grouping) --
    def group_pass(self, groups: dict):
        if not self.available:
            return
        gap = float(self.cfg.get("group_outlier", 0.16))
        for g in groups.values():
            members = g.get("members") or []
            feats = []
            for m in members:
                f = self._run_feats.get(m.filename)
                sc = (m.scores.get(self.name) or {}).get("subs", {}).get("feats", {})
                if f is not None and sc.get("ccip", {}).get("measurable"):
                    feats.append((m, f))
            if len(feats) < 3:
                continue
            mat = np.stack([f for _, f in feats])
            centroid = mat.mean(axis=0)
            dists = [float(self._extractor.difference(f, centroid)) for _, f in feats]
            med = float(np.median(dists))
            for (m, _f), d in zip(feats, dists):
                entry = m.scores.get(self.name)
                if entry is None:
                    continue
                outlier = d > med + gap
                entry.setdefault("subs", {}).setdefault("feats", {}).setdefault(
                    "ccip", {})["group_dist"] = round(d, 4)
                entry["subs"]["feats"]["ccip"]["group_outlier"] = outlier
                if outlier:
                    entry.setdefault("flags", []).append(
                        f"组内角色特征离群({d:.3f})")
