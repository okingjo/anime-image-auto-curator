"""Scan -> parse -> match character/outfit -> score -> group -> rank."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from .characters import load_characters, match_character, match_outfit
from .metadata import ImageMeta, group_key, is_image, read_meta
from .scorers.base import get_scorers

_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ImageRecord:
    filename: str
    path: str
    width: int = 0
    height: int = 0
    positive: str = ""
    negative: str = ""
    params: dict = field(default_factory=dict)
    loras: list = field(default_factory=list)
    character_id: Optional[str] = None
    outfit: Optional[str] = None
    outfit_ratio: float = 0.0
    outfit_missing: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)   # name -> result dict
    composite: float = 0.0

    def to_dict(self):
        return {
            "filename": self.filename,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "positive": self.positive,
            "negative": self.negative,
            "seed": self.params.get("Seed") or self.params.get("seed"),
            "model": self.params.get("Model") or self.params.get("model"),
            "loras": self.loras,
            "character_id": self.character_id,
            "outfit": self.outfit,
            "outfit_ratio": round(self.outfit_ratio, 3),
            "outfit_missing": self.outfit_missing,
            "scores": self.scores,
            "composite": round(self.composite, 3),
        }


class Curator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.characters = load_characters(_ROOT / self.config["characters_file"])
        self.scorers = get_scorers()
        self.weights = self.config.get("weights", {})
        self.recursive = bool(self.config.get("recursive", True))
        self.thumbs_size = int(self.config.get("thumbs_size", 512))
        # calibrated 2026-10 (see docs/CALIBRATION_2026-10.md):
        # flat penalty for quality-flagged members (blur/exposure) + group-level
        # "suspect" hint when the whole group is soft (user rejects those often).
        self.quality_flag_penalty = float(self.config.get("quality_flag_penalty", 0.0))
        sus = self.config.get("suspect") or {}
        self.suspect_sharp = float(sus.get("mean_sharpness_below", 0.0)) or None
        # select aesthetic backend from config (default shadow-v2)
        aes = self.scorers.get("aesthetic")
        want = self.config.get("aesthetic_model")
        if aes is not None and want:
            aes.set_backend(want, prewarm=False)
        # configure structure scorer (DWPose + YOLO plate, shadow mode by default)
        st = self.scorers.get("structure")
        if st is not None and hasattr(st, "configure"):
            try:
                st.configure(self.config.get("structure") or {})
            except Exception:
                pass
        self.groups: dict = {}      # key -> {"key","positive","negative","character_id","members":[...]}
        self.selections: dict = {}  # key -> filename
        self.root: Optional[str] = None
        self.errors: list = []

    # ------------------------------------------------------------------ #
    def _load_config(self, path):
        candidates = [
            Path(path) if path else None,
            _ROOT / "config" / "curator.yaml",
            _ROOT / "config" / "curator.example.yaml",
        ]
        for c in candidates:
            if c and Path(c).exists():
                try:
                    data = yaml.safe_load(Path(c).read_text(encoding="utf-8")) or {}
                    data.setdefault("characters_file", "config/characters.example.yaml")
                    data.setdefault("weights", {"quality": 0.8, "aesthetic": 0.2,
                                                "structure": 0.05})
                    return data
                except Exception:
                    continue
        return {"characters_file": "config/characters.example.yaml",
                "weights": {"quality": 0.8, "aesthetic": 0.2, "structure": 0.05},
                "quality_flag_penalty": 0.25,
                "suspect": {"mean_sharpness_below": 0.45},
                "recursive": True, "thumbs_size": 512}

    # ------------------------------------------------------------------ #
    def _active_scorers(self):
        return [s for s in self.scorers.values() if s.available]

    def aesthetic_backends(self):
        aes = self.scorers.get("aesthetic")
        return aes.list_backends() if aes is not None else []

    def set_aesthetic_backend(self, backend_id: str) -> dict:
        """Switch backend + wipe existing scores so a rescan re-scores. Returns status."""
        aes = self.scorers.get("aesthetic")
        if aes is None:
            return {"ok": False, "reason": "aesthetic scorer 不存在"}
        ok = aes.set_backend(backend_id, prewarm=True)
        if not ok:
            b = aes.backends.get(backend_id)
            return {"ok": False, "reason": (b.reason if b else "未知后端")}
        # scores depend on the model -> invalidate so the UI prompts a rescan
        for g in self.groups.values():
            for m in g["members"]:
                m.scores.pop("aesthetic", None)
                m.composite = 0.0
        return {"ok": True, "active": backend_id, "needs_rescan": bool(self.groups)}

    def prewarm(self):
        """Eagerly load heavy scorers in a background thread so the first
        scan isn't blocked by a model download. Opt out with AIC_PREWARM=0."""
        import os
        if os.environ.get("AIC_PREWARM", "1") == "0":
            return
        import threading

        def _bg():
            for s in self.scorers.values():
                pre = getattr(s, "prewarm", None)
                if s.available and callable(pre):
                    try:
                        pre()
                    except Exception:
                        pass

        threading.Thread(target=_bg, daemon=True).start()

    def scan(self, folder: str, recursive: Optional[bool] = None):
        folder_p = Path(folder).expanduser()
        if not folder_p.is_dir():
            raise ValueError(f"不是有效目录: {folder}")
        self.root = str(folder_p.resolve())
        self.groups = {}
        self.selections = {}
        self.errors = []

        rec = bool(self.recursive if recursive is None else recursive)
        files = folder_p.rglob("*") if rec else folder_p.glob("*")
        files = [f for f in files if f.is_file() and is_image(f)]
        files.sort()

        active = self._active_scorers()
        for f in files:
            try:
                rec_meta = read_meta(f)
                record = self._build_record(f, rec_meta, active)
            except Exception as e:
                self.errors.append(f"{f.name}: {e}")
                continue
            key = group_key(rec_meta)
            g = self.groups.setdefault(key, {
                "key": key,
                "positive": rec_meta.positive,
                "negative": rec_meta.negative,
                "character_id": record.character_id,
                "members": [],
            })
            g["members"].append(record)

        # rank within groups + recommend best + group-level suspect hint
        for g in self.groups.values():
            g["members"].sort(key=lambda r: r.composite, reverse=True)
            g["recommended"] = g["members"][0].filename if g["members"] else None
            g["suspect"] = self._group_suspect(g)
        return self.summary()

    def _group_suspect(self, g) -> Optional[str]:
        """Group-level 'probably reject' hint (calibrated 2026-10).

        Feedback analysis: ~16% of groups are rejected entirely ('全部不选');
        group mean sharpness < 0.45 catches ~65% of those rejections at a
        ~23% false-alarm rate (2.2x lift over baseline). This is a HINT for
        the UI only — it never changes ranking or selection.
        """
        if not self.suspect_sharp:
            return None
        sharps = []
        for m in g["members"]:
            q = m.scores.get("quality") or {}
            v = (q.get("subs") or {}).get("sharpness")
            if isinstance(v, (int, float)):
                sharps.append(float(v))
        if not sharps:
            return None
        mean_sharp = sum(sharps) / len(sharps)
        if mean_sharp < self.suspect_sharp:
            return f"整组偏软（平均清晰度 {mean_sharp:.2f} < {self.suspect_sharp:g}），历史上此类组常被整组弃选"
        return None

    def _build_record(self, f: Path, meta: ImageMeta, active) -> ImageRecord:
        rec = ImageRecord(
            filename=f.name, path=str(f), width=meta.width, height=meta.height,
            positive=meta.positive, negative=meta.negative,
            params=meta.params, loras=meta.loras,
        )
        ch = match_character(meta, self.characters)
        if ch:
            rec.character_id = ch.id
            name, ratio, _matched, missing = match_outfit(meta, ch)
            rec.outfit, rec.outfit_ratio, rec.outfit_missing = name, ratio, missing

        # open image once for image-based scorers
        img = None
        need_img = any(s.needs_image for s in active)
        if need_img:
            try:
                img = Image.open(f).convert("RGB")
            except Exception as e:
                self.errors.append(f"{f.name}: open failed: {e}")

        weighted_sum, weight_total = 0.0, 0.0
        for s in active:
            try:
                res = s.score(img, meta)
            except Exception as e:
                res = None
                self.errors.append(f"{f.name}: scorer {s.name}: {e}")
            if res is None:
                continue
            rec.scores[s.name] = res.to_dict()
            # shadow mode: computed + displayed + recorded, but NOT in composite
            if getattr(s, "shadow", False):
                continue
            w = float(self.weights.get(s.name, 0.0))
            weighted_sum += w * float(res.score)
            weight_total += w
        if img is not None:
            img.close()
        rec.composite = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        # calibrated penalty (2026-10): members carrying quality flags (偏糊/过曝等)
        # are rejected by the user far more often than their raw score suggests.
        if self.quality_flag_penalty > 0:
            q = rec.scores.get("quality") or {}
            if q.get("flags"):
                rec.composite = max(0.0, rec.composite - self.quality_flag_penalty)
        return rec

    # ------------------------------------------------------------------ #
    def summary(self):
        return {
            "root": self.root,
            "n_groups": len(self.groups),
            "n_images": sum(len(g["members"]) for g in self.groups.values()),
            "errors": self.errors[:50],
        }

    def state(self):
        scorers_info = [
            {"name": s.name, "label": s.label, "available": s.available,
             "shadow": bool(getattr(s, "shadow", False)),
             "reason": getattr(s, "reason", "")}
            for s in self.scorers.values()
        ]
        groups = []
        for g in self.groups.values():
            gd = {
                "key": g["key"],
                "positive": g["positive"],
                "negative": g["negative"],
                "character_id": g["character_id"],
                "recommended": g.get("recommended"),
                "suspect": g.get("suspect"),
                "selected": self.selections.get(g["key"], g.get("recommended")),
                "members": [m.to_dict() for m in g["members"]],
            }
            groups.append(gd)
        return {
            "root": self.root,
            "scorers": scorers_info,
            "weights": self.weights,
            "quality_flag_penalty": self.quality_flag_penalty,
            "characters": list(self.characters.keys()),
            "groups": groups,
        }

    # ------------------------------------------------------------------ #
    def set_selection(self, key: str, filename: Optional[str]):
        if key in self.groups:
            self.selections[key] = filename
            return True
        return False

    def selected_paths(self):
        out = []
        for key, g in self.groups.items():
            sel = self.selections.get(key, g.get("recommended"))
            if not sel:
                continue
            for m in g["members"]:
                if m.filename == sel:
                    out.append({"group": key, "filename": sel, "path": m.path})
                    break
        return out

    def export_copy(self, dest: str):
        dest_p = Path(dest).expanduser()
        dest_p.mkdir(parents=True, exist_ok=True)
        copied = []
        for item in self.selected_paths():
            src = Path(item["path"])
            dst = dest_p / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
        return copied

    # ------------------------------------------------------------------ #
    # Feedback recording (see docs/FEEDBACK_FORMAT.md)
    # ------------------------------------------------------------------ #
    def record_feedback(self, mode: str, exported: Optional[list] = None) -> Optional[str]:
        """Append one JSONL line capturing this session's selections + features.

        Called only when the user ticks “记录本次导出”. Returns the file path.
        """
        import datetime as _dt
        import uuid as _uuid

        st = self.scorers.get("structure")
        record = {
            "schema": "aic.feedback.v1.1",
            "session_id": _uuid.uuid4().hex[:12],
            "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": mode,
            "root": self.root,
            "weights": self.weights,
            "quality_flag_penalty": self.quality_flag_penalty,
            "aesthetic_backend": None,
            "structure_cfg": None,
            "scorers": [{"name": s.name, "available": s.available,
                         "shadow": bool(getattr(s, "shadow", False))}
                        for s in self.scorers.values()],
            "n_groups": len(self.groups),
            "exported": exported,
            "groups": [],
        }
        aes = self.scorers.get("aesthetic")
        if aes is not None and getattr(aes, "active", None) is not None:
            record["aesthetic_backend"] = aes.active.id
        if st is not None and hasattr(st, "cfg"):
            record["structure_cfg"] = st.cfg
        for g in self.groups.values():
            record["groups"].append({
                "key": g["key"],
                "character_id": g["character_id"],
                "positive": g["positive"][:2000],
                "recommended": g.get("recommended"),
                "suspect": g.get("suspect"),
                "selected": self.selections.get(g["key"], g.get("recommended")),
                "members": [m.to_dict() for m in g["members"]],
            })
        out_dir = _ROOT / "data" / "feedback"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"feedback-{_dt.date.today().isoformat()}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(path)
