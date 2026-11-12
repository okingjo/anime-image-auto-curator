"""Per-run telemetry for perception calibration.

Every scan appends one JSONL file under ``data/telemetry/`` capturing the
evidence each scorer produced (detector A/B agreement, CCIP distances,
VLM issues/latency, visibility checklist). Combined with user feedback
(docs/FEEDBACK_FORMAT.md) this is the dataset for the next calibration
round — detector A/B final choice, threshold tuning, deformity classifier.

Schema: aic.telemetry.v1 (one row per image).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = _ROOT / "data" / "telemetry"


def _json_default(o):
    """numpy scalars/arrays leak in from ONNX scorers -> plain Python."""
    if np is not None:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def run_path(root: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    return TELEMETRY_DIR / f"run_{stamp}.jsonl"


def image_row(record, group_key: str) -> dict:
    """Extract the calibration-relevant evidence from one ImageRecord."""
    row = {
        "schema": "aic.telemetry.v1",
        "filename": record.filename,
        "group": group_key,
        "character_id": record.character_id,
        "outfit": record.outfit,
        "outfit_ratio": round(record.outfit_ratio, 3),
        "composite": round(record.composite, 4),
        "scores": {},
    }
    for name, sc in record.scores.items():
        subs = sc.get("subs") or {}
        entry = {"score": sc.get("score"), "flags": sc.get("flags") or []}
        feats = subs.get("feats")
        if isinstance(feats, dict):
            entry["feats"] = feats
        # keep numeric sub-scores (cheap, useful for calibration)
        nums = {}
        for k, v in subs.items():
            if k == "feats":
                continue
            if isinstance(v, (int, float)):
                nums[k] = v
            elif np is not None and isinstance(v, np.generic):
                nums[k] = v.item()
        if nums:
            entry["subs"] = nums
        row["scores"][name] = entry
    return row


def write_run(path: Path, rows: list, meta: dict | None = None):
    try:
        with open(path, "w", encoding="utf-8") as f:
            if meta:
                f.write(json.dumps({"schema": "aic.telemetry.v1",
                                    "type": "run_header", **meta},
                                   ensure_ascii=False,
                                   default=_json_default) + "\n")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False,
                                   default=_json_default) + "\n")
        return str(path)
    except OSError:
        return ""
