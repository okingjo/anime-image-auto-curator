"""Pluggable scorer framework.

Each scorer inspects one image and returns a dict:
    {
      "score": float in [0, 1],      # overall for this scorer
      "subs":  {name: float[0,1]},   # optional sub-scores (for UI)
      "flags": [str, ...],           # human-readable warnings
      "notes": str,                  # optional free text
    }

Scorers that need heavy/optional deps must set `available = False` (with a
reason) when those deps or models are missing, so the pipeline degrades
gracefully and the UI can guide the user to enable them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from ..metadata import ImageMeta


@dataclass
class ScoreResult:
    score: float = 1.0
    subs: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    notes: str = ""

    def to_dict(self):
        return {"score": self.score, "subs": self.subs, "flags": self.flags, "notes": self.notes}


class Scorer:
    name: str = "base"
    label: str = "Base"
    available: bool = True
    reason: str = ""          # why unavailable (shown in UI)
    needs_image: bool = True  # False = metadata-only, cheap

    def score(self, image: Optional[Image.Image], meta: ImageMeta) -> ScoreResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Registry: import-guarded so a missing optional dep never breaks startup.
# --------------------------------------------------------------------------- #
def get_scorers() -> dict:
    """Return {name: scorer_instance} for every scorer module.

    Each scorer module is import-safe (heavy deps are guarded inside it and
    reflected via instance.available). We still wrap in try/except as a
    backstop so one bad scorer can never break startup.
    """
    scorers: dict = {}
    for mod, cls_name, fallback_label in [
        (".quality", "QualityScorer", "质检"),
        (".aesthetic", "AestheticScorer", "美观度"),
        (".vlm_judge", "VLMJudgeScorer", "VLM判定"),
    ]:
        try:
            import importlib

            m = importlib.import_module(__name__.rsplit(".", 1)[0] + mod)
            _add(scorers, getattr(m, cls_name)())
        except Exception as e:  # pragma: no cover
            name = mod.lstrip(".")
            _add(scorers, _Unavailable(name, fallback_label, f"import failed: {e}"))
    return scorers


class _Unavailable(Scorer):
    def __init__(self, name, label, reason):
        self.name = name
        self.label = label
        self.available = False
        self.reason = reason
        self.needs_image = False

    def score(self, image, meta):  # never called when unavailable
        return ScoreResult(score=1.0, notes=self.reason)


def _add(d: dict, s: Scorer):
    d[s.name] = s
