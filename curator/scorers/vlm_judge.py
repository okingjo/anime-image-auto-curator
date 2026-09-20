"""Local VLM judge for hands / anatomy / clothing structure (Phase 3).

Strategy (the only realistic way to *reason* about the hard criteria):
  give a local vision-language model (e.g. Qwen2.5-VL-7B, fits in 24GB) the
  image + prompt + expected outfit tags, and ask structured yes/no questions
  with confidence: extra/missing fingers, palm/back orientation, broken
  limbs, implausible clothing structure, character-fidelity problems.
  Aggregate into a score + per-issue flags.

Requires: uv sync --extra vlm   (transformers + accelerate + bitsandbytes)
The model (~15GB, or quantized ~5-8GB) is downloaded from HF on first use.
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from ..metadata import ImageMeta
from .base import Scorer, ScoreResult

try:
    import transformers  # noqa: F401
    import accelerate  # noqa: F401

    _DEPS_OK = True
    _DEPS_ERR = ""
except Exception as _e:
    _DEPS_OK = False
    _DEPS_ERR = str(_e)


class VLMJudgeScorer(Scorer):
    name = "vlm_judge"
    label = "VLM判定"
    needs_image = True

    def __init__(self):
        # Phase 3 not yet implemented -> report unavailable with guidance.
        self.available = False
        self.reason = (
            "Phase 3 未启用。安装依赖：uv sync --extra vlm（本地 VLM 结构/手部判定）。"
            if not _DEPS_OK
            else "Phase 3 打分逻辑开发中。"
        )

    def score(self, image: Optional[Image.Image], meta: ImageMeta) -> ScoreResult:
        return ScoreResult(score=1.0, notes=self.reason)
