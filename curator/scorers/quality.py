"""Lightweight, model-free image QA.

Anime images are rarely blurry/overexposed, so this scorer is intentionally
low-weight: it mostly catches *degenerate* outputs (blank/near-solid frames,
extreme saturation/clipping, tiny corrupt images) that waste a review slot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from ..metadata import ImageMeta
from .base import Scorer, ScoreResult


class QualityScorer(Scorer):
    name = "quality"
    label = "质检"
    available = True
    needs_image = True

    def score(self, image: Optional[Image.Image], meta: ImageMeta) -> ScoreResult:
        if image is None:
            return ScoreResult(score=1.0, notes="no image")
        flags = []
        subs = {}

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        gray = arr.mean(axis=2)
        mean = float(gray.mean())
        std = float(gray.std())

        # near-solid / blank
        if std < 4.0:
            flags.append("画面几乎无内容/纯色")
        # clipping (over/under-exposed) — fraction of saturated pixels
        sat = float(((gray <= 2) | (gray >= 253)).mean())
        if sat > 0.35:
            flags.append("大面积过曝/死黑")

        # sharpness via Laplacian variance (cheap, no OpenCV)
        sharp = _laplacian_var(gray)
        subs["sharpness"] = _clamp01(sharp / 400.0)
        if sharp < 8.0 and std >= 4.0:
            flags.append("偏糊")

        # color diversity (guard against monochrome garbage)
        color_std = float(arr.std(axis=(0, 1)).mean())
        subs["contrast"] = _clamp01(std / 64.0)

        # Map to a 0..1 score: start at 1, penalise flags heavily.
        score = 1.0
        score -= 0.6 * min(1.0, sat / 0.5)
        if std < 4.0:
            score -= 0.6
        score -= 0.3 * (1.0 - subs["sharpness"])
        score = _clamp01(score)
        subs["overall"] = round(score, 3)

        return ScoreResult(score=score, subs=subs, flags=flags)


def _laplacian_var(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # naive convolution via slicing (avoid scipy/opencv dependency)
    p = gray.astype(np.float32)
    lap = (
        p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    )
    return float(lap.var()) if lap.size else 0.0


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
