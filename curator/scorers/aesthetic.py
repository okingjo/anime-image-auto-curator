"""Aesthetic scorer — thin facade over pluggable backends.

The active backend is selectable at runtime (config `aesthetic_model` or the
WebUI dropdown). Defaults to shadow-v2 (the strongest anime-specific model).
Each backend downloads its model lazily on first use.
"""
from __future__ import annotations

import threading
from typing import Optional

from PIL import Image

from ..metadata import ImageMeta
from .aesthetic_backends import BACKEND_IDS, build_backends
from .base import Scorer, ScoreResult

DEFAULT_BACKEND = "shadow-v2"


class AestheticScorer(Scorer):
    name = "aesthetic"
    label = "美观度"
    needs_image = True

    def __init__(self):
        self.backends = build_backends()
        self._lock = threading.Lock()
        self._active = None
        # default to shadow-v2, else first available in display order
        self.set_backend(
            DEFAULT_BACKEND if self.backends[DEFAULT_BACKEND].available
            else next((bid for bid in BACKEND_IDS if self.backends[bid].available), None),
            prewarm=False,
        )

    # ---- backend management -------------------------------------------------
    @property
    def active(self):
        return self._active

    def set_backend(self, backend_id: Optional[str], prewarm: bool = True) -> bool:
        if backend_id is None or backend_id not in self.backends:
            return False
        b = self.backends[backend_id]
        if not b.available:
            return False
        with self._lock:
            self._active = b
        if prewarm:
            b.prewarm()
        return True

    def list_backends(self):
        out = []
        for bid in BACKEND_IDS:
            b = self.backends[bid]
            out.append({
                "id": b.id,
                "label": b.label,
                "domain": b.domain,
                "desc": b.desc,
                "scale": b.scale,
                "approx_size": b.approx_size,
                "available": b.available,
                "reason": b.reason,
                "active": self._active is b,
            })
        return out

    # ---- Scorer interface ---------------------------------------------------
    @property
    def available(self):  # type: ignore[override]
        return self._active is not None and self._active.available

    @available.setter
    def available(self, _v):  # facade; real state lives in backends
        pass

    @property
    def reason(self):  # type: ignore[override]
        if self._active is None:
            return "没有可用的美观度后端（检查 --extra aesthetic 是否安装）"
        return self._active.reason

    def prewarm(self):
        if self._active is not None:
            self._active.prewarm()

    def score(self, image: Optional[Image.Image], meta: ImageMeta) -> ScoreResult:
        b = self._active
        if b is None:
            return ScoreResult(score=1.0, notes="无可用美观度后端")
        if image is None:
            return ScoreResult(score=1.0, notes="no image")
        try:
            raw, notes = b.score(image)
            norm = b.normalize(raw)
            return ScoreResult(
                score=norm,
                subs={"raw": round(raw, 4), "model": b.id},
                notes=notes,
            )
        except Exception as e:
            b.available = False
            b.reason = f"打分出错：{e}"
            return ScoreResult(score=1.0, notes=f"[{b.id}] {e}")
