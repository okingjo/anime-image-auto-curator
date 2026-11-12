"""CCIP character-feature extraction on ONNX Runtime + DirectML.

Numerically identical preprocessing to dghs-imgutils' ccip module:
  resize 384x384 BILINEAR (no aspect preservation) -> RGB CHW /255 ->
  CLIP mean/std normalization. Verified against imgutils in tests.

We run the ONNX session ourselves (instead of imgutils' CPU-only loader) so
the feature model uses the GPU via DirectML, and so the whole stack needs
only onnxruntime at inference time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .ort_utils import make_session, resolve_model

CCIP_REPO = "deepghs/ccip_onnx"
DEFAULT_MODEL = "ccip-caformer_b36-24"
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(img: Image.Image, size: int = 384) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    data = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (data - _MEAN[:, None, None]) / _STD[:, None, None]


def difference(fa: np.ndarray, fb: np.ndarray) -> float:
    """CCIP difference = 1 - cosine similarity (lower = more same)."""
    a = np.asarray(fa, dtype=np.float64).ravel()
    b = np.asarray(fb, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


class CCIPExtractor:
    def __init__(self, model: str = DEFAULT_MODEL):
        p = resolve_model(CCIP_REPO, f"{model}/model_feat.onnx")
        if p is None:
            raise FileNotFoundError(
                f"CCIP model missing ({model}) — run scripts/setup_models.py --only ccip")
        self.sess = make_session(p)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.model = model
        self.path = str(p)
        self._threshold = None

    def extract(self, img: Image.Image) -> np.ndarray:
        data = preprocess(img)[None].astype(np.float32)
        feat = self.sess.run([self.out], {self.inp: data})[0][0]
        return np.asarray(feat, dtype=np.float32)

    def difference(self, fa: np.ndarray, fb: np.ndarray) -> float:
        return difference(fa, fb)

    def default_threshold(self) -> float:
        """Model-card threshold (metrics.json); fallback 0.213 (b36 card)."""
        if self._threshold is None:
            self._threshold = 0.213
            p = resolve_model(CCIP_REPO, f"{self.model}/metrics.json")
            if p is not None:
                try:
                    m = json.loads(Path(p).read_text(encoding="utf-8"))
                    t = m.get("threshold") or m.get("default_threshold")
                    if isinstance(t, (int, float)):
                        self._threshold = float(t)
                except Exception:
                    pass
        return self._threshold