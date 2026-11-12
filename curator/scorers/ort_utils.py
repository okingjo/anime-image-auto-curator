"""Shared ONNX Runtime helpers: GPU provider selection + YOLO post-processing.

Provider order: CUDA (if onnxruntime-gpu present) -> DirectML (Windows GPU,
works on the stock onnxruntime-directml wheel) -> CPU. Everything stays inside
the project venv; no system-wide CUDA/cuDNN install is required for DirectML.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_PROVIDERS = None


def pick_providers() -> list:
    global _PROVIDERS
    if _PROVIDERS is None:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        _PROVIDERS = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                  "CPUExecutionProvider") if p in avail]
        if not _PROVIDERS:
            _PROVIDERS = ["CPUExecutionProvider"]
    return list(_PROVIDERS)


def make_session(path, providers=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=providers or pick_providers())


def resolve_model(repo: str, filename: str, local_candidates=()) -> Path | None:
    """Locate a model file: HF cache first (shared with imgutils/dghs), then
    the project's .models/ fallback locations written by scripts/setup_models.py."""
    try:
        from huggingface_hub import try_to_load_from_cache
        c = try_to_load_from_cache(repo, filename)
        if isinstance(c, str) and Path(c).exists() and Path(c).stat().st_size > 0:
            return Path(c)
    except Exception:
        pass
    root = Path(__file__).resolve().parents[2]
    cands = list(local_candidates) + [
        root / ".models" / "hub" / repo.replace("/", "--") / filename,
        root / ".models" / filename,
    ]
    for c in cands:
        c = Path(c)
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


# --------------------------------------------------------------------------- #
# YOLOv8-style letterbox + postprocess (ultralytics ONNX exports)
# --------------------------------------------------------------------------- #
def letterbox_yolo(img: Image.Image, size: int = 640):
    """Center-padded letterbox, RGB 0-1 CHW (ultralytics convention).
    Returns (tensor, scale, pad_w, pad_h)."""
    W, H = img.size
    r = min(size / W, size / H)
    nw, nh = max(1, int(round(W * r))), max(1, int(round(H * r)))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR),
                 (int(round(dw)), int(round(dh))))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
    return tensor, r, dw, dh


def yolo_post(out: np.ndarray, scale: float, dw: float, dh: float,
              img_w: int, img_h: int, conf: float = 0.35, iou: float = 0.45,
              class_id: int | None = None):
    """Ultralytics export (1, 4+nc, N) or (1, N, 4+nc) -> (boxes xyxy, scores).
    Boxes are mapped back to original image coordinates."""
    from .dwpose_infer import nms
    pred = out[0] if out.ndim == 3 else out
    if pred.ndim != 2:
        return [], []
    if pred.shape[0] < pred.shape[1]:      # (4+nc, N) -> (N, 4+nc)
        pred = pred.T
    if pred.shape[1] < 5:
        return [], []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    cls = pred[:, 4:]
    if cls.shape[1] == 1:
        s = cls[:, 0]
    else:
        cid = class_id if class_id is not None else int(cls.argmax(axis=1)[0])
        s = cls[:, cid] if class_id is not None else cls.max(axis=1)
    mask = s > conf
    if not mask.any():
        return [], []
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)[mask]
    xyxy[:, 0::2] = (xyxy[:, 0::2] - dw) / scale
    xyxy[:, 1::2] = (xyxy[:, 1::2] - dh) / scale
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, img_w)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, img_h)
    ss = s[mask]
    keep = nms(xyxy, ss, iou)
    return list(xyxy[keep]), list(ss[keep])


def crop_box(img: Image.Image, box, pad: float = 1.0, min_side: int = 0):
    """Crop `box` (xyxy) with multiplicative padding, clipped to the image.
    Optionally upscale so the longest side >= min_side (for VLM detail crops)."""
    W, H = img.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max((x2 - x1) * pad, 8.0), max((y2 - y1) * pad, 8.0)
    x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    x2, y2 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img.crop((x1, y1, x2, y2))
    if min_side and max(crop.size) < min_side:
        r = min_side / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * r)), max(1, int(crop.height * r))),
                           Image.LANCZOS)
    return crop