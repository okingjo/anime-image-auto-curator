"""Compact DWPose inference: YOLOX-L person detector + DW-LL wholebody pose.

Pure numpy + PIL + onnxruntime — no torch / mmpose / opencv dependency.
Replicates the two-stage top-down pipeline used by ControlNet & ADetailer:

  1. YOLOX-L (640x640, BGR 0-255)          -> person boxes (conf filter + NMS)
  2. DW-LL wholebody (288x384, SimCC head) -> 133 COCO-WholeBody keypoints/person

COCO-WholeBody 133 keypoint layout:
    0-16    body (COCO17: nose, eyes, ears, shoulders, elbows, wrists,
                  hips, knees, ankles)
    17-22   feet (L big toe, L small toe, L heel, R big toe, R small toe, R heel)
    23-90   face (68)
    91-111  left hand  (21: wrist, thumb x4, index x4, middle x4, ring x4, pinky x4)
    112-132 right hand (21)

Score/activation convention:
    The yzd-v ONNX exports apply sigmoid to YOLOX obj/cls internally, while
    SimCC logits may be raw. Both are auto-detected at runtime (values > 1.2
    => raw logits => apply sigmoid), so thresholds behave consistently.

Model files (~340MB total) are downloaded on first use from HF (mirror-first),
reusing the multi-source/resume helpers of the aesthetic backends.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MODEL_REPO = "yzd-v/DWPose"
DET_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"

DET_SIZE = 640
POSE_W, POSE_H = 288, 384        # model input (w, h)
SIMCC_SPLIT = 2.0                # SimCC bin ratio -> x bins 576, y bins 768
N_KPTS = 133

# COCO-WholeBody index ranges
BODY = (0, 17)
FEET = (17, 23)
FACE = (23, 91)
HAND_L = (91, 112)
HAND_R = (112, 133)

# COCO17 body indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, \
    L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = range(17)

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _maybe_sigmoid(arr: np.ndarray) -> np.ndarray:
    """Apply sigmoid only if the array looks like raw logits (max > 1.2)."""
    return _sigmoid(arr) if float(arr.max()) > 1.2 else arr


# --------------------------------------------------------------------------- #
# Model download
# --------------------------------------------------------------------------- #
def ensure_models(cache_dir: Path):
    """Download det+pose ONNX (mirror-first, resumable) -> (det_path, pose_path)."""
    from .aesthetic_backends import _HF_ENDPOINTS, _ensure_endpoint, _raw_download

    _ensure_endpoint()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in (DET_FILE, POSE_FILE):
        p = None
        try:
            from huggingface_hub import hf_hub_download, try_to_load_from_cache
            c = try_to_load_from_cache(MODEL_REPO, fn)
            if isinstance(c, str) and Path(c).exists() and Path(c).stat().st_size > 0:
                p = c
            else:
                for ep in _HF_ENDPOINTS:
                    try:
                        p = hf_hub_download(MODEL_REPO, fn, endpoint=ep)
                        break
                    except Exception as e:
                        print(f"[dwpose] hub {ep} {fn}: {type(e).__name__} {str(e)[:80]}",
                              flush=True)
        except Exception:
            pass
        if not p:
            dest = cache_dir / fn
            if dest.exists() and dest.stat().st_size > 1_000_000:
                p = str(dest)
            else:
                for ep in _HF_ENDPOINTS:
                    if _raw_download(f"{ep}/{MODEL_REPO}/resolve/main/{fn}", dest):
                        p = str(dest)
                        break
        if not p:
            raise RuntimeError(
                f"DWPose 模型下载失败: {fn}。可手动下载 "
                f"https://hf-mirror.com/{MODEL_REPO}/resolve/main/{fn} 放入 .models/")
        paths.append(Path(p))
    return paths[0], paths[1]


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #
def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.45) -> np.ndarray:
    """Greedy IoU NMS. boxes: (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= thr]
    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------- #
# YOLOX detection
# --------------------------------------------------------------------------- #
def _det_preprocess(img: Image.Image):
    """Letterbox to 640x640 (pad 114), BGR 0-255 CHW. Returns (tensor, scale)."""
    W, H = img.size
    r = min(DET_SIZE / W, DET_SIZE / H)
    nw, nh = max(1, int(round(W * r))), max(1, int(round(H * r)))
    canvas = Image.new("RGB", (DET_SIZE, DET_SIZE), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (0, 0))
    arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]   # RGB -> BGR
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None]), r


_DET_GRIDS = None


def _det_grids(strides=(8, 16, 32)):
    """YOLOX proposal grids; grid[f] = (f % g, f // g) = (x, y) in cells."""
    grids, estr = [], []
    for s in strides:
        g = DET_SIZE // s
        xv, yv = np.meshgrid(np.arange(g), np.arange(g))   # xy indexing
        grids.append(np.stack((xv, yv), axis=-1).reshape(-1, 2))
        estr.append(np.full((g * g, 1), s, dtype=np.float32))
    return (np.concatenate(grids, 0).astype(np.float32),
            np.concatenate(estr, 0))


def _det_postprocess(out: np.ndarray, scale: float, img_w: int, img_h: int,
                     conf: float, iou: float):
    """Raw YOLOX output -> (boxes xyxy in original coords, scores)."""
    global _DET_GRIDS
    pred = out[0] if out.ndim == 3 else out
    if pred.ndim != 2:
        return [], []
    if pred.shape[1] != 85 and pred.shape[0] == 85:   # transposed export
        pred = pred.T
    n_prop = pred.shape[0]
    if n_prop == 8400:                                  # needs grid decoding
        if _DET_GRIDS is None:
            _DET_GRIDS = _det_grids()
        grids, estr = _DET_GRIDS
        cx, cy = pred[:, 0], pred[:, 1]
        w, h = np.exp(np.clip(pred[:, 2], -20, 20)), np.exp(np.clip(pred[:, 3], -20, 20))
        px = (cx + grids[:, 0]) * estr[:, 0]
        py = (cy + grids[:, 1]) * estr[:, 0]
        pw, ph = w * estr[:, 0], h * estr[:, 0]
        xyxy = np.stack([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], axis=1)
        obj = _maybe_sigmoid(pred[:, 4])
        cls = _maybe_sigmoid(pred[:, 5:])
        s = obj * cls[:, 0]                              # person class
    else:                                                # pre-decoded export
        xyxy = pred[:, :4].copy()
        obj = _maybe_sigmoid(pred[:, 4])
        s = obj
    mask = s > conf
    if not mask.any():
        return [], []
    b, s = xyxy[mask], s[mask]
    keep = nms(b, s, iou)
    boxes, scores = b[keep] / scale, s[keep]
    if len(boxes):
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)
    return list(boxes), list(scores)


# --------------------------------------------------------------------------- #
# Wholebody pose (top-down, SimCC)
# --------------------------------------------------------------------------- #
def _expand_box(box, pad: float = 1.25):
    """Pad box (1.25x like mmpose) and fit to pose input aspect (288:384)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * pad, (y2 - y1) * pad
    bw, bh = max(bw, 1.0), max(bh, 1.0)
    aspect = POSE_W / POSE_H
    if bw / bh > aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    return cx - bw / 2, cy - bh / 2, bw, bh


def _crop_scale(img: Image.Image, X1, Y1, BW, BH):
    canvas = Image.new("RGB", (max(1, int(round(BW))), max(1, int(round(BH)))), (0, 0, 0))
    canvas.paste(img.convert("RGB"), (int(round(-X1)), int(round(-Y1))))
    resized = canvas.resize((POSE_W, POSE_H), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)          # RGB (mmpose to_rgb convention)
    arr = (arr - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def _simcc_decode(sim: np.ndarray, win: int = 5):
    """(N,K,W) SimCC logits -> refined bin positions (N,K) + peak scores (N,K).

    Peak scores are returned on a 0..1 scale (sigmoid applied if raw logits).
    """
    N, K, W = sim.shape
    idx = sim.argmax(axis=2)
    half = win // 2
    refined = idx.astype(np.float32)
    for n in range(N):
        row = sim[n]
        for k in range(K):
            i = int(idx[n, k])
            lo, hi = max(0, i - half), min(W, i + half + 1)
            w = np.maximum(row[k, lo:hi], 0.0)
            s = float(w.sum())
            if s > 1e-8:
                pos = np.arange(lo, hi, dtype=np.float32)
                refined[n, k] = float((w * pos).sum() / s)
    peaks = np.take_along_axis(sim, idx[..., None], axis=2)[..., 0]
    return refined, _maybe_sigmoid(peaks)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class DWPoseRunner:
    """Two-stage wholebody pose. `infer(PIL image)` -> list of person dicts:
        {"box": xyxy, "det_score": f, "kpts": (133,2) px, "kscores": (133,)}
    """

    def __init__(self, det_path, pose_path, det_conf: float = 0.3,
                 nms_iou: float = 0.45, max_persons: int = 4):
        import onnxruntime as ort
        provs = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in provs else ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.det = ort.InferenceSession(str(det_path), sess_options=so, providers=providers)
        self.pose = ort.InferenceSession(str(pose_path), sess_options=so, providers=providers)
        self.det_in = self.det.get_inputs()[0].name
        self.pose_in = self.pose.get_inputs()[0].name
        self.det_conf = float(det_conf)
        self.nms_iou = float(nms_iou)
        self.max_persons = int(max_persons)
        self.providers = providers

    def infer(self, img: Image.Image) -> list:
        img = img.convert("RGB")
        W, H = img.size
        tensor, scale = _det_preprocess(img)
        out = self.det.run(None, {self.det_in: tensor})[0]
        boxes, scores = _det_postprocess(np.asarray(out), scale, W, H,
                                         self.det_conf, self.nms_iou)
        persons = []
        for box, ds in list(zip(boxes, scores))[: self.max_persons]:
            X1, Y1, BW, BH = _expand_box(box)
            crop = _crop_scale(img, X1, Y1, BW, BH)
            outs = [np.asarray(o) for o in self.pose.run(None, {self.pose_in: crop})]
            # identify x/y SimCC by bin count: x -> W*2=576, y -> H*2=768
            if outs[0].shape[-1] <= outs[1].shape[-1]:
                sx, sy = outs[0], outs[1]
            else:
                sx, sy = outs[1], outs[0]
            rx, px = _simcc_decode(sx)
            ry, py = _simcc_decode(sy)
            ks = np.clip(0.5 * (px + py), 0.0, 1.0)[0]        # (133,)
            kx = rx[0] / SIMCC_SPLIT                           # px in 0..288
            ky = ry[0] / SIMCC_SPLIT                           # px in 0..384
            kpts = np.stack([X1 + kx * (BW / POSE_W),
                             Y1 + ky * (BH / POSE_H)], axis=1)  # (133,2) orig coords
            body_ks = ks[BODY[0]:BODY[1]]
            persons.append({
                "box": [float(v) for v in box],
                "det_score": float(ds),
                "kpts": kpts.astype(np.float32),
                "kscores": ks.astype(np.float32),
                "body_score": float(body_ks.mean()),
            })
        persons.sort(key=lambda p: p["body_score"], reverse=True)
        return persons
