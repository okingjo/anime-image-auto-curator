"""Person detectors for the pose stage — swappable, A/B-able.

Three interchangeable backends, all returning [(box_xyxy, score), ...]:

  anime    deepghs/anime_person_detection YOLOv8m v1.1_m (ONNX, DirectML).
           Anime-domain trained; model card reports F1 0.87 (best in table).
           PRIMARY default.
  humanart HumanArt YOLOX-L (openmmlab ONNX). Same architecture/output as the
           COCO yolox_l, so it reuses dwpose_infer's pre/post processing.
           A/B shadow candidate (human-domain, 50k images / 20 scenes).
  coco     yzd-v/DWPose yolox_l.onnx (COCO). The legacy baseline.

The pose model itself (DW-LL wholebody) is unchanged — only the boxes feeding
it differ, which is exactly the variable the A/B telemetry measures.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .ort_utils import letterbox_yolo, make_session, resolve_model, yolo_post

ANIME_REPO = "deepghs/anime_person_detection"
ANIME_DIR = "person_detect_v1.1_m"
HUMANART_LOCAL = ".models/humanart/yolox_l_humanart.onnx"


class AnimePersonDetector:
    """YOLOv8m anime person detector (single class), ONNX + DirectML."""

    kind = "anime"

    def __init__(self, conf: float = 0.35, iou: float = 0.45, path=None):
        p = Path(path) if path else resolve_model(
            ANIME_REPO, f"{ANIME_DIR}/model.onnx")
        if p is None:
            raise FileNotFoundError(
                "anime person detector missing — run scripts/setup_models.py --only detector")
        self.sess = make_session(p)
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou = float(conf), float(iou)
        self.path = str(p)

    def detect(self, img: Image.Image):
        W, H = img.size
        tensor, scale, dw, dh = letterbox_yolo(img, 640)
        out = self.sess.run(None, {self.inp: tensor})[0]
        boxes, scores = yolo_post(out, scale, dw, dh, W, H,
                                  self.conf, self.iou, class_id=0)
        return list(zip(boxes, scores))


class HumanArtDetector:
    """HumanArt YOLOX-L — identical I/O contract to the COCO YOLOX-L."""

    kind = "humanart"

    def __init__(self, conf: float = 0.30, iou: float = 0.45, path=None):
        from .dwpose_infer import _det_preprocess, _det_postprocess
        self._pre, self._post = _det_preprocess, _det_postprocess
        root = Path(__file__).resolve().parents[2]
        p = Path(path) if path else root / HUMANART_LOCAL
        if not p.exists():
            raise FileNotFoundError(
                "HumanArt ONNX missing — run scripts/setup_models.py --only detector")
        self.sess = make_session(p)
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou = float(conf), float(iou)
        self.path = str(p)

    def detect(self, img: Image.Image):
        W, H = img.size
        tensor, scale = self._pre(img)
        out = self.sess.run(None, {self.inp: tensor})[0]
        boxes, scores = self._post(out, scale, W, H, self.conf, self.iou)
        return list(zip(boxes, scores))


class CocoYoloxDetector:
    """Legacy COCO YOLOX-L from the DWPose repo (baseline)."""

    kind = "coco"

    def __init__(self, conf: float = 0.30, iou: float = 0.45, path=None):
        from .dwpose_infer import _det_preprocess, _det_postprocess, ensure_models
        self._pre, self._post = _det_preprocess, _det_postprocess
        if path is None:
            root = Path(__file__).resolve().parents[2]
            path, _ = ensure_models(root / ".models")
        self.sess = make_session(path)
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou = float(conf), float(iou)
        self.path = str(path)

    def detect(self, img: Image.Image):
        W, H = img.size
        tensor, scale = self._pre(img)
        out = self.sess.run(None, {self.inp: tensor})[0]
        boxes, scores = self._post(out, scale, W, H, self.conf, self.iou)
        return list(zip(boxes, scores))


def build_detector(kind: str, conf: float = 0.35, iou: float = 0.45, path=None):
    kind = (kind or "anime").lower()
    if kind == "anime":
        return AnimePersonDetector(conf, iou, path)
    if kind == "humanart":
        return HumanArtDetector(conf, iou, path)
    if kind == "coco":
        return CocoYoloxDetector(conf, iou, path)
    raise ValueError(f"unknown detector kind: {kind}")


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def detector_agreement(primary_boxes, shadow_boxes) -> dict:
    """Mean best-IoU of primary boxes against the shadow set (+counts).
    Logged per image for the detector A/B decision."""
    if not primary_boxes:
        return {"n_primary": 0, "n_shadow": len(shadow_boxes), "mean_best_iou": None}
    ious = []
    for b in primary_boxes:
        best = max((box_iou(b, s) for s in shadow_boxes), default=0.0)
        ious.append(best)
    return {"n_primary": len(primary_boxes), "n_shadow": len(shadow_boxes),
            "mean_best_iou": round(sum(ious) / len(ious), 3)}