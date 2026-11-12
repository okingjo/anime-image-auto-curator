"""Perception v2 unit tests: preprocessing parity, geometry rules, parsing.

Run: .venv/Scripts/python.exe -m pytest tests/test_perception_v2.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- CCIP ------
def test_ccip_preprocess_matches_imgutils():
    from curator.scorers.ccip_infer import preprocess
    imgutils = pytest.importorskip("imgutils.metrics.ccip")
    import inspect
    # imgutils' internal preprocessing: locate its transform
    img = Image.fromarray(np.random.RandomState(0).randint(
        0, 255, (217, 331, 3), dtype=np.uint8))
    ours = preprocess(img)
    # replicate imgutils reference pipeline explicitly (384 BILINEAR + CLIP norm)
    ref = img.convert("RGB").resize((384, 384), Image.BILINEAR)
    data = np.asarray(ref, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    ref = (data - mean[:, None, None]) / std[:, None, None]
    assert ours.shape == (3, 384, 384)
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_ccip_difference_cosine():
    from curator.scorers.ccip_infer import difference
    a = np.array([1.0, 0.0, 0.0])
    assert difference(a, a) == pytest.approx(0.0, abs=1e-9)
    assert difference(a, -a) == pytest.approx(2.0, abs=1e-9)
    assert difference(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0, abs=1e-9)
    assert difference(np.zeros(3), a) == 1.0


# ------------------------------------------------------- YOLO letterbox -----
def test_letterbox_and_post():
    from curator.scorers.ort_utils import letterbox_yolo, yolo_post
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    blob, r, dw, dh = letterbox_yolo(img, 640)
    assert blob.shape == (1, 3, 640, 640)
    # synthetic YOLOv8 output: one high-confidence box at canvas center
    # layout (1, 4+nc, 8400); cx,cy,w,h in canvas px
    out = np.zeros((1, 5, 8400), dtype=np.float32)
    out[0, 0, 100] = 320.0   # cx
    out[0, 1, 100] = 320.0   # cy
    out[0, 2, 100] = 100.0   # w
    out[0, 3, 100] = 80.0    # h
    out[0, 4, 100] = 0.9     # score
    boxes, scores = yolo_post(out, r, dw, dh, 640, 480, conf=0.25, iou=0.45)
    assert len(boxes) == 1
    box, score = boxes[0], scores[0]
    assert score == pytest.approx(0.9, abs=1e-3)
    x1, y1, x2, y2 = box
    # box should map back near original-image center (320, 240)
    assert abs((x1 + x2) / 2 - 320) < 12
    assert abs((y1 + y2) / 2 - 240) < 12
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480


# --------------------------------------------------------- foot geometry ----
def _foot_kp(toe_spread=0.05, heel_angle_ok=True):
    """Build a 133x3 keypoint array with a plausible left foot."""
    ks = np.zeros((133, 3), dtype=np.float32)
    # left foot: big toe 17, small toe 18, heel 19, ankle 15
    ks[15] = [0.50, 0.80, 0.9]   # ankle
    ks[19] = [0.50, 0.90, 0.9]   # heel
    ks[17] = [0.48, 0.97, 0.9]   # big toe
    ks[18] = [0.48 + toe_spread, 0.97, 0.9]  # small toe (same y: pure spread)
    return ks


def test_foot_rule_normal_vs_deformed():
    from curator.scorers.structure import StructureScorer
    s = StructureScorer()
    s.configure({"shadow": True})
    pcfg = {"foot_vis": 0.40}

    def run(ks):
        p = {"kpts": ks[:, :2].astype(np.float32),
             "kscores": ks[:, 2].astype(np.float32),
             "box": [0, 0, 100, 100]}
        feats: dict = {}
        return s._check_foot(p, pcfg, feats, "")

    normal = _foot_kp(toe_spread=0.05)
    sc_n, flags_n, feats_n = run(normal)
    # fused toes (spread ~ 0) should be flagged or score lower than normal
    fused = _foot_kp(toe_spread=0.0005)
    sc_f, flags_f, feats_f = run(fused)
    assert sc_n is not None and sc_f is not None
    assert sc_f <= sc_n
    assert any("融合" in fl for fl in flags_f)
    # invisible feet -> N/A (None), never penalized
    hidden = np.zeros((133, 3), dtype=np.float32)
    sc_h, _, _ = run(hidden)
    assert sc_h is None


# ------------------------------------------------------------- VLM parse ----
def test_extract_json_variants():
    from curator.scorers.vlm_judge import _extract_json
    obj = {"issues": [], "anatomy_ok": True}
    assert _extract_json(json.dumps(obj)) == obj
    assert _extract_json("```json\n" + json.dumps(obj) + "\n```") == obj
    assert _extract_json("思考……\n" + json.dumps(obj) + "\n以上。") == obj
    assert _extract_json("no json here") is None
    assert _extract_json("{broken") is None


def test_vlm_gating():
    from curator.scorers.vlm_judge import VLMJudgeScorer
    s = VLMJudgeScorer()
    s.configure({"enabled": True, "base_url": "http://127.0.0.1:1/v1"})
    ok, why = s._should_call({"structure": {"score": 0.95, "flags": []}})
    assert not ok
    ok, why = s._should_call({"structure": {"score": 0.6, "flags": []}})
    assert ok
    ok, why = s._should_call({"structure": {"score": 0.95,
                                            "flags": ["手指融合"]}})
    assert ok
    ok, why = s._should_call({"structure": {"score": 0.95, "flags": []},
                              "character_ccip": {"checklist": [
                                  {"key": "outfit_0", "question": "x?"}]}})
    assert ok


# ------------------------------------------------------------- characters ---
def test_characters_variants_load(tmp_path):
    from curator.characters import load_characters
    p = tmp_path / "chars.yaml"
    p.write_text("""
characters:
  - id: testc
    name_tags: [testc]
    reference_images: [refs/a.png]
    variants:
      - name: swim
        reference_images: [refs/b.png]
      - name: shorthair
        hair_changed: true
        reference_images: [refs/c.png]
    outfits:
      - name: default
        default: true
        tags: [dress]
""", encoding="utf-8")
    chars = load_characters(p)
    c = chars["testc"]
    assert c.reference_images == ["refs/a.png"]
    assert len(c.variants) == 2
    assert c.variants[1]["hair_changed"] is True
    assert c.variants[0]["hair_changed"] is False


# ------------------------------------------------------------- telemetry ----
def test_telemetry_roundtrip(tmp_path):
    from types import SimpleNamespace
    from curator import telemetry
    rec = SimpleNamespace(filename="a.png", path="a.png", width=100, height=100,
                          character_id=None, outfit=None, outfit_ratio=0.0,
                          composite=0.5,
                          scores={"structure": {"score": 0.8, "flags": ["x"],
                                                "subs": {"feats": {"detector": "anime"}}}})
    row = telemetry.image_row(rec, "g1")
    assert row["schema"] == "aic.telemetry.v1"
    assert row["scores"]["structure"]["feats"]["detector"] == "anime"
    out = tmp_path / "run.jsonl"
    telemetry.write_run(out, [row], {"root": "x"})
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["filename"] == "a.png"
