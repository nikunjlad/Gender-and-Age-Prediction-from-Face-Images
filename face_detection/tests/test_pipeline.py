"""Numpy-side tests (no torch needed): priors, encoding, matching, augmentation, parsing, AP."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest


from face_detection.models.anchors import generate_priors
from face_detection.processing.annotations import parse_label_file
from face_detection.config import load_config
from face_detection.metrics.ap import average_precision
from face_detection.models.targets import (
    decode,
    decode_landm,
    encode,
    encode_landm,
    iou_matrix,
    match,
    nms,
    point_form,
)
from face_detection.processing.transforms import TrainTransform, ValTransform, hflip

MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]
VAR = (0.1, 0.2)
ROOT = Path(__file__).resolve().parents[1]


def reference_priors(image_size):
    """Literal port of biubug6 PriorBox.forward."""
    import math

    h, w = image_size
    anchors = []
    for k, step in enumerate(STEPS):
        fh, fw = math.ceil(h / step), math.ceil(w / step)
        for i, j in product(range(fh), range(fw)):
            for ms in MIN_SIZES[k]:
                anchors += [(j + 0.5) * step / w, (i + 0.5) * step / h, ms / w, ms / h]
    return np.asarray(anchors, np.float32).reshape(-1, 4)


@pytest.mark.parametrize("size", [(640, 640), (840, 840), (480, 736), (33, 65)])
def test_priors_match_reference(size):
    """generate_priors gives exactly the same anchors as biubug6's PriorBox, for several image sizes."""
    np.testing.assert_allclose(generate_priors(size, MIN_SIZES, STEPS), reference_priors(size), atol=1e-6)


def test_prior_count():
    """A 640x640 image has 16,800 anchors: 2 per cell on 80x80, 40x40 and 20x20 grids."""
    assert len(generate_priors((640, 640), MIN_SIZES, STEPS)) == 2 * (80 * 80 + 40 * 40 + 20 * 20) == 16800


def test_encode_decode_roundtrip():
    """Encoding boxes and landmarks and then decoding them gives back the originals."""
    rng = np.random.default_rng(0)
    pri = generate_priors((640, 640), MIN_SIZES, STEPS)[rng.integers(0, 16800, 500)]
    xy = rng.uniform(0, 0.8, (500, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(0.01, 0.2, (500, 2))], 1).astype(np.float32)
    np.testing.assert_allclose(decode(encode(boxes, pri, VAR), pri, VAR), boxes, atol=1e-5)
    lm = rng.uniform(0, 1, (500, 10)).astype(np.float32)
    np.testing.assert_allclose(decode_landm(encode_landm(lm, pri, VAR), pri, VAR), lm, atol=1e-5)


def test_match_forces_best_prior_and_handles_landmark_validity():
    """Every face claims its best anchor, faces without landmarks get no landmark targets, and encoded boxes decode back."""
    priors = generate_priors((640, 640), MIN_SIZES, STEPS)
    boxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.52, 0.53], [0.6, 0.1, 0.9, 0.5]], np.float32)
    lms = np.tile(np.array([[0.4, 0.4]], np.float32), (3, 5))
    has = np.array([True, False, True])
    t = match(boxes, lms, has, priors, VAR)
    ov = iou_matrix(boxes, point_form(priors))
    for g in range(3):
        best = ov[g].argmax()
        assert t.conf[best] == 1  # every GT owns at least its best prior
    assert t.conf.sum() >= 3
    # positives of GT 1 (no landmarks) must not get landmark targets
    owner = ov.argmax(0)
    pos = t.conf == 1
    assert not t.landm_mask[pos & (owner == 1) & (ov.max(0) > 0.5)].any()
    assert t.landm_mask[pos & (owner == 0) & (ov.max(0) > 0.5)].all()
    # decoded loc of positives reproduce their GT
    dec = decode(t.loc[pos], priors[pos], VAR)
    assert iou_matrix(dec, boxes).max(1).min() > 0.999


def test_match_chunking_is_exact():
    """Matching in small chunks of faces gives exactly the same targets as matching all at once."""
    rng = np.random.default_rng(1)
    priors = generate_priors((640, 640), MIN_SIZES, STEPS)
    xy = rng.uniform(0, 0.95, (700, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(0.005, 0.05, (700, 2))], 1).astype(np.float32)
    lms = rng.uniform(0, 1, (700, 10)).astype(np.float32)
    has = rng.random(700) > 0.3
    a = match(boxes, lms, has, priors, VAR, chunk=10_000)
    b = match(boxes, lms, has, priors, VAR, chunk=64)
    for k in ("loc", "conf", "landm", "landm_mask"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


def test_match_empty_and_tiny():
    """An image with no faces, or with only a face far smaller than any anchor, gets no face anchors."""
    priors = generate_priors((640, 640), MIN_SIZES, STEPS)
    t = match(np.zeros((0, 4), np.float32), np.zeros((0, 10), np.float32), np.zeros(0, bool), priors)
    assert t.conf.sum() == 0
    tiny = np.array([[0.5, 0.5, 0.5005, 0.5005]], np.float32)  # far below the smallest anchor
    t = match(tiny, np.zeros((1, 10), np.float32), np.zeros(1, bool), priors)
    assert t.conf.sum() == 0


def test_hflip_swaps_landmarks():
    """Flipping mirrors the box and swaps left/right landmarks, so the left eye stays left of the right eye."""
    img = np.zeros((100, 200, 3), np.uint8)
    box = np.array([[20, 10, 60, 50]], np.float32)
    lm = np.array([[30, 20, 50, 20, 40, 30, 32, 40, 48, 40]], np.float32)  # LE, RE, nose, LM, RM
    _, b, l2, _ = hflip(img, box, lm, np.array([True]))
    np.testing.assert_allclose(b, [[140, 10, 180, 50]])
    l2 = l2.reshape(5, 2)
    assert l2[0, 0] < l2[1, 0] and l2[3, 0] < l2[4, 0]  # left stays left of right after flip
    np.testing.assert_allclose(l2[2], [160, 30])


def test_train_transform_outputs():
    """Over 50 random augmentations, outputs are 640x640 uint8 and all boxes/landmarks stay valid and in 0-1."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (480, 720, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 180, 200], [400, 150, 460, 230], [600, 300, 610, 312]], np.float32)
    lms = np.full((3, 10), -1, np.float32)
    lms[0] = [120, 140, 160, 140, 140, 160, 125, 180, 155, 180]
    has = np.array([True, False, False])
    tf = TrainTransform(640, [0.3, 0.45, 0.6, 0.8, 1.0], seed=0)
    for _ in range(50):
        out, b, lm_out, h = tf(img, boxes, lms, has)
        assert out.shape == (640, 640, 3) and out.dtype == np.uint8
        assert len(b) == len(lm_out) == len(h)
        if len(b):
            assert (b >= -1e-6).all() and (b <= 1 + 1e-6).all()
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
            if h.any():
                assert (lm_out[h] >= 0).all() and (lm_out[h] <= 1 + 1e-6).all()
            assert (lm_out[~h] == -1).all()


def test_background_only_image():
    """An image with no faces passes through augmentation and comes out with no boxes."""
    img = np.zeros((300, 400, 3), np.uint8)
    tf = TrainTransform(640, [0.3, 1.0], seed=0)
    out, b, _, _ = tf(img, np.zeros((0, 4), np.float32), np.zeros((0, 10), np.float32), np.zeros(0, bool))
    assert out.shape == (640, 640, 3) and len(b) == 0


def test_val_transform():
    """Validation transform pads to square, resizes to 640, and normalises boxes to 0-1."""
    img = np.zeros((300, 600, 3), np.uint8)
    out, b, _, _ = ValTransform(640)(
        img, np.array([[0, 0, 300, 150]], np.float32), np.full((1, 10), -1, np.float32), np.array([False])
    )
    assert out.shape == (640, 640, 3)
    np.testing.assert_allclose(b, [[0, 0, 0.5, 0.25]])


def test_parse_label_file(tmp_path):
    """The label parser reads boxes and landmarks, drops zero-size boxes, and keeps images with no faces."""
    f = tmp_path / "label.txt"
    f.write_text(
        "# a/1.jpg\n"
        "10 20 30 40 15 25 0.0 30 25 0.0 22 35 0.0 16 45 0.0 30 45 0.0 0.9\n"
        "50 60 10 12 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"
        "1 1 0 5\n"  # degenerate, dropped
        "# a/2.jpg\n"
        "# b/3.jpg\n"
        "5 5 20 20\n"
    )
    s = parse_label_file(f)
    assert [x.path for x in s] == ["a/1.jpg", "a/2.jpg", "b/3.jpg"]
    np.testing.assert_allclose(s[0].boxes, [[10, 20, 40, 60], [50, 60, 60, 72]])
    assert s[0].has_landmarks.tolist() == [True, False]
    np.testing.assert_allclose(s[0].landmarks[0], [15, 25, 30, 25, 22, 35, 16, 45, 30, 45])
    assert len(s[1].boxes) == 0 and s[2].has_landmarks.tolist() == [False]


def test_average_precision():
    """AP is 1.0 for perfect detections, drops for false positives and duplicates, and ignores tiny faces."""
    gt = {"a": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32), "b": np.zeros((0, 4), np.float32)}
    perfect = {"a": np.array([[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8]], np.float32)}
    assert average_precision(perfect, gt)["ap"] == pytest.approx(1.0)
    with_fp = {"a": perfect["a"], "b": np.array([[0, 0, 5, 5, 0.95]], np.float32)}
    r = average_precision(with_fp, gt)
    assert 0.5 < r["ap"] < 1.0
    dup = {"a": np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.85], [20, 20, 30, 30, 0.8]], np.float32)}
    assert average_precision(dup, gt)["ap"] < 1.0  # duplicate is a FP
    # tiny GT ignored -> detection on it is neither TP nor FP
    gt2 = {"a": np.array([[0, 0, 10, 10], [50, 50, 52, 52]], np.float32)}
    d2 = {"a": np.array([[50, 50, 52, 52, 0.99], [0, 0, 10, 10, 0.5]], np.float32)}
    assert average_precision(d2, gt2, min_gt_px=4)["ap"] == pytest.approx(1.0)


def test_nms():
    """NMS removes the overlapping lower-score box and keeps the separate one."""
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.4).tolist() == [0, 2]


@pytest.mark.parametrize("cfg", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.stem)
def test_configs_load(cfg):
    """Every YAML file in configs/ loads and validates."""
    c = load_config(cfg)
    assert c["data"]["image_size"] % 8 == 0


def test_config_rejects_typos(tmp_path):
    """An unknown config key (a typo) raises KeyError."""
    f = tmp_path / "c.yaml"
    f.write_text("train:\n  lrr: 0.1\n")
    with pytest.raises(KeyError):
        load_config(f)


def test_overrides(tmp_path):
    """Command-line overrides and scientific-notation values like 5e-4 are parsed correctly."""
    c = load_config(ROOT / "configs/resnet34_transfer.yaml", ["train.lr=5e-4", "init.exclude=[body.]"])
    assert c["train"]["lr"] == 5e-4 and c["init"]["exclude"] == ["body."]
    f = tmp_path / "c.yaml"
    f.write_text("train:\n  lr: 5e-4\n")
    assert load_config(f)["train"]["lr"] == 5e-4
