"""Box math and training-target assignment for RetinaFace (numpy only).

During training, the loss needs to know, for every anchor, whether it should predict
"face" or "background", and if "face", which offsets would move it onto the real box.
These answers are called targets. This module builds them.

Targets are computed inside the DataLoader worker processes, so the GPU training loop
only receives ready-made arrays. Box offsets use the SSD/RetinaFace encoding with
variances (0.1, 0.2). The variances are scaling factors that keep the offsets near the
range -1 to 1, which makes them easier for the model to learn.

Box formats used here:
    xyxy: (x1, y1, x2, y2), the top-left and bottom-right corners.
    cxcywh: (center_x, center_y, width, height). Anchors use this format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------- box helpers


def point_form(priors: np.ndarray) -> np.ndarray:
    """Converts boxes from cxcywh to xyxy format.

    Example: (0.5, 0.5, 0.2, 0.4) becomes (0.4, 0.3, 0.6, 0.7).

    Args:
        priors: Array of shape (N, 4) in cxcywh format.

    Returns:
        Array of shape (N, 4) in xyxy format.
    """
    return np.concatenate([priors[:, :2] - priors[:, 2:] / 2, priors[:, :2] + priors[:, 2:] / 2], axis=1)


def box_area(b: np.ndarray) -> np.ndarray:
    """Computes the area of each xyxy box.

    Boxes with negative width or height count as area 0.

    Args:
        b: Array of shape (N, 4) in xyxy format.

    Returns:
        Array of shape (N,) with one area per box.
    """
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes intersection over union (IoU) between every pair of boxes.

    IoU is the overlap area divided by the combined area of both boxes. It is 1 for
    identical boxes and 0 for boxes that do not touch. Example: (0, 0, 10, 10) and
    (5, 0, 15, 10) overlap by 50 out of a combined 150, so IoU is 0.333.

    Args:
        a: Array of shape (N, 4) in xyxy format.
        b: Array of shape (M, 4) in xyxy format.

    Returns:
        A float32 array of shape (N, M), where entry [i, j] is the IoU of a[i] and b[j].
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # top-left corner of each overlap
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # bottom-right corner of each overlap
    wh = np.clip(rb - lt, 0, None)  # overlap width and height, 0 if no overlap
    inter = wh[..., 0] * wh[..., 1]  # overlap area
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter  # combined area
    return inter / np.maximum(union, 1e-12)  # avoid divide-by-zero for empty boxes


def iof_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes intersection over foreground: the share of each box in a covered by b.

    Unlike IoU, this divides by the area of a only. It answers "how much of this face is
    still inside the crop?" Example: a face (0, 0, 10, 10) and a crop (5, 0, 15, 10)
    overlap by 50 of the face's 100, so IoF is 0.5.

    Args:
        a: Array of shape (N, 4) in xyxy format, for example face boxes.
        b: Array of shape (M, 4) in xyxy format, for example crop windows.

    Returns:
        Array of shape (N, M), where entry [i, j] is the share of a[i] inside b[j].
    """
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # top-left corner of each overlap
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # bottom-right corner of each overlap
    wh = np.clip(rb - lt, 0, None)  # overlap width and height
    inter = wh[..., 0] * wh[..., 1]  # overlap area
    return inter / np.maximum(box_area(a)[:, None], 1e-12)  # divide by the area of a only


# ----------------------------------------------------------------------------- encode / decode


def encode(matched: np.ndarray, priors: np.ndarray, variance) -> np.ndarray:
    """Encodes face boxes as offsets from their anchors. This is what the model learns.

    The centre shift is divided by the anchor size, and the size change is stored as a
    log ratio. Both are then divided by the variances. Example: anchor
    (0.5, 0.5, 0.1, 0.1) and face (0.46, 0.45, 0.56, 0.55) have a centre shift of 0.01
    in x, which encodes to (1.0, 0.0, 0.0, 0.0).

    Args:
        matched: Face boxes of shape (N, 4) in xyxy format, one per anchor.
        priors: Anchors of shape (N, 4) in cxcywh format.
        variance: The two scaling factors, for example (0.1, 0.2).

    Returns:
        A float32 array of shape (N, 4) with (dx, dy, dw, dh) per anchor.
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # centre shift
    g_cxcy /= variance[0] * priors[:, 2:]  # relative to anchor size
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # size ratio face / anchor
    g_wh = np.log(np.maximum(g_wh, 1e-6)) / variance[1]  # log ratio, guarded against log(0)
    return np.concatenate([g_cxcy, g_wh], axis=1).astype(np.float32)


def decode(loc: np.ndarray, priors: np.ndarray, variance) -> np.ndarray:
    """Turns predicted offsets back into boxes. This is the reverse of encode.

    Args:
        loc: Predicted offsets of shape (N, 4).
        priors: Anchors of shape (N, 4) in cxcywh format.
        variance: The two scaling factors, for example (0.1, 0.2).

    Returns:
        Boxes of shape (N, 4) in xyxy format.
    """
    cxcy = priors[:, :2] + loc[:, :2] * variance[0] * priors[:, 2:]  # shifted centre
    wh = priors[:, 2:] * np.exp(loc[:, 2:] * variance[1])  # resized width and height
    return np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)  # back to corners


def encode_landm(matched: np.ndarray, priors: np.ndarray, variance) -> np.ndarray:
    """Encodes the 5 facial landmarks as offsets from their anchor's centre.

    The 5 landmarks are left eye, right eye, nose, left mouth corner and right mouth
    corner. Each point is stored as its shift from the anchor centre, divided by the
    anchor size and the first variance.

    Args:
        matched: Landmarks of shape (N, 10) as (x1, y1, ..., x5, y5), one row per anchor.
        priors: Anchors of shape (N, 4) in cxcywh format.
        variance: The two scaling factors. Only the first is used.

    Returns:
        A float32 array of shape (N, 10).
    """
    m = matched.reshape(-1, 5, 2)  # (N, 5 points, xy)
    out = (m - priors[:, None, :2]) / (variance[0] * priors[:, None, 2:])  # shift from centre, scaled
    return out.reshape(-1, 10).astype(np.float32)


def decode_landm(pre: np.ndarray, priors: np.ndarray, variance) -> np.ndarray:
    """Turns predicted landmark offsets back into points. This is the reverse of encode_landm.

    Args:
        pre: Predicted landmark offsets of shape (N, 10).
        priors: Anchors of shape (N, 4) in cxcywh format.
        variance: The two scaling factors. Only the first is used.

    Returns:
        Landmarks of shape (N, 10) as (x1, y1, ..., x5, y5).
    """
    p = pre.reshape(-1, 5, 2)  # (N, 5 points, xy)
    out = priors[:, None, :2] + p * variance[0] * priors[:, None, 2:]  # anchor centre + scaled shift
    return out.reshape(-1, 10)


# ----------------------------------------------------------------------------- matching


@dataclass
class Targets:
    """Training targets for one image, one row per anchor.

    Attributes:
        loc: Encoded box offsets, shape (P, 4), float32. Only used where conf is 1.
        conf: Class per anchor, shape (P,), int64. 0 is background and 1 is face.
        landm: Encoded landmark offsets, shape (P, 10), float32.
        landm_mask: Shape (P,), bool. True for face anchors whose face has labelled
            landmarks. Many WIDER faces have no landmark labels.
    """

    loc: np.ndarray  # (P, 4) float32
    conf: np.ndarray  # (P,) int64, 0 = background, 1 = face
    landm: np.ndarray  # (P, 10) float32
    landm_mask: np.ndarray  # (P,) bool, positives whose GT has landmarks

    def as_dict(self) -> dict[str, np.ndarray]:
        """Returns the targets as a dict, the form the DataLoader batches together."""
        return {"loc": self.loc, "conf": self.conf, "landm": self.landm, "landm_mask": self.landm_mask}


def match(
    gt_boxes: np.ndarray,
    gt_landms: np.ndarray,
    gt_has_landm: np.ndarray,
    priors: np.ndarray,
    variance=(0.1, 0.2),
    pos_iou: float = 0.35,
    min_gt_best_iou: float = 0.2,
    chunk: int = 256,
) -> Targets:
    """Assigns each anchor to a labelled face, or to background, using the RetinaFace rules.

    The rules are:
        1. An anchor becomes a face anchor (positive) if its IoU with the closest face is
           at least pos_iou.
        2. Every face also claims its single best anchor, even below pos_iou, so small
           faces are not left without an anchor. This only happens if that best IoU is at
           least min_gt_best_iou.
        3. Faces that cannot reach min_gt_best_iou with any anchor are ignored. They are
           usually too small or oddly shaped to learn from.

    Example: a 40x40-pixel face whose best anchor has IoU 0.3 gets that anchor by rule 2,
    even though 0.3 is below pos_iou of 0.35.

    IoU is computed for chunk faces at a time, which caps memory use on crowded images.
    Some WIDER images have about 2,000 faces.

    Args:
        gt_boxes: Labelled faces of shape (G, 4), xyxy, normalised to 0 to 1.
        gt_landms: Labelled landmarks of shape (G, 10), normalised to 0 to 1.
        gt_has_landm: Shape (G,), bool. True where the face has landmark labels.
        priors: Anchors of shape (P, 4) in cxcywh format.
        variance: The two encoding scaling factors.
        pos_iou: Minimum IoU for an anchor to count as a face anchor.
        min_gt_best_iou: Minimum IoU a face needs with its best anchor to be used at all.
        chunk: Number of faces per IoU batch.

    Returns:
        A Targets object with one row per anchor.
    """
    num_p = priors.shape[0]  # number of anchors
    loc_t = np.zeros((num_p, 4), np.float32)
    conf_t = np.zeros((num_p,), np.int64)  # everything starts as background
    landm_t = np.zeros((num_p, 10), np.float32)
    landm_mask = np.zeros((num_p,), bool)
    g = gt_boxes.shape[0]  # number of faces
    if g == 0:
        return Targets(loc_t, conf_t, landm_t, landm_mask)  # image with no faces: all background

    pf = point_form(priors)  # anchors as corners, for IoU
    best_truth_ov = np.full((num_p,), -1.0, np.float32)  # per anchor: best IoU with any face
    best_truth_idx = np.zeros((num_p,), np.int64)  # per anchor: index of that face
    best_prior_ov = np.empty((g,), np.float32)  # per face: best IoU with any anchor
    best_prior_idx = np.empty((g,), np.int64)  # per face: index of that anchor
    for s in range(0, g, chunk):
        ov = iou_matrix(gt_boxes[s : s + chunk], pf)  # (c, P)
        best_prior_idx[s : s + chunk] = ov.argmax(1)  # best anchor for each face in this chunk
        best_prior_ov[s : s + chunk] = ov.max(1)
        c_idx = ov.argmax(0)  # best face in this chunk for each anchor
        c_ov = ov[c_idx, np.arange(num_p)]
        better = c_ov > best_truth_ov  # anchors where this chunk beats earlier chunks
        best_truth_ov[better] = c_ov[better]
        best_truth_idx[better] = c_idx[better] + s  # chunk index -> global face index

    valid = np.nonzero(best_prior_ov >= min_gt_best_iou)[0]  # faces usable under rule 3
    if valid.size == 0:
        return Targets(loc_t, conf_t, landm_t, landm_mask)
    # Rule 2: force-match each face to its best anchor (later faces win ties, as in the reference code).
    for gi in valid:
        p = best_prior_idx[gi]
        best_truth_idx[p] = gi
        best_truth_ov[p] = 2.0  # above any real IoU, so rule 1 always keeps it

    pos = best_truth_ov >= pos_iou  # rule 1: face anchors
    idx = best_truth_idx[pos]  # which face each face anchor belongs to
    loc_t[pos] = encode(gt_boxes[idx], priors[pos], variance)
    conf_t[pos] = 1
    lm_ok = gt_has_landm[idx]  # face anchors whose face has landmarks
    pos_idx = np.nonzero(pos)[0]
    lm_priors = pos_idx[lm_ok]
    if lm_priors.size:
        landm_t[lm_priors] = encode_landm(gt_landms[idx[lm_ok]], priors[lm_priors], variance)
        landm_mask[lm_priors] = True
    return Targets(loc_t, conf_t, landm_t, landm_mask)


class TargetAssigner:
    """Stores the anchor settings and runs match for each training image.

    The dataset calls this object once per image. Example:
    assigner(boxes, landms, has_landm) returns a Targets object for that image.

    Args:
        priors: Anchors of shape (P, 4) in cxcywh format.
        variance: The two encoding scaling factors.
        pos_iou: Minimum IoU for an anchor to count as a face anchor.
        min_gt_best_iou: Minimum IoU a face needs with its best anchor to be used.
    """

    def __init__(self, priors: np.ndarray, variance, pos_iou: float, min_gt_best_iou: float):
        self.priors = priors.astype(np.float32)
        self.variance = tuple(variance)
        self.pos_iou = pos_iou
        self.min_gt_best_iou = min_gt_best_iou

    def __call__(self, boxes: np.ndarray, landms: np.ndarray, has_landm: np.ndarray) -> Targets:
        """Builds targets for one image. See match for the arguments."""
        return match(boxes, landms, has_landm, self.priors, self.variance, self.pos_iou, self.min_gt_best_iou)


# ----------------------------------------------------------------------------- nms


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    """Removes duplicate boxes with greedy non-maximum suppression (NMS).

    NMS keeps the highest-scoring box, deletes every other box that overlaps it by more
    than thresh IoU, and repeats. Example: two boxes on the same face with scores 0.9 and
    0.8 and IoU 0.7 become one box (the 0.9 one) when thresh is 0.4.

    The detector uses torchvision's version. This numpy version is kept as a
    reference that the tests compare against.

    Args:
        boxes: Array of shape (N, 4) in xyxy format.
        scores: Array of shape (N,) with one confidence per box.
        thresh: IoU above which a lower-scoring box is removed.

    Returns:
        Indices of the kept boxes, highest score first.
    """
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]  # box indices, highest score first
    keep = []
    while order.size:
        i = order[0]  # best remaining box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  # overlap with every other remaining box
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-12)  # IoU with the kept box
        order = order[1:][ov <= thresh]  # drop boxes that overlap too much
    return np.asarray(keep, dtype=np.int64)
