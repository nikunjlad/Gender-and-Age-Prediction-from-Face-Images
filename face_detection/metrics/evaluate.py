"""WIDER FACE benchmark metrics: official Easy/Medium/Hard AP, COCO-style AP, and P/R/F1.

WIDER FACE grades every image three times, as Easy, Medium and Hard. Each setting has
its own list of faces that count. Faces not on the list (for example tiny or heavily
blurred faces in the Easy setting) are "ignored": finding them earns nothing, and
detections on them are not counted as mistakes. Hard includes the most faces, so it
gives the lowest scores.

Metrics in this module:
    official_ap: A faithful port of the official WIDER evaluation tool, so numbers are
        comparable with papers.
    coco_ap: COCO-style AP (101-point interpolation) at a chosen IoU, for example 0.5
        or 0.75.
    prf_at / best_f1: precision, recall and F1 at one confidence threshold, with
        TP/FP/FN counts.

Box format: WIDER files store boxes as x, y, w, h (top-left corner plus size).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SETTINGS = ["easy", "medium", "hard"]


# ----------------------------------------------------------------------------- ground truth
def load_gt(gt_dir: Path, split: str = "val"):
    """Loads WIDER ground truth from the official .mat files.

    Reads wider_face_<split>.mat for the boxes and wider_<setting>_<split>.mat for each
    setting's list of faces that count.

    Args:
        gt_dir: Folder with the .mat files, for example data/widerface/val/ground_truth.
        split: Split name used in the file names, "val" or "test".

    Returns:
        A dict: setting -> {(event, image name): (boxes, keep)}. boxes has shape (M, 4)
        in xywh pixels; keep has shape (M,) and is True for faces that count.
        Example key: ("0--Parade", "0_Parade_marchingband_1_465").
    """
    from scipy.io import loadmat

    full = loadmat(gt_dir / f"wider_face_{split}.mat")
    events = full["event_list"]  # e.g. "0--Parade"
    files = full["file_list"]  # image names per event
    boxes = full["face_bbx_list"]  # boxes per image
    out = {}
    for s in SETTINGS:
        gl = loadmat(gt_dir / f"wider_{s}_{split}.mat")["gt_list"]  # faces that count in this setting
        d = {}
        for i in range(len(events)):
            ev = str(events[i][0][0])
            for j in range(len(files[i][0])):
                name = str(files[i][0][j][0][0])
                b = boxes[i][0][j][0].astype(np.float64).reshape(-1, 4)
                keep_idx = gl[i][0][j][0].reshape(-1).astype(int)
                keep = np.zeros(len(b), bool)
                if len(keep_idx):
                    keep[keep_idx - 1] = True  # .mat indices start at 1
                d[(ev, name)] = (b, keep)
        out[s] = d
    return out


def load_preds(pred_dir: Path):
    """Loads WIDER-format prediction files written by write_wider_predictions.

    Args:
        pred_dir: Folder with one sub-folder per event, each holding one .txt per image.

    Returns:
        A dict: (event, image name) -> array (N, 5) of x, y, w, h, score, highest score first.
    """
    preds = {}
    for ev_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
        for f in ev_dir.glob("*.txt"):
            lines = f.read_text().strip().split("\n")
            # Line 1 is the image name, line 2 the box count, then one box per line.
            arr = np.array([[float(v) for v in ln.split()[:5]] for ln in lines[2:] if ln.strip()], np.float64).reshape(
                -1, 5
            )
            arr = arr[np.argsort(-arr[:, 4], kind="stable")]  # highest score first
            preds[(ev_dir.name, f.stem)] = arr
    return preds


# ----------------------------------------------------------------------------- official WIDER AP
def _iou_xyxy(a, b):
    """IoU between every pair of xyxy boxes, using the official tool's +1 pixel convention.

    The official tool counts both edge pixels, so a box from x 0 to 9 is 10 pixels wide.
    This is kept for exact agreement with published numbers.
    """
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])  # overlap corners
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(ix2 - ix1 + 1, 0)
    ih = np.maximum(iy2 - iy1 + 1, 0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    ab = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return inter / (aa[:, None] + ab[None, :] - inter)


def _xywh_to_xyxy(b):
    """Converts boxes from x, y, w, h to x1, y1, x2, y2 (returns a copy)."""
    b = b.copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    return b


def _voc_ap(rec, prec):
    """Area under a precision-recall curve, using all-point interpolation (as the WIDER tool does)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # make precision non-increasing
    i = np.where(mrec[1:] != mrec[:-1])[0]  # points where recall changes
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def official_ap(preds, gt_setting, iou_thresh=0.5, thresh_num=1000):
    """Computes the official WIDER AP for one setting (Easy, Medium or Hard).

    A port of the official tool (image_eval, img_pr_info, dataset_pr_info), vectorised
    over thresholds. Scores are first min-max normalised over the whole dataset, then
    precision and recall are measured at thresh_num evenly spaced thresholds. Example:
    biubug6's Resnet50 model scores about 0.95 / 0.94 / 0.84 on Easy / Medium / Hard.

    Args:
        preds: Output of load_preds.
        gt_setting: One setting's entry from load_gt.
        iou_thresh: Minimum IoU for a detection to find a face.
        thresh_num: Number of confidence thresholds.

    Returns:
        A tuple (ap, precision, recall). precision and recall have one value per threshold.
    """
    all_s = np.concatenate([preds[k][:, 4] for k in gt_setting if k in preds and len(preds[k])])
    smin, smax = all_s.min(), all_s.max()
    thresholds = 1 - (np.arange(thresh_num) + 1) / thresh_num  # 0.999 down to 0.0
    pr = np.zeros((thresh_num, 2))  # per threshold: [detections counted, faces found]
    count_face = 0
    for key, (gt, keep) in gt_setting.items():
        count_face += int(keep.sum())
        p = preds.get(key, np.zeros((0, 5)))
        if len(gt) == 0 or len(p) == 0:
            continue
        score = (p[:, 4] - smin) / (smax - smin)  # min-max normalise, as the official tool does
        ov = _iou_xyxy(_xywh_to_xyxy(p[:, :4]), _xywh_to_xyxy(gt))
        recall_list = np.zeros(len(gt))  # per face: 1 found, -1 ignored, 0 not yet found
        proposal = np.ones(len(p))  # per detection: 1 counts, -1 on an ignored face
        pred_recall = np.zeros(len(p))  # faces found after the first h+1 detections
        for h in range(len(p)):
            j = ov[h].argmax()
            if ov[h, j] >= iou_thresh:
                if not keep[j]:
                    recall_list[j] = -1
                    proposal[h] = -1
                elif recall_list[j] == 0:
                    recall_list[j] = 1
            pred_recall[h] = (recall_list == 1).sum()
        # last index with score >= thresh (scores sorted desc)
        n_above = np.searchsorted(-score, -thresholds, side="right")
        cum_prop = np.cumsum(proposal == 1)
        has = n_above > 0
        r_idx = np.maximum(n_above - 1, 0)
        pr[has, 0] += cum_prop[r_idx[has]]
        pr[has, 1] += pred_recall[r_idx[has]]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pr[:, 0] > 0, pr[:, 1] / pr[:, 0], 0.0)
    recall = pr[:, 1] / count_face
    return _voc_ap(recall, precision), precision, recall


# ----------------------------------------------------------------------------- COCO-style / P-R-F1
def match_records(preds, gt_setting, iou_thresh):
    """Matches detections to faces greedily, highest score first, for one setting.

    Each detection takes the best unused face that counts. If it hits nothing that counts
    but overlaps an ignored face, it is dropped (neither TP nor FP).

    Args:
        preds: Output of load_preds.
        gt_setting: One setting's entry from load_gt.
        iou_thresh: Minimum IoU for a match.

    Returns:
        A tuple (scores, tp, n_pos). scores and tp are arrays over all kept detections,
        highest score first; tp is 1 for a true positive and 0 for a false positive.
        n_pos is the number of faces that count.
    """
    scores, tps = [], []
    n_pos = 0
    for key, (gt, keep) in gt_setting.items():
        n_pos += int(keep.sum())
        p = preds.get(key, np.zeros((0, 5)))
        if len(p) == 0:
            continue
        if len(gt) == 0:
            scores.append(p[:, 4])
            tps.append(np.zeros(len(p)))  # no faces: every detection is an FP
            continue
        ov = _iou_xyxy(_xywh_to_xyxy(p[:, :4]), _xywh_to_xyxy(gt))
        used = np.zeros(len(gt), bool)
        s_img, t_img = [], []
        for h in range(len(p)):
            o = ov[h].copy()
            o[used | ~keep] = -1  # best available valid GT
            j = o.argmax()
            if o[j] >= iou_thresh:
                used[j] = True
                s_img.append(p[h, 4])
                t_img.append(1)
                continue
            # overlaps an ignored face -> neither TP nor FP
            if (~keep).any() and ov[h, ~keep].max() >= iou_thresh:
                continue
            s_img.append(p[h, 4])
            t_img.append(0)
        scores.append(np.array(s_img))
        tps.append(np.array(t_img))
    s = np.concatenate(scores) if scores else np.zeros(0)
    t = np.concatenate(tps) if tps else np.zeros(0)
    o = np.argsort(-s, kind="stable")  # all images together, highest score first
    return s[o], t[o], n_pos


def match_image(p, gt, keep, iou_thresh=0.5):
    """Matches one image's detections to its faces, using the same rule as match_records.

    Detections take the best unused face that counts, highest score first. Used by
    evaluate.py --vis to colour the boxes. Example: a face whose best detection has IoU
    0.42 is not found at iou_thresh 0.5, and that detection is a false positive.

    Args:
        p: Detections for the image, shape (N, 5) as x, y, w, h, score, highest score first.
        gt: Faces, shape (M, 4) as x, y, w, h.
        keep: Shape (M,), bool. True for faces that count in the setting.
        iou_thresh: Minimum IoU for a match.

    Returns:
        A tuple (found, det). found has shape (M,), bool: True for faces matched by a
        detection. det has shape (N,), int: 1 for a true positive, 0 for a false
        positive, -1 for a detection on an ignored face (counted as neither).
    """
    found = np.zeros(len(gt), bool)
    det = np.zeros(len(p), int)  # everything starts as a false positive
    if len(gt) == 0 or len(p) == 0:
        return found, det
    ov = _iou_xyxy(_xywh_to_xyxy(p[:, :4]), _xywh_to_xyxy(gt))
    for h in range(len(p)):
        o = ov[h].copy()
        o[found | ~keep] = -1  # best available face that counts
        j = o.argmax()
        if o[j] >= iou_thresh:
            found[j] = True
            det[h] = 1
        elif (~keep).any() and ov[h, ~keep].max() >= iou_thresh:
            det[h] = -1  # on an ignored face
    return found, det


def coco_ap(tp, n_pos):
    """Computes COCO-style AP: average precision at 101 evenly spaced recall levels.

    Example: if precision is 1.0 up to recall 0.5 and the detector never gets further,
    AP is about 0.5.

    Args:
        tp: TP flags from match_records, highest score first.
        n_pos: Number of faces that count.

    Returns:
        AP from 0 to 1.
    """
    if n_pos == 0 or len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_pos
    prec = ctp / (ctp + cfp)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # make precision non-increasing
    rs = np.linspace(0, 1, 101)  # recall levels 0.00, 0.01, ..., 1.00
    idx = np.searchsorted(rec, rs, side="left")  # first detection reaching each level
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))  # unreached levels score 0


def pr_curve_101(tp, n_pos):
    """Returns the smoothed precision-recall curve at the 101 COCO recall levels.

    Uses the same rule as coco_ap: the precision at recall r is the best precision
    reached at recall r or higher, so the curve never rises. Levels the detector never
    reaches are left out. Example: if recall tops out at 0.83, the curve has 84 points,
    for recall 0.00 to 0.83.

    Args:
        tp: TP flags from match_records, highest score first.
        n_pos: Number of faces that count.

    Returns:
        A dict with "recall" and "precision" lists of equal length, ready for JSON.
    """
    if n_pos == 0 or len(tp) == 0:
        return {"recall": [], "precision": []}
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_pos
    prec = ctp / (ctp + cfp)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # make precision non-increasing
    rs = np.linspace(0, 1, 101)  # recall levels 0.00, 0.01, ..., 1.00
    idx = np.searchsorted(rec, rs, side="left")  # first detection reaching each level
    reached = idx < len(prec)  # drop levels the detector never reaches
    return {"recall": np.round(rs[reached], 2).tolist(), "precision": np.round(prec[idx[reached]], 4).tolist()}


def prf_at(scores, tp, n_pos, conf):
    """Computes precision, recall and F1 when keeping detections with score >= conf.

    F1 is the harmonic mean of precision and recall. Example: precision 0.9 and
    recall 0.6 give F1 0.72.

    Args:
        scores: Detection scores from match_records.
        tp: TP flags from match_records.
        n_pos: Number of faces that count.
        conf: Confidence threshold.

    Returns:
        A dict with "conf", "precision", "recall", "f1" and "confusion" (TP, FP, FN
        counts; TN is None because detection has no countable true negatives).
    """
    sel = scores >= conf
    TP = int(tp[sel].sum())
    FP = int(sel.sum() - TP)
    FN = n_pos - TP  # faces never found
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / n_pos if n_pos else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {
        "conf": float(conf),
        "precision": P,
        "recall": R,
        "f1": F1,
        "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": None},
    }


def best_f1(scores, tp, n_pos):
    """Finds the confidence threshold with the highest F1 and returns prf_at there.

    Args:
        scores: Detection scores from match_records, highest first.
        tp: TP flags from match_records.
        n_pos: Number of faces that count.

    Returns:
        The prf_at dict at the best threshold.
    """
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    P = ctp / np.maximum(ctp + cfp, 1)
    R = ctp / n_pos
    F1 = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0)
    # only cut between distinct scores
    last_of_score = np.r_[scores[1:] != scores[:-1], True]
    F1 = np.where(last_of_score, F1, -1)
    k = int(F1.argmax())
    return prf_at(scores, tp, n_pos, scores[k])
