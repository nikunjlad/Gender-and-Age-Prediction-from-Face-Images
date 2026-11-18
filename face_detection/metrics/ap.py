"""Quick average precision (AP) for checkpoint selection, plus the WIDER prediction file writer.

Average precision summarises a detector in one number from 0 to 1. Detections are sorted
by confidence. Walking down that list, precision is the share of detections so far that
are real faces, and recall is the share of all real faces found so far. AP is the area
under the precision-versus-recall curve. Example: a detector that finds every face
before making any mistake has AP 1.0.

average_precision gives one AP over all faces, which is fast and good for picking the
best epoch during training. For the official WIDER Easy/Medium/Hard numbers, use
evaluate.py (metrics/evaluate.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_detection.models.targets import iou_matrix


def average_precision(
    detections: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    iou_thresh: float = 0.5,
    min_gt_px: float = 0.0,
) -> dict[str, float]:
    """Computes single-class AP over all images.

    A detection is a true positive (TP) if it overlaps an unused labelled face by at least
    iou_thresh IoU; otherwise it is a false positive (FP). Labelled faces whose shorter
    side is below min_gt_px are "ignore" regions: detections on them count as neither TP
    nor FP. Example: with min_gt_px 10, a detection on a 6-pixel face is not penalised.

    Args:
        detections: Image name -> array (N, 5) of x1, y1, x2, y2, score.
        ground_truth: Image name -> array (M, 4) of x1, y1, x2, y2.
        iou_thresh: Minimum IoU for a detection to count as finding a face.
        min_gt_px: Faces smaller than this (shorter side, in pixels) are ignored.

    Returns:
        A dict with "ap", "recall" (the final recall), "num_gt" (faces counted) and
        "num_det" (detections counted).
    """
    records = []  # (score, tp, ignore)
    n_pos = 0  # number of faces that count
    for img, gt in ground_truth.items():
        gt = np.asarray(gt, np.float32).reshape(-1, 4)
        ign = np.minimum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) < min_gt_px  # too-small faces
        n_pos += int((~ign).sum())
        det = np.asarray(detections.get(img, np.zeros((0, 5))), np.float32).reshape(-1, 5)
        if len(det) == 0:
            continue
        det = det[np.argsort(-det[:, 4])]  # highest score first
        if len(gt) == 0:
            records += [(s, 0, 0) for s in det[:, 4]]  # no faces: every detection is an FP
            continue
        ov = iou_matrix(det[:, :4], gt)
        used = np.zeros(len(gt), bool)  # each face can be found only once
        for i in range(len(det)):
            # prefer unused non-ignored GT, then ignored GT
            cand = ov[i].copy()
            cand[used] = -1
            j = int(cand.argmax())
            if cand[j] >= iou_thresh:
                if ign[j]:
                    records.append((det[i, 4], 0, 1))  # on an ignored face
                else:
                    used[j] = True
                    records.append((det[i, 4], 1, 0))  # TP
            else:
                records.append((det[i, 4], 0, 0))  # FP
    if n_pos == 0:
        return {"ap": 0.0, "recall": 0.0, "num_gt": 0, "num_det": len(records)}
    rec = np.asarray([r for r in records if not r[2]], np.float64).reshape(-1, 3)  # drop ignored
    if len(rec) == 0:
        return {"ap": 0.0, "recall": 0.0, "num_gt": n_pos, "num_det": 0}
    rec = rec[np.argsort(-rec[:, 0], kind="stable")]  # all images together, highest score first
    tp = np.cumsum(rec[:, 1])  # running TP count
    fp = np.cumsum(1 - rec[:, 1])  # running FP count
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # make precision non-increasing (all-point interpolation)
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]  # points where recall changes
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))  # area under the curve
    return {"ap": ap, "recall": float(recall[-1]), "num_gt": n_pos, "num_det": int(len(rec))}


def write_wider_predictions(out_dir: str | Path, rel_path: str, dets: np.ndarray) -> None:
    """Writes one image's detections in the official WIDER FACE submission format.

    The file is out_dir/<event>/<image name>.txt with the image name, the number of boxes,
    then one "x y w h score" line per box. Example: "0--Parade/a.jpg" is written to
    out_dir/0--Parade/a.txt.

    Args:
        out_dir: Root folder for prediction files.
        rel_path: Image path relative to the image root.
        dets: Detections of shape (N, 5+), x1, y1, x2, y2, score first.
    """
    rel = Path(rel_path)
    target = Path(out_dir) / rel.parent / (rel.stem + ".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        f.write(f"{rel.stem}\n{len(dets)}\n")
        for d in dets:
            x1, y1, x2, y2, s = d[:5]
            f.write(f"{x1:.1f} {y1:.1f} {x2 - x1 + 1:.1f} {y2 - y1 + 1:.1f} {s:.5f}\n")  # WIDER uses +1 width


def evaluate_ap(model, cfg, samples, root, max_size=None, out_dir=None, conf_thresh=None, progress=False):
    """Runs the model over labelled images and returns average_precision.

    Used by train.py after each epoch to pick the best checkpoint. Example: with 500
    validation samples it returns {"ap": 0.87, "recall": 0.91, ...}.

    Args:
        model: The RetinaFace model.
        cfg: The full config. Uses the "eval" section for iou_thresh and min_gt_px.
        samples: Labelled images from parse_label_file.
        root: Image root folder.
        max_size: If set, shrink images so the long side is at most this.
        out_dir: If set, also write WIDER prediction files here.
        conf_thresh: Override for the detector's confidence threshold.
        progress: Print speed every 100 images.

    Returns:
        The dict from average_precision.

    Raises:
        FileNotFoundError: If an image cannot be read.
    """
    import time

    import cv2

    from face_detection.models.postprocess import Detector
    from face_detection.processing.annotations import resolve_image_path

    det = Detector(model, cfg, max_size=max_size, conf_thresh=conf_thresh)
    dets, gts = {}, {}
    t0 = time.time()
    for i, s in enumerate(samples):
        img = cv2.imread(str(resolve_image_path(root, s.path)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(resolve_image_path(root, s.path))
        out = det(img)
        dets[s.path] = out[:, :5]  # box + score only
        gts[s.path] = s.boxes
        if out_dir:
            write_wider_predictions(out_dir, s.path, out)
        if progress and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)}  {(time.time() - t0) / (i + 1) * 1000:.1f} ms/img")
    e = cfg["eval"]
    return average_precision(dets, gts, e["iou_thresh"], e["min_gt_px"])
