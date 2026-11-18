#!/usr/bin/env python3
"""Benchmark a model on WIDER FACE and write its metrics.

Two steps:
    1. Inference: run the model on every image of the split and save one WIDER-format
       .txt per image. This step is skipped if the predictions already exist, so
       re-running only recomputes the metrics. The protocol matches biubug6's
       test_widerface.py.
    2. Metrics: compute them per setting (Easy, Medium, Hard) and save
       results/<split>/<name>/metrics.json, including a 101-point PR curve per setting.

Run from face_detection/ (PYTHONPATH must include the repo root):
    python evaluate.py --checkpoint weights/Resnet50_Final.pth \
        --config configs/resnet50_finetune.yaml --name resnet50 --split test

Metrics:
    Official AP (Easy/Medium/Hard): a faithful port of the WIDER evaluation tool used by
        biubug6, so numbers are comparable with papers.
    AP50 / AP75 / AP50:95: COCO-style AP at IoU 0.5, at 0.75, and averaged over 0.5 to
        0.95. With one class (face), AP and mAP are the same.
    Precision / recall / F1 at IoU 0.5, at a fixed confidence (--conf-op) and at the
        confidence with the best F1. The confusion matrix is TP / FP / FN only: detection
        has no countable true negatives, because background is not a set of objects.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from face_detection.models.checkpoint import load_for_inference
from face_detection.config import load_config
from face_detection.metrics.ap import write_wider_predictions
from face_detection.models.postprocess import Detector
from face_detection.metrics.evaluate import (
    SETTINGS,
    best_f1,
    coco_ap,
    load_gt,
    load_preds,
    match_image,
    match_records,
    official_ap,
    pr_curve_101,
    prf_at,
)


# ----------------------------------------------------------------------------- inference
def run_inference(args, pred_dir: Path, image_list: list[str]):
    """Runs the model on every image and writes WIDER-format prediction files.

    Uses biubug6's evaluation settings: a low confidence threshold (0.02) and many
    boxes kept, because AP needs the full range of scores.

    Args:
        args: Parsed command-line arguments (checkpoint, config, device, max_size, images).
        pred_dir: Folder to write the prediction files to.
        image_list: Image paths relative to args.images.

    Raises:
        FileNotFoundError: If an image cannot be read.
    """
    cfg = load_config(args.config, []) if args.config else None
    model, cfg = load_for_inference(args.checkpoint, cfg, args.device)
    det = Detector(model, cfg, conf_thresh=0.02, nms_thresh=0.4, top_k=5000, keep_top_k=750, max_size=args.max_size)
    root = Path(args.images)
    t0 = time.time()
    with torch.inference_mode():
        for i, rel in enumerate(image_list):
            img = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(root / rel)
            out = det(img)
            write_wider_predictions(pred_dir, rel, out)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(image_list)}  {(time.time() - t0) / (i + 1) * 1000:.1f} ms/img", flush=True)
    print(f"inference done: {len(image_list)} images in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- visualisation
GREEN, RED, BLUE = (0, 200, 0), (0, 0, 255), (255, 0, 0)  # BGR


def save_visualizations(gt_setting, preds, image_dir: Path, vis_dir: Path, image_list, conf, iou_thresh=0.5):
    """Draws the outcome of every prediction and label on each image.

    Colours:
        Green: a prediction matched to a labelled face (true positive), with its score.
        Red: a prediction that matches no face (false positive), with its score.
        Blue: the label box of a face the model missed (false negative).
    Label boxes of found faces are not drawn. Ignored faces, and predictions that land on
    them, are not drawn either, so every box matches a TP, FP or FN count in metrics.json.
    Example: an image with 3 faces where the model finds 2 and adds 1 wrong box shows
    2 green, 1 red and 1 blue box.

    Args:
        gt_setting: One setting's entry from load_gt, for example gt["hard"].
        preds: Output of load_preds.
        image_dir: Folder with the split's images.
        vis_dir: Output folder; images are saved as <vis_dir>/<event>/<image>.jpg.
        image_list: Image paths relative to image_dir.
        conf: Only predictions with score >= conf are drawn and matched.
        iou_thresh: Minimum IoU for a match.

    Raises:
        FileNotFoundError: If an image cannot be read.
    """
    for rel in image_list:
        key = (Path(rel).parent.name, Path(rel).stem)  # (event, image name), as in load_gt
        gt, keep = gt_setting.get(key, (np.zeros((0, 4)), np.zeros(0, bool)))
        p = preds.get(key, np.zeros((0, 5)))
        p = p[p[:, 4] >= conf]  # operating threshold; p is already sorted by score
        found, det = match_image(p, gt, keep, iou_thresh)
        img = cv2.imread(str(image_dir / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_dir / rel)
        for (x, y, w, h), k, f in zip(gt.astype(int), keep, found):
            if k and not f:  # missed face; found and ignored faces are not drawn
                cv2.rectangle(img, (x, y), (x + w, y + h), BLUE, 2)
        for (x, y, w, h, s), d in zip(p, det):
            if d < 0:
                continue  # on an ignored face: neither TP nor FP
            color = GREEN if d == 1 else RED
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{s:.2f}", (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        out = vis_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
    print(f"visualisations written to {vis_dir}/")


# ----------------------------------------------------------------------------- main
def main():
    """Parses arguments, runs inference if needed, computes metrics and prints a summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config")
    ap.add_argument("--name", required=True, help="results go to results/<split>/<name>/")
    ap.add_argument("--split", default="val", help="val | test (data/widerface/<split>/)")
    ap.add_argument("--data", default="data/widerface")
    ap.add_argument("--out", default="results", help="results go to <out>/<split>/<name>/")
    ap.add_argument("--preds", help="existing WIDER-format prediction dir to reuse (skips inference if complete)")
    ap.add_argument("--max-size", type=int, default=None, help="default: original resolution (biubug6 protocol)")
    ap.add_argument("--conf-op", type=float, default=0.5, help="operating confidence for P/R/F1")
    ap.add_argument("--rerun", action="store_true", help="redo inference even if predictions exist")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vis", action="store_true", help="also save images with labels vs predictions to <out>/vis/")
    args = ap.parse_args()

    # ------------------------------------------------------------------ step 1: predictions
    split_dir = Path(args.data) / args.split
    args.images = str(split_dir / "images")
    out_dir = Path(args.out) / args.split / args.name
    pred_dir = Path(args.preds) if args.preds else out_dir / "predictions"
    images = [  # image list for the split, e.g. "0--Parade/0_Parade_marchingband_1_465.jpg"
        ln.strip().lstrip("/") for ln in (split_dir / f"wider_{args.split}.txt").read_text().splitlines() if ln.strip()
    ]
    missing = [r for r in images if not (pred_dir / Path(r).with_suffix(".txt")).exists()]
    if args.preds and (args.rerun or missing):
        raise SystemExit(f"--preds {pred_dir} is missing {len(missing)} images of {args.split}")
    if args.rerun or missing:
        print(f"running inference on {len(images)} images ({args.device})")
        run_inference(args, pred_dir, images)
    else:
        print(f"using cached predictions in {pred_dir}")

    # ------------------------------------------------------------------ step 2: metrics
    gt = load_gt(split_dir / "ground_truth", args.split)
    keys = set(gt["hard"])  # every image of the split
    preds = {k: v for k, v in load_preds(pred_dir).items() if k in keys}
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_images": len(images),
        "max_size": args.max_size,
        "settings": {},
    }
    for s in SETTINGS:
        a, prec, rec = official_ap(preds, gt[s])
        r = {"official_ap": a}
        ious = np.round(np.arange(0.5, 0.96, 0.05), 2)  # 0.50, 0.55, ..., 0.95
        aps = {}
        for t in ious:
            sc, tp, n_pos = match_records(preds, gt[s], float(t))
            aps[float(t)] = coco_ap(tp, n_pos)
            if t == 0.5:  # P/R/F1 and the PR curve use IoU 0.5 only
                r["num_gt"] = n_pos
                r["at_conf_op"] = prf_at(sc, tp, n_pos, args.conf_op)
                r["best_f1"] = best_f1(sc, tp, n_pos)
                r["pr_curve"] = pr_curve_101(tp, n_pos)  # 101-point curve, read by tools/plot_pr.py
        r["ap50"], r["ap75"] = aps[0.5], aps[0.75]
        r["ap50_95"] = float(np.mean(list(aps.values())))
        res["settings"][s] = r

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    if args.vis:  # Hard setting, IoU 0.5, at the operating confidence
        save_visualizations(gt["hard"], preds, Path(args.images), out_dir / "vis", images, args.conf_op)

    # ------------------------------------------------------------------ summary table
    print(f"\n== {args.name} on {args.split} ({len(images)} images) ==  (IoU 0.5 for P/R/F1)")
    hdr = (
        f"{'setting':8} {'officialAP':>10} {'AP50':>6} {'AP75':>6} {'AP50:95':>7} | "
        f"{'P@'+str(args.conf_op):>7} {'R':>6} {'F1':>6} | {'bestF1':>6} {'@conf':>6}"
    )
    print(hdr)
    for s, r in res["settings"].items():
        o, b = r["at_conf_op"], r["best_f1"]
        print(
            f"{s:8} {r['official_ap']:10.4f} {r['ap50']:6.4f} {r['ap75']:6.4f} {r['ap50_95']:7.4f} | "
            f"{o['precision']:7.4f} {o['recall']:6.4f} {o['f1']:6.4f} | {b['f1']:6.4f} {b['conf']:6.3f}"
        )
    print("\nConfusion matrix @ conf", args.conf_op, "(TN undefined for detection)")
    for s, r in res["settings"].items():
        c = r["at_conf_op"]["confusion"]
        print(f"  {s:7}  TP={c['TP']:6d}  FP={c['FP']:6d}  FN={c['FN']:6d}   (GT faces={r['num_gt']})")
    print(f"\nwritten: {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
