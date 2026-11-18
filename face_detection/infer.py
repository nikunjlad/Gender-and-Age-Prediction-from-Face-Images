#!/usr/bin/env python3
"""Detect faces in one image or a folder of images.

For every image it saves a copy with boxes, scores and landmarks drawn on it, and adds
one line to detections.jsonl in the output folder.

Run from face_detection/ (PYTHONPATH must include the repo root):
    python infer.py --checkpoint runs/<run>/best.pt --source samples/ --out out/ --conf 0.6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from face_detection.models.checkpoint import load_for_inference
from face_detection.config import load_config
from face_detection.models.postprocess import Detector

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # image file types picked up from a folder
LM_COLORS = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR colour per landmark


def draw(img, dets):
    """Draws boxes, scores and landmarks onto the image (in place).

    Args:
        img: BGR image.
        dets: Detections of shape (N, 15) from Detector.

    Returns:
        The same image, for convenience.
    """
    for d in dets:
        x1, y1, x2, y2, s = d[:5]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # green box
        cv2.putText(img, f"{s:.2f}", (int(x1), max(int(y1) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for k in range(5):
            cv2.circle(img, (int(d[5 + 2 * k]), int(d[6 + 2 * k])), 2, LM_COLORS[k], -1)  # landmark k
    return img


def main():
    """Parses arguments, loads the model and runs detection on every image."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)  # model file
    ap.add_argument("--config")  # needed only for checkpoints without an embedded config (e.g. biubug6)
    ap.add_argument("--source", required=True)  # an image file or a folder
    ap.add_argument("--out", default="detections")  # output folder
    ap.add_argument("--conf", type=float, default=0.5)  # minimum face score to keep
    ap.add_argument("--nms", type=float, default=None)  # NMS IoU threshold; None = config value
    ap.add_argument("--max-size", type=int, default=None)  # shrink so the long side fits; None = full size
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trust", action="store_true")  # allow unsafe checkpoint loading
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else None
    model, cfg = load_for_inference(args.checkpoint, cfg, args.device, trust=args.trust)
    det = Detector(model, cfg, conf_thresh=args.conf, nms_thresh=args.nms, max_size=args.max_size)

    src = Path(args.source)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS) if src.is_dir() else [src]  # folder: recurse
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "detections.jsonl", "w") as jf:
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                print(f"skip unreadable {p}")
                continue
            t0 = time.time()
            dets = det(img)
            ms = (time.time() - t0) * 1000  # detection time in milliseconds
            jf.write(  # one JSON object per image
                json.dumps(
                    {
                        "image": str(p),
                        "faces": [
                            {
                                "box": d[:4].round(1).tolist(),
                                "score": round(float(d[4]), 4),
                                "landmarks": d[5:].reshape(5, 2).round(1).tolist(),
                            }
                            for d in dets
                        ],
                    }
                )
                + "\n"
            )
            cv2.imwrite(str(out / p.name), draw(img, dets))  # annotated copy
            print(f"{p.name}: {len(dets)} faces, {ms:.1f} ms")


if __name__ == "__main__":
    main()
