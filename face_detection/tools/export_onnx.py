#!/usr/bin/env python3
"""Export a model to ONNX at a fixed input size, with box and landmark decoding built in.

ONNX is a portable model format that runtimes such as TensorRT and ONNX Runtime can load
without PyTorch.

Run from face_detection/ (PYTHONPATH must include the repo root):
    python tools/export_onnx.py --checkpoint runs/<run>/best.pt --height 640 --width 640 --out retinaface.onnx

Input  "input":     (B, 3, H, W) float32, BGR, 0-255 (normalisation is inside the graph)
Output "boxes":     (B, P, 4) pixel x1 y1 x2 y2
       "scores":    (B, P) face probability
       "landmarks": (B, P, 10) pixel x/y for left eye, right eye, nose, left mouth, right mouth

NMS is not in the graph: use TensorRT EfficientNMS, DeepStream clustering, or your own
post-processing. Build a TensorRT engine afterwards, for example:
    trtexec --onnx=retinaface.onnx --fp16 --saveEngine=retinaface.plan
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from face_detection.models.checkpoint import load_for_inference
from face_detection.config import load_config
from face_detection.models.postprocess import ExportWrapper


def main():
    """Exports the model, then checks the ONNX outputs against PyTorch if onnxruntime is installed."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config")  # needed only for checkpoints without an embedded config
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="retinaface.onnx")
    ap.add_argument("--opset", type=int, default=17)  # ONNX operator set version
    ap.add_argument("--dynamic-batch", action="store_true")  # allow any batch size at runtime
    ap.add_argument("--trust", action="store_true")  # allow unsafe checkpoint loading
    args = ap.parse_args()
    if args.height % 32 or args.width % 32:
        ap.error("height and width must be multiples of 32 (priors assume exact stride-32 grids)")

    cfg = load_config(args.config) if args.config else None
    model, cfg = load_for_inference(args.checkpoint, cfg, "cpu", trust=args.trust)
    wrapper = ExportWrapper(model, cfg, args.height, args.width).eval()
    x = torch.randint(0, 256, (1, 3, args.height, args.width)).float()  # random example image to trace with
    dyn = {"input": {0: "batch"}, "boxes": {0: "batch"}, "scores": {0: "batch"}, "landmarks": {0: "batch"}}
    torch.onnx.export(
        wrapper,
        (x,),
        args.out,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["boxes", "scores", "landmarks"],
        dynamic_axes=dyn if args.dynamic_batch else None,
        do_constant_folding=True,  # pre-compute constant parts such as the anchors
    )
    print(f"wrote {args.out}")

    # Optional check: run the same input through ONNX Runtime and compare with PyTorch.
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping numerical check")
        return
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(x)]
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})
    for name, r, g in zip(["boxes", "scores", "landmarks"], ref, got):
        err = float(np.abs(r - g).max())  # largest difference between the two runtimes
        print(f"  {name}: shape {g.shape}, max abs diff vs torch {err:.2e}")
        if name == "scores" and err > 1e-3:
            print("  WARNING: large score mismatch between ONNX and torch")


if __name__ == "__main__":
    main()
