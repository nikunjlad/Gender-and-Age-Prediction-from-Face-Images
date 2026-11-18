#!/usr/bin/env python3
"""Show what a checkpoint contains, and what would transfer into a given config, before training.

Without --config it lists the checkpoint's weights per model part and guesses its
out_channel. With --config it does a dry run of the weight transfer and prints the same
report train.py would print, without training anything.

Run from face_detection/ (PYTHONPATH must include the repo root):
    python tools/inspect_checkpoint.py --checkpoint weights/Resnet50_Final.pth
    python tools/inspect_checkpoint.py --checkpoint weights/Resnet50_Final.pth --config configs/resnet34_transfer.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from face_detection.models.checkpoint import read_checkpoint, transfer_weights
from face_detection.config import load_config
from face_detection.models.model import RetinaFace, key_group


def main():
    """Prints the checkpoint summary and, with --config, the transfer dry run."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", help="dry-run the transfer into this config's model")
    ap.add_argument("--verbose", action="store_true")  # list every tensor
    ap.add_argument("--trust", action="store_true")  # allow unsafe checkpoint loading
    ap.add_argument("overrides", nargs="*")  # dotted config overrides, as in train.py
    args = ap.parse_args()

    # ------------------------------------------------------------------ checkpoint summary
    sd, meta = read_checkpoint(args.checkpoint, trust=args.trust)
    print(f"{args.checkpoint}: {len(sd)} tensors (source: {meta.get('source')})")
    if meta.get("cfg"):  # only checkpoints from this repo have an embedded config
        m = meta["cfg"]["model"]
        print(f"  embedded config: backbone={m['backbone']} out_channel={m['out_channel']} norm={meta.get('norm')}")
    groups = defaultdict(lambda: [0, 0])  # part -> [tensor count, parameter count]
    for k, v in sd.items():
        g = key_group(k)
        groups[g][0] += 1
        groups[g][1] += v.numel()
    for g, (n, numel) in groups.items():
        print(f"  {g:<7} {n:5d} tensors  {numel / 1e6:7.2f}M params")
    # Tensor shapes that reveal the architecture.
    probes = ["fpn.output1.0.weight", "fpn.merge1.0.weight", "ClassHead.0.conv1x1.weight"]
    for k in probes:
        if k in sd:
            print(f"  {k}: {tuple(sd[k].shape)}")
    if "fpn.merge1.0.weight" in sd:
        print(f"  -> out_channel = {sd['fpn.merge1.0.weight'].shape[0]}")  # FPN output channels
    if args.verbose:
        for k, v in sd.items():
            print(f"    {k:70s} {tuple(v.shape)}")

    # ------------------------------------------------------------------ transfer dry run
    if args.config:
        cfg = load_config(  # skip backbone downloads: only the architecture is needed
            args.config, list(args.overrides) + ["model.backbone_pretrained=false", "model.backbone_weights=null"]
        )
        model = RetinaFace(cfg)
        i = cfg["init"]
        rep = transfer_weights(model, sd, i["include"], i["exclude"], i["rename"])
        print()
        print(
            f"dry run into {args.config} (backbone {cfg['model']['backbone']}, out_channel {cfg['model']['out_channel']}):"
        )
        print(rep.format(verbose=args.verbose))
        bad = [g for g in i["require"] if rep.fraction(g) < i["min_fraction"]]  # parts that would abort training
        print(f"\ninit.require {i['require']}: " + (f"WOULD FAIL for {bad}" if bad else "ok"))


if __name__ == "__main__":
    main()
