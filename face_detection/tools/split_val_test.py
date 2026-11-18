#!/usr/bin/env python3
"""Carve a local test split out of WIDER FACE val, because the official test labels are not public.

This is a one-off tool. It has already been run: data/widerface/test/split_manifest.json
records the result.

Run from face_detection/:
    python tools/split_val_test.py               # dry run: prints the chosen split
    python tools/split_val_test.py --apply       # moves images, writes GT for val/ and test/

Easy, Medium and Hard are sets of faces, not images, and they are nested (every Easy face
is also Medium and Hard). So the split picks images, chosen so that about 10% of the faces
of each setting land in test:
    It is stratified by the 61 event categories (like the official WIDER split), so every
        event appears in both val and test.
    Inside each event, images are sorted by difficulty (has an Easy face / Medium only /
        Hard only / no faces) and face count, then every k-th image is taken. This keeps
        the difficulty mix the same in both splits.
    Out of --trials random seeds, the one whose Easy/Medium/Hard face shares are closest
        to --frac wins.

Writes, for both data/widerface/val and data/widerface/test:
    ground_truth/wider_{face,easy,medium,hard}_<split>.mat   (official eval-tool format)
    label.txt                                                (retinaface_gt format, boxes only)
    wider_<split>.txt                                        (image list)
The original full-val GT is backed up to val/ground_truth/original_full_val/, and a
manifest (seed, image list, counts) is written to test/split_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

SETTINGS = ["easy", "medium", "hard"]
PER_IMAGE_KEYS = [  # .mat fields that hold one entry per image
    "file_list",
    "face_bbx_list",
    "blur_label_list",
    "expression_label_list",
    "illumination_label_list",
    "invalid_label_list",
    "occlusion_label_list",
    "pose_label_list",
]


def load(gt_dir: Path):
    """Loads the full val ground truth and counts each image's faces per setting.

    Args:
        gt_dir: Folder with wider_face_val.mat and wider_<setting>_val.mat.

    Returns:
        A tuple (full, sets, events, images). full and sets are the loaded .mat dicts,
        events is the list of event names, and images is a list of
        (event_idx, img_idx, rel_path, n_easy, n_medium, n_hard) tuples.
    """
    full = loadmat(gt_dir / "wider_face_val.mat")
    sets = {s: loadmat(gt_dir / f"wider_{s}_val.mat") for s in SETTINGS}
    events = [str(e[0][0]) for e in full["event_list"]]
    images = []  # (event_idx, img_idx, rel_path, n_easy, n_medium, n_hard)
    for i, ev in enumerate(events):
        for j in range(len(full["file_list"][i][0])):
            name = str(full["file_list"][i][0][j][0][0])
            n = [len(sets[s]["gt_list"][i][0][j][0]) for s in SETTINGS]  # faces counted per setting
            images.append((i, j, f"{ev}/{name}.jpg", *n))
    return full, sets, events, images


def choose(images, n_events, frac, seed):
    """Picks the test images for one random seed.

    Each event gets a quota of about frac of its images (at least 1), with leftovers
    handed out by largest remainder so the total is round(frac * N). Within an event,
    images are sorted by difficulty and sampled at even spacing from a random offset.
    Example: an event with 50 images and frac 0.1 gives 5 test images, one from each
    tenth of the difficulty-sorted list.

    Args:
        images: Image tuples from load.
        n_events: Number of events.
        frac: Target share of images for test.
        seed: Random seed for tie-breaking and the sampling offset.

    Returns:
        The list of chosen image tuples.
    """
    rng = np.random.default_rng(seed)
    by_ev = [[im for im in images if im[0] == e] for e in range(n_events)]
    # per-event quota, total = round(frac * N) via largest remainder, >= 1 per event
    raw = np.array([frac * len(v) for v in by_ev])
    q = np.maximum(np.floor(raw).astype(int), 1)
    rem = int(round(frac * len(images))) - q.sum()
    for e in np.argsort(-(raw - np.floor(raw)))[: max(rem, 0)]:
        q[e] += 1
    picked = []
    for v, k in zip(by_ev, q):
        bucket = lambda im: 0 if im[3] else 1 if im[4] else 2 if im[5] else 3  # noqa: E731
        v = sorted(v, key=lambda im: (bucket(im), im[5], rng.random()))  # easiest first, random tie-break
        step = len(v) / k
        off = rng.random() * step  # random start inside the first step
        picked += [v[int(off + t * step)] for t in range(k)]  # evenly spaced picks
    return picked


def face_fracs(picked, images):
    """Returns the share of Easy, Medium and Hard faces that fall in the picked images."""
    tot = np.array([sum(im[3 + s] for im in images) for s in range(3)], float)
    sel = np.array([sum(im[3 + s] for im in picked) for s in range(3)], float)
    return sel / tot


def subset_mat(mat, keep: dict[int, list[int]], keys):
    """Builds a .mat dict that contains only the kept images.

    Args:
        mat: A loaded WIDER .mat dict.
        keep: event_idx -> image indices to keep, in original order.
        keys: The per-image fields to filter.

    Returns:
        A dict ready for scipy.io.savemat, with event_list unchanged.
    """
    out = {}
    n_ev = len(mat["event_list"])
    for k in keys:
        arr = np.empty((n_ev, 1), dtype=object)  # MATLAB cell array layout
        for i in range(n_ev):
            arr[i, 0] = mat[k][i][0][keep[i]]
        out[k] = arr
    out["event_list"] = mat["event_list"]
    return out


def write_split(split_dir: Path, split: str, full, sets, keep, events):
    """Writes the ground-truth .mat files, label.txt and the image list for one split.

    Args:
        split_dir: The split folder, for example data/widerface/test.
        split: "val" or "test", used in file names.
        full: The full val .mat dict.
        sets: Setting name -> its .mat dict.
        keep: event_idx -> image indices in this split.
        events: Event names.
    """
    gt_dir = split_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    savemat(gt_dir / f"wider_face_{split}.mat", subset_mat(full, keep, PER_IMAGE_KEYS))
    for s in SETTINGS:
        savemat(gt_dir / f"wider_{s}_{split}.mat", subset_mat(sets[s], keep, PER_IMAGE_KEYS + ["gt_list"]))
    lines, paths = [], []
    for i, ev in enumerate(events):
        for j in keep[i]:
            rel = f"{ev}/{full['file_list'][i][0][j][0][0]}.jpg"
            paths.append("/" + rel)
            lines.append(f"# {rel}")  # label.txt image header
            for b in full["face_bbx_list"][i][0][j][0].reshape(-1, 4):
                lines.append(" ".join(str(int(v)) for v in b))  # x y w h
    (split_dir / "label.txt").write_text("\n".join(lines) + "\n")
    (split_dir / f"wider_{split}.txt").write_text("\n".join(paths) + "\n")


def main():
    """Finds the best split and, with --apply, moves images and writes the new ground truth."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/widerface")
    ap.add_argument("--frac", type=float, default=0.10)  # target share for test
    ap.add_argument("--trials", type=int, default=500)  # seeds to try
    ap.add_argument("--apply", action="store_true")  # without it, only print the plan
    args = ap.parse_args()

    root = Path(args.root)
    val, test = root / "val", root / "test"
    backup = val / "ground_truth" / "original_full_val"
    if test.exists():
        raise SystemExit(f"{test} already exists; refusing to split twice")
    src_gt = backup if backup.exists() else val / "ground_truth"  # always start from the full val GT
    full, sets, events, images = load(src_gt)

    # Try every seed and keep the one whose face shares are closest to frac.
    best = None
    for seed in range(args.trials):
        p = choose(images, len(events), args.frac, seed)
        dev = np.abs(face_fracs(p, images) - args.frac).max()  # worst setting's distance from frac
        if best is None or dev < best[0]:
            best = (dev, seed, p)
    dev, seed, picked = best
    fr = face_fracs(picked, images)
    print(
        f"seed {seed}: {len(picked)}/{len(images)} images ({len(picked) / len(images):.2%}); face fractions "
        + ", ".join(f"{s} {f:.2%}" for s, f in zip(SETTINGS, fr))
    )
    if not args.apply:
        print("dry run; pass --apply to move files")
        return

    test_set = {(im[0], im[1]) for im in picked}
    keep_test = {i: [j for j in range(len(full["file_list"][i][0])) if (i, j) in test_set] for i in range(len(events))}
    keep_val = {
        i: [j for j in range(len(full["file_list"][i][0])) if (i, j) not in test_set] for i in range(len(events))
    }

    # back up original full-val GT + list once
    if not backup.exists():
        backup.mkdir(parents=True)
        for f in (val / "ground_truth").glob("*.mat"):
            shutil.copy2(f, backup / f.name)
        shutil.copy2(val / "wider_val.txt", backup / "wider_val.txt")

    for im in picked:  # move test images out of val
        dst = test / "images" / im[2]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(val / "images" / im[2]), dst)

    for f in (val / "ground_truth").glob("wider_*_val.mat"):
        f.unlink()  # replaced by the reduced versions below
    write_split(val, "val", full, sets, keep_val, events)
    write_split(test, "test", full, sets, keep_test, events)

    def counts(keep):  # image and face totals for the manifest
        return {
            "images": sum(len(v) for v in keep.values()),
            "faces_all": int(sum(len(full["face_bbx_list"][i][0][j][0]) for i, v in keep.items() for j in v)),
            **{
                f"faces_{s}": int(sum(len(sets[s]["gt_list"][i][0][j][0]) for i, v in keep.items() for j in v))
                for s in SETTINGS
            },
        }

    manifest = {
        "source": "WIDER FACE val (official)",
        "frac": args.frac,
        "seed": seed,
        "method": "event-stratified, difficulty-ordered systematic sampling; best of "
        f"{args.trials} seeds by max |face fraction - frac| over easy/medium/hard",
        "val": counts(keep_val),
        "test": counts(keep_test),
        "test_images": sorted(im[2] for im in picked),
    }
    (test / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("val", "test")}, indent=2))


if __name__ == "__main__":
    main()
