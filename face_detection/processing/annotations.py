"""Reads RetinaFace / WIDER FACE style label files.

The format (retinaface_gt_v1.1, as used by biubug6/Pytorch_Retinaface) has one header
line per image, followed by one line per face:

    # 0--Parade/0_Parade_marchingband_1_849.jpg
    449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82

Each face line starts with the box as x y w h in pixels. It may then have 5 landmarks,
each as x y visibility, and a final score. A landmark x of -1 means "not annotated".
Lines with only 4 numbers (as in the val split) have no landmarks.

Landmark order: left eye, right eye, nose, left mouth corner, right mouth corner, where
left and right are as seen in the image.

Your own data can use the same format. Images with no face lines are kept as
background-only samples, which teach the model what "no face" looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    """The labels of one image.

    Attributes:
        path: Image path, relative to the image root (or absolute).
        boxes: Face boxes of shape (N, 4), xyxy, in pixels.
        landmarks: Shape (N, 10) as (x1, y1, ..., x5, y5) in pixels; -1 where missing.
        has_landmarks: Shape (N,), bool. True where the face has landmark labels.
    """

    path: str  # relative to image root (or absolute)
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy pixels
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, 10), np.float32))
    has_landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))


def _finalize(path: str, rows: list[list[float]]) -> Sample:
    """Turns the parsed number rows of one image into a Sample.

    Args:
        path: Image path from the header line.
        rows: One list of numbers per face line.

    Returns:
        The Sample for that image.
    """
    boxes = np.zeros((len(rows), 4), np.float32)
    landms = np.full((len(rows), 10), -1.0, np.float32)  # -1 = no landmark
    has = np.zeros((len(rows),), bool)
    for i, r in enumerate(rows):
        x, y, w, h = r[:4]
        boxes[i] = (x, y, x + w, y + h)  # xywh -> xyxy
        if len(r) >= 19:  # 4 box + 5 * 3 landmark numbers
            # Take x, y of each landmark and skip the visibility flags at 6, 9, 12, 15, 18.
            lm = np.asarray([r[4], r[5], r[7], r[8], r[10], r[11], r[13], r[14], r[16], r[17]], np.float32)
            landms[i] = lm
            has[i] = bool(lm[0] >= 0)  # -1 means not annotated
    return Sample(path, boxes, landms, has)


def parse_label_file(label_file: str | Path, drop_degenerate: bool = True) -> list[Sample]:
    """Parses a label file into one Sample per image.

    Example: the two lines "# a.jpg" and "10 20 30 40" give one Sample with path "a.jpg"
    and boxes [[10, 20, 40, 60]].

    Args:
        label_file: Path to the label file, for example data/widerface/train/label.txt.
        drop_degenerate: Skip faces with zero or negative width or height.

    Returns:
        A list of Samples in file order.

    Raises:
        ValueError: If a face line appears before any image line, or has fewer than
            4 numbers.
    """
    samples: list[Sample] = []
    cur_path: str | None = None
    rows: list[list[float]] = []
    with open(label_file) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):  # new image: close the previous one
                if cur_path is not None:
                    samples.append(_finalize(cur_path, rows))
                cur_path = line[1:].strip()
                rows = []
                continue
            if cur_path is None:
                raise ValueError(f"{label_file}:{ln}: face line before any '# image' line")
            vals = [float(v) for v in line.split()]
            if len(vals) == 1:  # original WIDER format has a face count line; skip it
                continue
            if len(vals) < 4:
                raise ValueError(f"{label_file}:{ln}: expected at least 4 numbers, got {len(vals)}")
            if drop_degenerate and (vals[2] <= 0 or vals[3] <= 0):
                continue  # zero-size box
            rows.append(vals)
    if cur_path is not None:
        samples.append(_finalize(cur_path, rows))  # close the last image
    return samples


def resolve_image_path(root: str | Path | None, rel: str) -> Path:
    """Joins an image path onto the image root, unless it is already absolute.

    Example: root "data/widerface/val/images" and "0--Parade/a.jpg" give
    "data/widerface/val/images/0--Parade/a.jpg".
    """
    p = Path(rel)
    if p.is_absolute() or root is None:
        return p
    return Path(root) / p


def summarize(samples: list[Sample]) -> str:
    """Returns a one-line dataset summary.

    Example: "12880 images, 159424 faces (78681 with landmarks), 4 background-only images".
    """
    n_img = len(samples)
    n_face = sum(len(s.boxes) for s in samples)
    n_lm = sum(int(s.has_landmarks.sum()) for s in samples)
    n_empty = sum(1 for s in samples if len(s.boxes) == 0)
    return f"{n_img} images, {n_face} faces ({n_lm} with landmarks), {n_empty} background-only images"
