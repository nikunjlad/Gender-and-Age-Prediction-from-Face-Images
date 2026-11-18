"""Anchor (prior) box generation for RetinaFace.

An anchor, also called a prior, is a fixed box placed on the image before the model
runs. The model does not draw boxes from scratch. Instead, for every anchor it predicts
(1) how likely the anchor contains a face and (2) small offsets that move and resize
the anchor to fit that face.

RetinaFace looks at the image at three scales (pyramid levels). Each level is a grid,
and every grid cell gets one anchor per entry in min_sizes for that level. With the
default config on a 640x640 image:

    level  step  grid     anchor sizes (px)  anchors
    0      8     80x80    16, 32             12,800
    1      16    40x40    64, 128             3,200
    2      32    20x20    256, 512              800
                                       total  16,800

The anchor order must match the order of the model's head outputs exactly, or every
prediction would be compared to the wrong anchor. The order is: level, then grid row,
then grid column, then anchor size. This is the same order as biubug6/Pytorch_Retinaface.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def feature_sizes_for(image_size: tuple[int, int], steps: Sequence[int]) -> list[tuple[int, int]]:
    """Computes the grid size of each pyramid level.

    A level with stride s shrinks the image by a factor of s, rounding up.
    Example: image_size=(640, 480) with steps [8, 16, 32] gives
    [(80, 60), (40, 30), (20, 15)].

    Args:
        image_size: Image size as (height, width) in pixels.
        steps: Stride of each pyramid level, in pixels.

    Returns:
        One (grid_height, grid_width) tuple per level.
    """
    h, w = image_size
    return [(math.ceil(h / s), math.ceil(w / s)) for s in steps]  # round up so edge pixels get a cell


def generate_priors(
    image_size: tuple[int, int],
    min_sizes: Sequence[Sequence[float]],
    steps: Sequence[int],
    clip: bool = False,
    feature_sizes: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Builds every anchor box for an image of the given size.

    Each anchor is centred on a grid cell and is square in pixels. Coordinates are
    normalised to the range 0 to 1 by the image width and height. Example: on a
    640x640 image, the first anchor of level 0 sits at pixel (4, 4) with size 16, so it
    is stored as (0.00625, 0.00625, 0.025, 0.025).

    Args:
        image_size: Image size as (height, width) in pixels.
        min_sizes: Anchor side lengths in pixels, one list per pyramid level.
            Example: [[16, 32], [64, 128], [256, 512]].
        steps: Stride of each pyramid level in pixels. Example: [8, 16, 32].
        clip: If True, clamp all values into the range 0 to 1. Large anchors near
            the border would otherwise extend past the image edge.
        feature_sizes: Grid size of each level. If None, it is computed from
            image_size and steps. Pass it to match the model's real feature maps
            when they differ from the computed sizes.

    Returns:
        A float32 array of shape (num_anchors, 4). Each row is
        (center_x, center_y, width, height), normalised to the image size.
    """
    h, w = image_size
    if feature_sizes is None:
        feature_sizes = feature_sizes_for(image_size, steps)
    levels = []
    for k, step in enumerate(steps):  # one pass per pyramid level
        fh, fw = feature_sizes[k]  # grid rows and columns on this level
        ms = np.asarray(min_sizes[k], dtype=np.float64)  # anchor sizes in pixels
        a = len(ms)  # anchors per grid cell
        ys, xs = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")  # row and column index of every cell
        cx = np.repeat(((xs + 0.5) * step / w)[..., None], a, axis=-1)  # cell centre x, once per anchor size
        cy = np.repeat(((ys + 0.5) * step / h)[..., None], a, axis=-1)  # cell centre y, once per anchor size
        sx = np.broadcast_to(ms / w, (fh, fw, a))  # anchor width, same for every cell
        sy = np.broadcast_to(ms / h, (fh, fw, a))  # anchor height, same for every cell
        levels.append(np.stack([cx, cy, sx, sy], axis=-1).reshape(-1, 4))  # flatten in (row, col, size) order
    priors = np.concatenate(levels, axis=0).astype(np.float32)  # stack levels 0, 1, 2
    if clip:
        np.clip(priors, 0.0, 1.0, out=priors)  # keep boxes inside the image
    return priors
