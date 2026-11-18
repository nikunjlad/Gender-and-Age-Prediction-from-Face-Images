"""Training and validation image augmentation, ported from biubug6's data_augment.

Augmentation means randomly changing training images (cropping, recolouring, flipping)
so the model sees more variety and does not memorise the training set.

Training pipeline, in order:
    1. Random square crop that keeps at least one whole face.
    2. Colour distortion: brightness, contrast, saturation and hue.
    3. Horizontal flip, half of the time.
    4. Pad to a square, only if no valid crop was found.
    5. Resize to image_size x image_size.
    6. Divide box and landmark coordinates by the image size, so they range 0 to 1.

Images stay uint8 BGR throughout; pixel normalisation happens inside the model.
"""

from __future__ import annotations

import cv2
import numpy as np

from face_detection.models.targets import iof_matrix

PAD_VALUE = (104, 117, 123)  # BGR mean, used for padding
FLIP_ORDER = [1, 0, 2, 4, 3]  # swap eyes and mouth corners on horizontal flip
INTERPS = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]  # resize methods


def _crop(image, boxes, landms, has_lm, img_dim, scales, min_face_px, rng):
    """Cuts a random square out of the image that keeps at least one whole face.

    The square's side is a random choice from scales times the image's short side. Faces
    whose centre falls outside the square are dropped, and the rest are clipped to it.
    Example: with scales [0.3, ..., 1.0] on a 1024x768 image, a 0.3 crop is 230x230 pixels.

    Args:
        image: Image of shape (H, W, 3).
        boxes: Face boxes of shape (N, 4), xyxy, in pixels.
        landms: Landmarks of shape (N, 10), in pixels.
        has_lm: Shape (N,), bool. True where the face has landmarks.
        img_dim: Final training size, used to judge whether faces end up too small.
        scales: Allowed crop sizes as fractions of the short side.
        min_face_px: Faces smaller than this at training size are dropped.
        rng: numpy random Generator.

    Returns:
        A tuple (image, boxes, landms, has_lm, failed). failed is True when no valid crop
        was found in 250 tries; the inputs are then returned unchanged.
    """
    height, width = image.shape[:2]
    short = min(width, height)
    for _ in range(250):  # try up to 250 random crops
        scale = float(rng.choice(scales))
        w = h = int(scale * short)  # square crop
        if w < 1:
            continue
        left = 0 if width == w else int(rng.integers(0, width - w))  # random left edge
        t = 0 if height == h else int(rng.integers(0, height - h))  # random top edge
        roi = np.array([left, t, left + w, t + h], np.float32)
        if len(boxes) == 0:  # background-only image: any crop works
            return image[t : t + h, left : left + w], boxes, landms, has_lm, False
        if not (iof_matrix(boxes, roi[None]) >= 1).any():
            continue  # need at least one face fully inside the crop
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        keep = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)  # face centre inside the crop
        b = boxes[keep].copy()
        lm = landms[keep].copy().reshape(-1, 5, 2)
        hl = has_lm[keep].copy()
        if len(b) == 0:
            continue
        b[:, :2] = np.maximum(b[:, :2], roi[:2]) - roi[:2]  # clip to crop, shift to crop origin
        b[:, 2:] = np.minimum(b[:, 2:], roi[2:]) - roi[:2]
        lm[hl] = np.clip(lm[hl] - roi[:2], 0, roi[2:] - roi[:2])
        bw = (b[:, 2] - b[:, 0]) / w * img_dim  # face width after resizing to img_dim
        bh = (b[:, 3] - b[:, 1]) / h * img_dim
        ok = np.minimum(bw, bh) >= min_face_px
        if not ok.any():
            continue  # every face would be too small
        return image[t : t + h, left : left + w], b[ok], lm.reshape(-1, 10)[ok], hl[ok], False
    return image, boxes, landms, has_lm, True


def _distort(image: np.ndarray, rng) -> np.ndarray:
    """Randomly changes brightness, contrast, saturation and hue.

    Each change is applied with probability 0.5. Contrast is applied either before or
    after the colour changes, also at random.

    Args:
        image: uint8 BGR image of shape (H, W, 3).
        rng: numpy random Generator.

    Returns:
        The distorted uint8 BGR image.
    """
    img = image.astype(np.float32)

    def brightness(x):
        return x + rng.uniform(-32, 32) if rng.random() < 0.5 else x  # shift all pixels

    def contrast(x):
        return x * rng.uniform(0.5, 1.5) if rng.random() < 0.5 else x  # scale all pixels

    img = brightness(img)
    contrast_first = rng.random() < 0.5
    if contrast_first:
        img = contrast(img)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    if rng.random() < 0.5:
        hsv[..., 1] *= rng.uniform(0.5, 1.5)  # saturation
    if rng.random() < 0.5:
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-18, 19)) % 180  # hue, OpenCV range is 0-179
    hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    if not contrast_first:
        img = contrast(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def hflip(image, boxes, landms, has_lm):
    """Mirrors the image left to right and fixes the labels to match.

    Boxes are mirrored, and the landmarks are mirrored and then reordered, because the
    person's left eye becomes the image's right eye. Example: on a 100-pixel-wide image,
    a box from x 10 to 30 becomes a box from x 70 to 90.

    Args:
        image: Image of shape (H, W, 3).
        boxes: Face boxes of shape (N, 4), xyxy, in pixels.
        landms: Landmarks of shape (N, 10), in pixels.
        has_lm: Shape (N,), bool. True where the face has landmarks.

    Returns:
        The flipped (image, boxes, landms, has_lm).
    """
    w = image.shape[1]
    image = image[:, ::-1]  # reverse columns
    boxes = boxes.copy()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # new x1 = w - old x2, new x2 = w - old x1
    landms = landms.copy().reshape(-1, 5, 2)
    if has_lm.any():
        lm = landms[has_lm]
        lm[..., 0] = w - lm[..., 0]  # mirror x
        landms[has_lm] = lm[:, FLIP_ORDER]  # swap left/right points
    return image, boxes, landms.reshape(-1, 10), has_lm


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pads the image on the bottom or right with the mean colour to make it square.

    Example: a 300x200 (height x width) image becomes 300x300, with the original on the left.
    """
    h, w = image.shape[:2]
    if h == w:
        return image
    s = max(h, w)
    out = np.empty((s, s, 3), image.dtype)
    out[:] = PAD_VALUE
    out[:h, :w] = image  # original in the top-left
    return out


class TrainTransform:
    """The full random training augmentation (see the module docstring for the steps).

    Example:
        tf = TrainTransform(640, [0.3, 0.45, 0.6, 0.8, 1.0])
        img, boxes, landms, has_lm = tf(img, boxes, landms, has_lm)  # img: 640x640, boxes in 0-1

    Args:
        img_dim: Output image side length in pixels.
        crop_scales: Allowed crop sizes as fractions of the short side.
        min_face_px: Faces smaller than this at img_dim are dropped.
        color_jitter: Apply the colour distortion step.
        flip: Apply the random horizontal flip.
        seed: Random seed. Each DataLoader worker reseeds through reseed().
    """

    def __init__(self, img_dim: int, crop_scales, min_face_px: float = 1.0, color_jitter=True, flip=True, seed=None):
        self.img_dim = img_dim
        self.scales = list(crop_scales)
        self.min_face_px = min_face_px
        self.color_jitter = color_jitter
        self.flip = flip
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """Replaces the random generator; called once per DataLoader worker."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, boxes, landms, has_lm):
        """Augments one image and its labels.

        Args:
            image: uint8 BGR image of shape (H, W, 3).
            boxes: Face boxes of shape (N, 4), xyxy, in pixels.
            landms: Landmarks of shape (N, 10), in pixels.
            has_lm: Shape (N,), bool.

        Returns:
            A tuple (image, boxes, landms, has_lm). image is img_dim x img_dim; boxes and
            landmarks are normalised to 0 to 1. Faces may have been dropped.
        """
        rng = self.rng
        image, boxes, landms, has_lm, pad = _crop(
            image, boxes, landms, has_lm, self.img_dim, self.scales, self.min_face_px, rng
        )
        if self.color_jitter:
            image = _distort(image, rng)
        if self.flip and rng.random() < 0.5:
            image, boxes, landms, has_lm = hflip(image, boxes, landms, has_lm)
        if pad:
            image = pad_to_square(image)  # crop failed, so make the whole image square instead
        h, w = image.shape[:2]
        interp = INTERPS[int(rng.integers(0, len(INTERPS)))]  # random resize method
        image = cv2.resize(np.ascontiguousarray(image), (self.img_dim, self.img_dim), interpolation=interp)
        scale = np.array([w, h], np.float32)
        boxes = (boxes.reshape(-1, 2, 2) / scale).reshape(-1, 4).astype(np.float32)  # pixels -> 0-1
        landms = landms.reshape(-1, 5, 2).copy()
        landms[has_lm] = landms[has_lm] / scale
        landms = landms.reshape(-1, 10).astype(np.float32)
        # after a failed crop + pad, drop faces that became too small at training scale
        if len(boxes):
            ok = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * self.img_dim >= self.min_face_px
            boxes, landms, has_lm = boxes[ok], landms[ok], has_lm[ok]
        return np.ascontiguousarray(image), boxes, landms, has_lm


class ValTransform:
    """Fixed, non-random transform for validation: pad to square, then resize.

    Used to compute the validation loss. Example: a 1024x768 image is padded to 1024x1024
    and resized to 640x640.

    Args:
        img_dim: Output image side length in pixels.
    """

    def __init__(self, img_dim: int):
        self.img_dim = img_dim

    def __call__(self, image, boxes, landms, has_lm):
        """Pads and resizes one image. Arguments and return value match TrainTransform."""
        image = pad_to_square(image)
        s = image.shape[0]  # side length after padding
        image = cv2.resize(image, (self.img_dim, self.img_dim), interpolation=cv2.INTER_LINEAR)
        boxes = (boxes / s).astype(np.float32)  # pixels -> 0-1
        landms = landms.reshape(-1, 5, 2).copy()
        landms[has_lm] = landms[has_lm] / s
        return image, boxes, landms.reshape(-1, 10).astype(np.float32), has_lm
