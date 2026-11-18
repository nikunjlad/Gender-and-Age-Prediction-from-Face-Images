"""Inference helpers: turn raw model outputs into face boxes and landmarks.

The model only outputs offsets and scores for each anchor. To get real detections this
module:
    1. Pads the image so its height and width are multiples of 32.
    2. Decodes the offsets into boxes and landmarks using the anchors.
    3. Drops low-confidence anchors and removes duplicates with NMS.

Detector does all of this for a single image in Python. ExportWrapper does only step 2
inside the model graph, for ONNX/TensorRT export, and leaves NMS to the runtime.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from .anchors import generate_priors
from face_detection.processing.transforms import PAD_VALUE


def decode_boxes(loc: torch.Tensor, priors: torch.Tensor, var) -> torch.Tensor:
    """Turns predicted box offsets into boxes. Torch version of targets.decode.

    Works on any number of leading dimensions, so it handles both (P, 4) and (B, P, 4).

    Args:
        loc: Predicted offsets of shape (..., 4).
        priors: Anchors of shape (..., 4) in cxcywh format, normalised to 0 to 1.
        var: The two encoding scaling factors, for example (0.1, 0.2).

    Returns:
        Boxes of shape (..., 4) in xyxy format, normalised to 0 to 1.
    """
    cxcy = priors[..., :2] + loc[..., :2] * var[0] * priors[..., 2:]  # shifted centre
    wh = priors[..., 2:] * torch.exp(loc[..., 2:] * var[1])  # resized width and height
    return torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=-1)  # back to corners


def decode_landms(pre: torch.Tensor, priors: torch.Tensor, var) -> torch.Tensor:
    """Turns predicted landmark offsets into points. Torch version of targets.decode_landm.

    Args:
        pre: Predicted landmark offsets of shape (..., 10).
        priors: Anchors of shape (..., 4) in cxcywh format.
        var: The two encoding scaling factors. Only the first is used.

    Returns:
        Landmarks of shape (..., 10) as (x1, y1, ..., x5, y5), normalised to 0 to 1.
    """
    shp = pre.shape
    p = pre.reshape(*shp[:-1], 5, 2)  # split into 5 (x, y) points
    out = priors[..., None, :2] + p * var[0] * priors[..., None, 2:]  # anchor centre + scaled shift
    return out.reshape(shp)


def pad_to_multiple(img: np.ndarray, m: int = 32) -> np.ndarray:
    """Pads an image on the bottom and right so both sides are multiples of m.

    The model shrinks the image by up to 32 times, so sides that are not multiples of 32
    would give misaligned feature maps. Padding uses the mean pixel colour, which the
    model treats as empty. Example: a 500x375 image becomes 512x384, with the original in
    the top-left corner.

    Args:
        img: Image of shape (H, W, 3).
        m: The multiple to pad to.

    Returns:
        The padded image, or the original image if no padding was needed.
    """
    h, w = img.shape[:2]
    H, W = math.ceil(h / m) * m, math.ceil(w / m) * m  # round each side up to a multiple of m
    if (H, W) == (h, w):
        return img
    out = np.empty((H, W, 3), img.dtype)
    out[:] = PAD_VALUE  # fill with mean BGR colour
    out[:h, :w] = img  # original image in the top-left
    return out


def _nms(boxes, scores, thr):
    """Runs torchvision's NMS. Returns the kept indices, highest score first."""
    from torchvision.ops import nms

    return nms(boxes, scores, thr)


class Detector:
    """Runs RetinaFace on one image and returns final face detections.

    Example:
        det = Detector(model, cfg)
        faces = det(cv2.imread("group.jpg"))  # shape (N, 15)
        x1, y1, x2, y2, score = faces[0, :5]

    Settings such as conf_thresh, nms_thresh, top_k, keep_top_k and max_size come from
    cfg["eval"]. Any of them can be overridden with a keyword argument; None means "use
    the config value".

    Args:
        model: A RetinaFace model. It is switched to eval mode.
        cfg: The full config dict. Uses the "anchors" and "eval" sections.
        device: Device to run on. Defaults to the device the model is on.
        **overrides: Replacement values for cfg["eval"] keys.
    """

    def __init__(self, model: nn.Module, cfg: dict, device=None, **overrides):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.a = cfg["anchors"]
        self.e = {**cfg["eval"], **{k: v for k, v in overrides.items() if v is not None}}  # config + overrides
        self._priors: dict[tuple[int, int], torch.Tensor] = {}  # anchor cache, keyed by padded image size

    def priors(self, h: int, w: int) -> torch.Tensor:
        """Returns the anchors for an h x w input, building them only the first time.

        Args:
            h: Padded image height in pixels.
            w: Padded image width in pixels.

        Returns:
            Anchors of shape (P, 4) on the detector's device.
        """
        key = (h, w)
        if key not in self._priors:
            pr = generate_priors((h, w), self.a["min_sizes"], self.a["steps"], self.a["clip"])
            self._priors[key] = torch.from_numpy(pr).to(self.device)
        return self._priors[key]

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Detects faces in one image.

        Args:
            img: Image of shape (H, W, 3), uint8, in BGR channel order (as cv2.imread gives).

        Returns:
            A float32 array of shape (N, 15), one row per face, best score first. Each row
            is x1, y1, x2, y2, score, then 5 (x, y) landmark points. Coordinates are in
            pixels of the original image.
        """
        e = self.e
        scale = 1.0
        if e.get("max_size") and max(img.shape[:2]) > e["max_size"]:
            scale = e["max_size"] / max(img.shape[:2])  # shrink so the long side fits max_size
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        padded = pad_to_multiple(img, 32)
        H, W = padded.shape[:2]
        # HWC uint8 image -> (1, 3, H, W) float tensor on the device.
        x = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).to(self.device).float()[None]
        loc, conf, landm = self.model(x)
        pr = self.priors(H, W)
        var = self.a["variance"]
        scores = conf[0].float().softmax(-1)[:, 1]  # probability of "face" per anchor
        keep = scores > e["conf_thresh"]  # drop clearly-empty anchors early
        scores = scores[keep]
        boxes = decode_boxes(loc[0].float()[keep], pr[keep], var)
        lms = decode_landms(landm[0].float()[keep], pr[keep], var)
        if scores.numel() == 0:
            return np.zeros((0, 15), np.float32)  # no faces found
        order = scores.argsort(descending=True)[: e["top_k"]]  # best top_k before NMS
        boxes, scores, lms = boxes[order], scores[order], lms[order]
        wh = torch.tensor([W, H], device=self.device, dtype=boxes.dtype)
        boxes = (boxes.view(-1, 2, 2) * wh).view(-1, 4) / scale  # 0-1 -> pixels of the original image
        lms = (lms.view(-1, 5, 2) * wh).view(-1, 10) / scale
        k = _nms(boxes, scores, e["nms_thresh"])[: e["keep_top_k"]]  # remove duplicates, cap the count
        out = torch.cat([boxes[k], scores[k, None], lms[k]], dim=1)
        return out.cpu().numpy().astype(np.float32)


class ExportWrapper(nn.Module):
    """Wraps RetinaFace for ONNX export with a fixed input size and decoding built in.

    The exported model takes an image and returns boxes, scores and landmarks in pixels
    for every anchor. It does not run NMS; the runtime does that, for example TensorRT's
    EfficientNMS plugin or DeepStream's clustering. Example: with height=640 and width=640
    the outputs have P = 16,800 rows.

    Args:
        model: A RetinaFace model. It is switched to eval mode.
        cfg: The full config dict. Uses the "anchors" section.
        height: Fixed input height in pixels.
        width: Fixed input width in pixels.
    """

    def __init__(self, model: nn.Module, cfg: dict, height: int, width: int):
        super().__init__()
        self.model = model.eval()
        a = cfg["anchors"]
        pr = generate_priors((height, width), a["min_sizes"], a["steps"], a["clip"])
        self.register_buffer("priors", torch.from_numpy(pr)[None])  # stored in the graph, shape (1, P, 4)
        self.register_buffer("scale", torch.tensor([width, height] * 5, dtype=torch.float32))  # 0-1 -> pixels
        self.var = tuple(a["variance"])

    def forward(self, x):
        """Runs the model and decodes its outputs.

        Args:
            x: Batch of shape (B, 3, height, width), float, BGR, pixel values 0 to 255.

        Returns:
            A tuple (boxes, scores, landmarks) with shapes (B, P, 4), (B, P) and
            (B, P, 10). Boxes are xyxy and all coordinates are in pixels.
        """
        loc, conf, landm = self.model(x)
        boxes = decode_boxes(loc, self.priors, self.var) * self.scale[:4]
        lms = decode_landms(landm, self.priors, self.var) * self.scale
        scores = conf.softmax(-1)[..., 1]  # probability of "face"
        return boxes, scores, lms
