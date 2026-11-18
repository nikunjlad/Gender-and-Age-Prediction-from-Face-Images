"""PyTorch Dataset that feeds training images and their targets to the DataLoader.

Each item goes through: read the image, augment it (transforms.py), then assign anchor
targets (models/targets.py). All of this runs in the DataLoader worker processes, so the
GPU never waits for target assignment.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .annotations import parse_label_file, resolve_image_path, summarize
from face_detection.models.targets import TargetAssigner


class FaceDetectionDataset(Dataset):
    """Face detection dataset built from a label file.

    Example:
        ds = FaceDetectionDataset("data/widerface/train/label.txt",
                                  "data/widerface/train/images", TrainTransform(...), assigner)
        image, target = ds[0]  # image: (3, 640, 640) uint8; target["conf"]: (16800,)

    Args:
        label_file: Path to a label file (see annotations.py for the format).
        image_root: Folder that the image paths in the label file are relative to.
        transform: Callable (img, boxes, landms, has_landm) -> the same four, after
            augmentation. Boxes must come out normalised to 0 to 1.
        assigner: TargetAssigner that turns boxes into per-anchor targets.

    Raises:
        ValueError: If the label file contains no images.
    """

    def __init__(self, label_file, image_root, transform, assigner: TargetAssigner):
        self.samples = parse_label_file(label_file)
        self.root = image_root
        self.transform = transform
        self.assigner = assigner
        if not self.samples:
            raise ValueError(f"No samples parsed from {label_file}")

    def __len__(self):
        """Returns the number of images."""
        return len(self.samples)

    def describe(self) -> str:
        """Returns a one-line summary of the dataset (see annotations.summarize)."""
        return summarize(self.samples)

    def __getitem__(self, i):
        """Loads, augments and encodes image i.

        Returns:
            A tuple (image, target). image is a uint8 BGR tensor of shape (3, H, W).
            target is a dict of tensors: "loc", "conf", "landm" and "landm_mask".

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        s = self.samples[i]
        path = resolve_image_path(self.root, s.path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        # Copies so the augmentation never edits the cached labels.
        img, boxes, landms, has = self.transform(img, s.boxes.copy(), s.landmarks.copy(), s.has_landmarks.copy())
        t = self.assigner(boxes, landms, has)  # per-anchor targets
        image = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # uint8, BGR
        target = {k: torch.from_numpy(v) for k, v in t.as_dict().items()}
        return image, target


def worker_init_fn(worker_id: int) -> None:
    """Gives each DataLoader worker its own random seed.

    Without this, workers could share a seed and produce identical augmentations.
    Pass it to the DataLoader as worker_init_fn.

    Args:
        worker_id: The worker's index, supplied by the DataLoader.
    """
    info = torch.utils.data.get_worker_info()
    seed = (torch.initial_seed() + worker_id) % 2**32  # unique per worker and per epoch
    tf = getattr(info.dataset, "transform", None)
    if tf is not None and hasattr(tf, "reseed"):
        tf.reseed(seed)  # the transform's own random generator
    np.random.seed(seed % 2**32)
