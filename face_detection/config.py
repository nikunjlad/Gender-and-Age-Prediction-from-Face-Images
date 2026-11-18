"""Config loading: defaults, YAML files and command-line overrides.

A run's config is built in three layers, each overriding the one before:
    1. DEFAULTS below.
    2. A YAML file such as configs/resnet34_transfer.yaml.
    3. Dotted command-line overrides such as train.lr=5e-4.

Example: python train.py --config configs/resnet34_transfer.yaml train.epochs=30 runs
with every default, then the YAML values, then 30 epochs.

Unknown keys raise an error, so a typo like train.lrr=5e-4 fails immediately instead of
being silently ignored.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "retinaface",  # run name; also the MLflow experiment name
    "model": {
        # mobilenet0.25 | resnet18/34/50/101/152 | resnext50_32x4d | wide_resnet50_2 | timm:<model_name>
        "backbone": "mobilenet0.25",
        # ImageNet weights for torchvision / timm bodies (ignored for mobilenet0.25)
        "backbone_pretrained": True,
        # Optional backbone-only checkpoint, e.g. biubug6's mobilenetV1X0.25_pretrain.tar
        "backbone_weights": None,
        "out_channel": 64,  # channels in FPN and SSH (biubug6: 64 for mobilenet, 256 for resnet50)
        # auto | retinaface_bgr | imagenet_rgb | {mean: [...], std: [...], rgb: bool} (0-255 scale)
        "input_norm": "auto",
        "timm_kwargs": {},  # extra arguments for timm.create_model
    },
    "anchors": {
        "min_sizes": [[16, 32], [64, 128], [256, 512]],  # anchor sizes in pixels, per pyramid level
        "steps": [8, 16, 32],  # stride of each pyramid level
        "variance": [0.1, 0.2],  # box encoding scale factors
        "clip": False,  # clamp anchors to the image
    },
    "init": {
        # Full RetinaFace checkpoint to transfer from (biubug6 .pth, or one produced by this repo)
        "checkpoint": None,
        # Only keys starting with one of these prefixes are considered (None = all)
        "include": None,
        # Keys starting with any of these prefixes are skipped
        "exclude": [],
        # Regex renames applied to source keys before matching: [[pattern, replacement], ...]
        "rename": [],
        # Groups (body | fpn | ssh | heads) that MUST be transferred, else training aborts
        "require": [],
        "min_fraction": 0.9,  # share of a required group that must load
        "prefer_ema": True,  # use EMA weights when the checkpoint has them
        # torch.load(weights_only=False) — only for checkpoints you trust
        "trust_checkpoint": False,
    },
    "data": {
        "train_root": None,  # training image folder
        "train_labels": None,  # training label file
        "val_root": None,  # validation image folder
        "val_labels": None,  # validation label file
        "image_size": 640,  # training images are resized to image_size x image_size
        "workers": 4,  # DataLoader worker processes
        "crop_scales": [0.3, 0.45, 0.6, 0.8, 1.0],  # random crop sizes, as fractions of the short side
        "min_face_px": 1.0,  # drop boxes smaller than this after resize to image_size
        "color_jitter": True,  # random brightness / contrast / saturation / hue
        "hflip": True,  # random horizontal flip
    },
    "loss": {
        "pos_iou": 0.35,  # anchor IoU needed to count as a face anchor
        "min_gt_best_iou": 0.2,  # faces whose best anchor IoU is below this are ignored
        "neg_pos_ratio": 7,  # hard background anchors kept per face anchor
        "min_negatives_per_image": 0,  # >0 lets background-only images contribute
        "loc_weight": 2.0,  # weight of the box loss
        "landm_weight": 1.0,  # weight of the landmark loss
    },
    "train": {
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "sgd",  # sgd | adamw
        "lr": 1e-3,  # main learning rate (neck and heads)
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 5e-4,
        "warmup_epochs": 1.0,  # linear learning-rate warmup
        "scheduler": "cosine",  # cosine | multistep
        "min_lr_ratio": 0.01,  # cosine ends at lr * min_lr_ratio
        "milestones": [],  # multistep: epochs at which lr is multiplied by gamma
        "gamma": 0.1,
        # --- transfer learning ---
        "freeze_backbone_epochs": 5,  # stage 1: train neck + heads only
        "freeze_stages": 0,  # backbone stages kept frozen for the whole run
        "backbone_lr_mult": 0.1,  # backbone lr relative to neck/heads after unfreeze
        "unfreeze_warmup_epochs": 1.0,  # ramp backbone lr from 0 after unfreezing
        "freeze_backbone_bn": False,  # keep backbone BN stats frozen (small batches)
        # ---
        "amp": True,  # mixed precision: faster, less GPU memory
        "amp_dtype": "float16",  # float16 | bfloat16
        "grad_clip": 10.0,  # maximum gradient norm
        "ema": True,  # keep an EMA copy of the weights
        "ema_decay": 0.9998,
        "val_every": 1,  # validation loss every N epochs
        "ap_every": 5,  # 0 disables AP evaluation during training
        "ap_max_images": 500,  # validation images used for AP
        "ap_max_size": 1024,  # shrink AP images so the long side fits
        "save_every": 10,  # extra numbered checkpoint every N epochs
        "output_dir": "runs",  # run folders go here
        "seed": 42,
        "log_every": 50,  # print progress every N steps
        "mlflow": False,  # log to MLflow
        "device": "cuda",
    },
    "eval": {
        "conf_thresh": 0.02,  # drop detections below this score before NMS
        "nms_thresh": 0.4,  # NMS IoU threshold
        "top_k": 5000,  # detections kept before NMS
        "keep_top_k": 750,  # detections kept after NMS
        "iou_thresh": 0.5,  # IoU needed to count a face as found
        "max_size": None,  # resize so the long side <= max_size (None = native resolution)
        "min_gt_px": 0,  # GT smaller than this is ignored in AP
    },
}


_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")  # scientific notation, e.g. 5e-4


def _coerce(v):
    """Turns numbers written in scientific notation into floats, recursively.

    PyYAML follows YAML 1.1, which reads "5e-4" as a string. Example: "5e-4" -> 0.0005,
    while "abc" and 5 are returned unchanged.
    """
    if isinstance(v, str) and _NUM.match(v.strip()):
        return float(v)
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    return v


def deep_merge(base: dict, override: dict) -> dict:
    """Merges override into a copy of base, recursing into nested dicts.

    Example: deep_merge({"train": {"lr": 1, "epochs": 5}}, {"train": {"lr": 2}}) gives
    {"train": {"lr": 2, "epochs": 5}}. Neither input is modified.

    Args:
        base: The starting dict.
        override: Values that replace or extend base.

    Returns:
        A new merged dict.
    """
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # merge nested sections key by key
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Applies dotted key=value overrides, such as train.lr=5e-4 or init.exclude=[body.].

    Values are parsed as YAML, so lists, numbers and booleans work.

    Args:
        cfg: The config to start from (not modified).
        overrides: Strings of the form dotted.key=value.

    Returns:
        A new config with the overrides applied.

    Raises:
        ValueError: If an override has no "=".
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = _coerce(yaml.safe_load(raw))  # "[body.]" -> ["body."], "5e-4" -> 0.0005
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:  # walk down to the parent of the final key
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg


def _check_unknown(cfg: dict, ref: dict, path: str = "") -> list[str]:
    """Returns dotted names of keys in cfg that do not exist in ref.

    Free-form sections (model.timm_kwargs and model.input_norm) are not checked.
    Example: {"train": {"lrr": 1}} against DEFAULTS returns ["train.lrr"].
    """
    unknown = []
    for k, v in cfg.items():
        if k not in ref:
            unknown.append(path + k)
        elif isinstance(v, dict) and isinstance(ref[k], dict) and k not in ("timm_kwargs",):
            if k == "input_norm":
                continue
            unknown += _check_unknown(v, ref[k], path + k + ".")
    return unknown


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    """Builds a full config from DEFAULTS, an optional YAML file and overrides.

    Example: load_config("configs/resnet34_transfer.yaml", ["train.lr=5e-4"]).

    Args:
        path: YAML config file, or None for defaults only.
        overrides: Dotted key=value strings applied last.

    Returns:
        The merged and validated config dict.

    Raises:
        KeyError: If any key is not in DEFAULTS (usually a typo).
        ValueError: If validate finds an invalid value.
    """
    user = {}
    if path is not None:
        with open(path) as f:
            user = _coerce(yaml.safe_load(f) or {})  # empty file -> {}
    cfg = deep_merge(DEFAULTS, user)
    cfg = apply_overrides(cfg, overrides)
    unknown = _check_unknown(cfg, DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys (typo?): {unknown}")
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """Checks config values that would otherwise fail later and less clearly.

    Args:
        cfg: A full config dict.

    Raises:
        ValueError: If a value is invalid, for example out_channel not divisible by 4.
    """
    a = cfg["anchors"]
    if len(a["min_sizes"]) != 3 or len(a["steps"]) != 3:
        raise ValueError("RetinaFace uses exactly 3 pyramid levels: min_sizes and steps need 3 entries")
    if cfg["model"]["out_channel"] % 4 != 0:
        raise ValueError("model.out_channel must be divisible by 4 (SSH splits it into 1/2 + 1/4 + 1/4)")
    # Any size works for backbones with ceil-mode downsampling (biubug6 trains R50 at 840);
    # RetinaFace.check_strides() verifies the real feature-map sizes at the start of training.
    if cfg["data"]["image_size"] % 8 != 0:
        raise ValueError("data.image_size must be a multiple of 8")
    t = cfg["train"]
    if t["scheduler"] not in ("cosine", "multistep"):
        raise ValueError("train.scheduler must be cosine or multistep")
    if t["optimizer"] not in ("sgd", "adamw"):
        raise ValueError("train.optimizer must be sgd or adamw")


def save_config(cfg: dict, path: str | Path) -> None:
    """Writes a config to a YAML file, keeping the key order (used to record each run's config)."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
