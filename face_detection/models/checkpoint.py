"""Checkpoint loading, weight transfer between architectures, and saving.

Transfer copies every weight whose name and shape match the new model and reports what
loaded. What loads depends on how much the architecture changed:
    Same backbone (plain fine-tuning): everything loads.
    New backbone, same out_channel: fpn.merge*, ssh* and the heads load. The backbone and
        fpn.output* (the 1x1 convs whose input width depends on the backbone) start fresh.
    Different out_channel: only tensors that happen to match in shape load, usually
        nothing in the FPN or SSH.

Example: starting a resnet34 model from biubug6's Resnet50_Final.pth keeps the trained
SSH modules and heads, and only the backbone and FPN input layers start from scratch.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .model import key_group

STRIP_PREFIXES = ("module.", "_orig_mod.")  # added by DataParallel / DDP and by torch.compile


def unwrap(model: nn.Module) -> nn.Module:
    """Returns the plain model inside DataParallel, DDP or torch.compile wrappers."""
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", None) or getattr(model, "_orig_mod")
    return model


def _strip_prefixes(sd: dict) -> dict:
    """Removes wrapper prefixes from weight names, even when stacked.

    Example: "_orig_mod.module.fpn.output1.0.weight" becomes "fpn.output1.0.weight".

    Args:
        sd: A state dict (weight name -> tensor).

    Returns:
        A new state dict with the prefixes removed.
    """
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:  # repeat until no prefix is left
            changed = False
            for p in STRIP_PREFIXES:
                if k.startswith(p):
                    k, changed = k[len(p) :], True
        out[k] = v
    return out


def read_checkpoint(path: str | Path, prefer_ema: bool = True, trust: bool = False):
    """Reads a checkpoint file and returns its weights plus any extra information.

    Handles raw state dicts (like biubug6's files) and dicts that wrap the weights under
    "model", "state_dict", "ema" or "model_state_dict" (like this repo's files).

    Files are loaded with weights_only=True, which refuses to run code stored in the file.
    Some older files need weights_only=False; that is only allowed with trust=True,
    because a malicious file could then run arbitrary code.

    Args:
        path: Path to the checkpoint.
        prefer_ema: If the file has EMA weights, return those instead of the live weights.
        trust: Allow the unsafe weights_only=False fallback.

    Returns:
        A tuple (state_dict, meta). meta holds the file's other entries (epoch, cfg,
        metrics, ...) plus "source", which says where the weights came from: "ema",
        "model", "state_dict", "model_state_dict" or "raw".

    Raises:
        RuntimeError: If safe loading fails and trust is False.
        ValueError: If no weights can be found in the file.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not trust:
            raise RuntimeError(
                f"Could not load {path} with weights_only=True ({type(e).__name__}: {e}). If you trust this file, "
                "set init.trust_checkpoint=true (or pass --trust)."
            ) from e
        warnings.warn(f"Loading {path} with weights_only=False (arbitrary code execution risk)")
        obj = torch.load(path, map_location="cpu", weights_only=False)

    meta: dict = {}
    if isinstance(obj, dict) and any(k in obj for k in ("model", "state_dict", "ema", "model_state_dict")):
        # Wrapped format: weights live under one key, everything else is metadata.
        meta = {k: v for k, v in obj.items() if k not in ("model", "state_dict", "ema", "optimizer", "scaler")}
        if prefer_ema and isinstance(obj.get("ema"), dict) and obj["ema"]:
            sd, meta["source"] = obj["ema"], "ema"
        else:
            for k in ("model", "state_dict", "model_state_dict"):
                if isinstance(obj.get(k), dict):
                    sd, meta["source"] = obj[k], k
                    break
            else:
                raise ValueError(f"No state dict found in {path}")
    elif isinstance(obj, dict):
        sd, meta["source"] = obj, "raw"  # the file is the state dict itself (biubug6 style)
    else:
        raise ValueError(f"Unsupported checkpoint type {type(obj)} in {path}")
    sd = {k: v for k, v in _strip_prefixes(sd).items() if torch.is_tensor(v)}  # keep tensors only
    return sd, meta


@dataclass
class TransferReport:
    """What happened during transfer_weights.

    Attributes:
        loaded: Names of the weights that were copied.
        mismatched: (name, checkpoint shape, model shape) for weights that were not
            copied because of a shape difference or an incomplete unit.
        unexpected: Names in the checkpoint that the model does not have.
        filtered: Names skipped by the include/exclude filters.
        missing: Model weights that were not loaded and keep their fresh values.
        group_stats: Per model part ("body", "fpn", ...): tensors and parameter counts,
            loaded and total.
    """

    loaded: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    group_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def fraction(self, group: str) -> float:
        """Returns the share of a group's parameters that loaded, from 0 to 1.

        Example: fraction("heads") is 1.0 when every head weight was copied.
        """
        s = self.group_stats.get(group, {})
        return s.get("loaded_numel", 0) / max(s.get("total_numel", 0), 1)

    def format(self, verbose: bool = False) -> str:
        """Returns a readable multi-line summary table.

        Args:
            verbose: If True, list every mismatched and unexpected name instead of the
                first few.

        Returns:
            The report as a string, ready to print.
        """
        lines = ["Weight transfer report", f"{'group':<8}{'tensors':>16}{'params':>22}"]
        for g in ("body", "fpn", "ssh", "heads", "other"):
            s = self.group_stats.get(g)
            if not s or s["total"] == 0:
                continue  # skip parts the model does not have
            lines.append(
                f"{g:<8}{s['loaded']:>7}/{s['total']:<8}"
                f"{s['loaded_numel'] / 1e6:>9.2f}M/{s['total_numel'] / 1e6:.2f}M ({100 * self.fraction(g):5.1f}%)"
            )
        if self.mismatched:
            lines.append(f"shape mismatch ({len(self.mismatched)}), re-initialised:")
            for k, src, dst in self.mismatched[: None if verbose else 12]:
                lines.append(f"  {k}: ckpt {tuple(src)} -> model {tuple(dst)}")
            if not verbose and len(self.mismatched) > 12:
                lines.append(f"  ... {len(self.mismatched) - 12} more")
        if self.unexpected:
            show = self.unexpected if verbose else self.unexpected[:8]
            lines.append(f"in checkpoint but not in model ({len(self.unexpected)}): {show}")
        if self.filtered:
            lines.append(f"skipped by include/exclude: {len(self.filtered)} tensors")
        return "\n".join(lines)


def _selected(k: str, include, exclude) -> bool:
    """Returns True if a weight name passes the include/exclude prefix filters.

    Example: with exclude ["body."], every backbone weight is skipped.
    """
    if include and not any(k.startswith(p) for p in include):
        return False
    return not any(k.startswith(p) for p in exclude or [])


def _unit(k: str) -> str:
    """Returns the module a weight belongs to, by dropping the last two name parts.

    Examples:
        "fpn.output1.0.weight" -> "fpn.output1" (a conv + BatchNorm block)
        "body.layer2.0.conv1.weight" -> "body.layer2.0" (a residual block)
        "ClassHead.0.conv1x1.bias" -> "ClassHead.0"
    """
    return k.rsplit(".", 2)[0]


def transfer_weights(
    model: nn.Module,
    src: dict[str, torch.Tensor],
    include=None,
    exclude=None,
    rename=None,
    require=None,
    min_fraction: float = 0.9,
    atomic: bool = True,
) -> TransferReport:
    """Copies every checkpoint weight whose name and shape match the model.

    With atomic=True, a module unit (for example a conv + BatchNorm pair, or a residual
    block) loads all or nothing. Example: if the conv in fpn.output1 has the wrong
    shape, its BatchNorm is not loaded either, even though its shape matches. A BatchNorm
    calibrated for a different conv is worse than a fresh one.

    Args:
        model: The model to load into.
        src: Checkpoint state dict.
        include: If given, only names starting with one of these prefixes are used.
        exclude: Names starting with one of these prefixes are skipped.
        rename: List of (regex, replacement) pairs applied to checkpoint names first.
        require: Model parts ("body", "fpn", "ssh", "heads") that must load at least
            min_fraction of their parameters.
        min_fraction: Threshold for require, from 0 to 1.
        atomic: Load module units all or nothing (see above).

    Returns:
        A TransferReport describing what loaded.

    Raises:
        RuntimeError: If a required part loaded less than min_fraction.
    """
    model = unwrap(model)
    for pat, repl in rename or []:
        src = {re.sub(pat, repl, k): v for k, v in src.items()}
    tgt = model.state_dict()
    rep = TransferReport()
    loaded: dict[str, torch.Tensor] = {}
    for k, v in src.items():  # sort each checkpoint weight into one bucket
        if not _selected(k, include, exclude):
            rep.filtered.append(k)
        elif k not in tgt:
            rep.unexpected.append(k)
        elif tgt[k].shape != v.shape:
            rep.mismatched.append((k, tuple(v.shape), tuple(tgt[k].shape)))
        else:
            loaded[k] = v.to(dtype=tgt[k].dtype)
    if atomic:
        # a unit is broken if any of its tensors (other than BN step counters) did not load
        broken = {_unit(k) for k in tgt if k not in loaded and not k.endswith("num_batches_tracked")}
        for k in [k for k in loaded if _unit(k) in broken]:
            t = loaded.pop(k)  # drop the rest of a broken unit
            rep.mismatched.append((k + "  [unit incomplete]", tuple(t.shape), tuple(tgt[k].shape)))
    model.load_state_dict(loaded, strict=False)
    rep.loaded = sorted(loaded)
    rep.missing = [k for k in tgt if k not in loaded]

    # Per-part counts for the report table.
    stats = defaultdict(lambda: {"loaded": 0, "total": 0, "loaded_numel": 0, "total_numel": 0})
    for k, t in tgt.items():
        if k.endswith("num_batches_tracked"):
            continue  # BN step counter, not a weight
        g = key_group(k)
        stats[g]["total"] += 1
        stats[g]["total_numel"] += t.numel()
        if k in loaded:
            stats[g]["loaded"] += 1
            stats[g]["loaded_numel"] += t.numel()
    rep.group_stats = dict(stats)

    bad = [g for g in (require or []) if rep.fraction(g) < min_fraction]
    if bad:
        raise RuntimeError(
            rep.format() + f"\n\nRequired groups {bad} transferred < {min_fraction:.0%}. Wrong checkpoint, backbone or "
            "out_channel? Fix the config or relax init.require."
        )
    return rep


def init_from_config(model: nn.Module, cfg: dict) -> TransferReport | None:
    """Loads starting weights for training, as set in the config's "init" section.

    Prints the transfer report. Example: init.checkpoint "weights/Resnet50_Final.pth"
    with init.exclude ["body."] loads everything except the backbone.

    Args:
        model: The freshly built model.
        cfg: The full config dict.

    Returns:
        The TransferReport, or None if init.checkpoint is not set.
    """
    icfg = cfg["init"]
    if not icfg.get("checkpoint"):
        return None
    sd, meta = read_checkpoint(icfg["checkpoint"], icfg.get("prefer_ema", True), icfg.get("trust_checkpoint", False))
    rep = transfer_weights(
        model,
        sd,
        icfg.get("include"),
        icfg.get("exclude"),
        icfg.get("rename"),
        icfg.get("require"),
        icfg.get("min_fraction", 0.9),
    )
    print(f"[init] {icfg['checkpoint']} (source: {meta.get('source')})")
    print(rep.format())
    _warn_norm_mismatch(unwrap(model), meta, rep)
    return rep


def _warn_norm_mismatch(model, meta: dict, rep: TransferReport) -> None:
    """Warns if the backbone was trained with a different pixel normalisation.

    A backbone trained on one normalisation (for example BGR minus mean) and run with
    another (for example ImageNet RGB) loses accuracy.

    Args:
        model: The model the weights were loaded into.
        meta: Checkpoint metadata from read_checkpoint.
        rep: The transfer report, used to skip the check when little of the backbone loaded.
    """
    if rep.fraction("body") < 0.5:
        return  # backbone mostly fresh, so its old normalisation does not matter
    src_cfg = meta.get("cfg")
    if src_cfg:  # checkpoint from this repo records its normalisation
        src_norm = meta.get("norm")
        if src_norm and src_norm != model.norm:
            warnings.warn(f"Checkpoint was trained with input norm {src_norm} but model uses {model.norm}")
        return
    # foreign checkpoint (e.g. biubug6) - those were trained with BGR mean subtraction
    if model.norm["rgb"] or model.norm["std"] != [1.0, 1.0, 1.0]:
        warnings.warn(
            "Transferred backbone weights come from a checkpoint without recorded normalisation. biubug6 "
            "checkpoints expect model.input_norm=retinaface_bgr; you are using "
            f"{model.norm}. Set it explicitly if this is a biubug6 checkpoint."
        )


def save_checkpoint(path, model, cfg, epoch, ema=None, optimizer=None, scaler=None, metrics=None, best=None):
    """Saves a training checkpoint safely.

    The file is first written to path + ".tmp" and then renamed, so a crash mid-save
    never leaves a half-written checkpoint behind.

    Args:
        path: Destination file, for example runs/<run>/last.pt.
        model: The model being trained.
        cfg: The full config, stored so the checkpoint can rebuild its own model.
        epoch: The epoch just finished.
        ema: The ModelEMA, if used.
        optimizer: The optimizer, so training can resume.
        scaler: The AMP gradient scaler, so training can resume.
        metrics: Dict of metrics at this epoch.
        best: Best metric value so far.
    """
    m = unwrap(model)
    obj = {
        "model": m.state_dict(),
        "ema": unwrap(ema.module).state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "cfg": cfg,
        "norm": m.norm,  # checked by _warn_norm_mismatch when reloaded
        "metrics": metrics or {},
        "best": best,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # atomic rename


def load_for_inference(path, cfg=None, device="cpu", prefer_ema=True, trust=False):
    """Builds a ready-to-run model from a checkpoint.

    Checkpoints from this repo carry their own config. Foreign ones (like biubug6's) need
    a matching cfg. Every model part must load fully, otherwise this fails. Example:
    load_for_inference("weights/Resnet50_Final.pth", load_config("configs/resnet50_finetune.yaml")).

    Args:
        path: Checkpoint file.
        cfg: Config matching the checkpoint's architecture. Not needed for this repo's files.
        device: Device to put the model on.
        prefer_ema: Use EMA weights when the file has them.
        trust: Allow the unsafe weights_only=False fallback (see read_checkpoint).

    Returns:
        A tuple (model, cfg). The model is in eval mode on device.

    Raises:
        ValueError: If cfg is None and the checkpoint has no embedded config.
        RuntimeError: If any model part fails to load completely.
    """
    from face_detection.config import DEFAULTS, deep_merge
    from .model import RetinaFace

    sd, meta = read_checkpoint(path, prefer_ema, trust)
    if cfg is None:
        if "cfg" not in meta:
            raise ValueError(f"{path} has no embedded config; pass --config matching its architecture")
        cfg = deep_merge(DEFAULTS, meta["cfg"])
    # The checkpoint provides every weight, so skip any backbone download or file load.
    cfg = deep_merge(cfg, {"model": {"backbone_pretrained": False, "backbone_weights": None}})
    model = RetinaFace(cfg)
    rep = transfer_weights(model, sd, require=["body", "fpn", "ssh", "heads"], min_fraction=1.0)
    _warn_norm_mismatch(model, meta, rep)
    return model.to(device).eval(), cfg
