"""Transfer-learning tools: staged freezing, optimizer groups, learning-rate schedule and EMA.

Fine-tuning usually runs in two stages:
    Stage 1: the backbone is frozen (its weights do not change) and only the FPN, SSH and
        heads learn. This protects the pretrained backbone from the large, noisy updates
        that new random heads produce at the start.
    Stage 2: the backbone is unfrozen and trains with a smaller learning rate than the
        rest, ramping up from 0 so the change is gentle.

Example: with freeze_backbone_epochs 5 and backbone_lr_mult 0.1, epochs 0 to 4 train only
the neck and heads, and from epoch 5 the backbone trains at a tenth of the main rate.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .checkpoint import unwrap


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Freezes (flag False) or unfreezes (flag True) every weight in a module."""
    for p in module.parameters():
        p.requires_grad_(flag)


class FreezeController:
    """Decides which backbone layers are frozen at each epoch.

    Stage 1 (epoch < freeze_backbone_epochs): the whole backbone is frozen.
    Stage 2: the backbone trains, except its first freeze_stages stages, which stay
    frozen for the whole run.

    Frozen BatchNorm layers are also put in eval mode, so their running mean and variance
    stop changing as well. Example: with freeze_stages 1 on a ResNet, the stem
    (conv1 + bn1) never trains.

    Args:
        model: The RetinaFace model (wrapped models are unwrapped).
        freeze_backbone_epochs: Number of epochs with the whole backbone frozen.
        freeze_stages: Number of early backbone stages to keep frozen after that.
        freeze_bn: If True, keep every backbone BatchNorm in eval mode. This helps with
            small batches, where batch statistics are noisy.
    """

    def __init__(self, model: nn.Module, freeze_backbone_epochs: int, freeze_stages: int, freeze_bn: bool):
        self.model = unwrap(model)
        self.freeze_epochs = int(freeze_backbone_epochs)
        self.freeze_stages = int(freeze_stages)
        self.freeze_bn = bool(freeze_bn)
        self.state = None  # last state description, to report only changes

    def backbone_frozen(self, epoch: int) -> bool:
        """Returns True if the whole backbone is frozen at this epoch."""
        return epoch < self.freeze_epochs

    def apply(self, epoch: int) -> str:
        """Sets requires_grad on the backbone for this epoch.

        Args:
            epoch: The current epoch, starting at 0.

        Returns:
            A short description such as "frozen" or "trainable, kept frozen: ['stage1']"
            when the state changed since the last call, or "" if it did not.
        """
        body = self.model.body
        frozen = self.backbone_frozen(epoch)
        set_requires_grad(body, not frozen)  # stage 1: all off; stage 2: all on
        names = []
        if not frozen and self.freeze_stages > 0:
            named = {id(m): n for n, m in body.named_modules()}  # module -> its name, for the message
            for st in body.stages()[: self.freeze_stages]:
                set_requires_grad(st, False)  # keep the early stages frozen
                names.append(named.get(id(st), type(st).__name__))
        state = "frozen" if frozen else ("trainable" + (f", kept frozen: {names}" if names else ""))
        changed = state != self.state
        self.state = state
        return state if changed else ""

    def set_train_mode(self) -> None:
        """Puts the BatchNorm layers of frozen parts back in eval mode.

        Call this right after model.train(), which switches every layer to train mode.
        """
        body = self.model.body
        for m in body.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params_frozen = all(not p.requires_grad for p in m.parameters())
                if self.freeze_bn or params_frozen:
                    m.eval()  # stop updating running mean / variance


def build_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    """Builds the optimizer with separate parameter groups for backbone and head.

    Parameters are split two ways: backbone vs head (everything else), and weights vs
    biases/norm parameters. Biases and norm parameters get no weight decay. Each group is
    tagged with "part" so LRSchedule can give the backbone its own learning rate.

    Args:
        model: The RetinaFace model.
        tcfg: The "train" section of the config. Reads optimizer, lr, momentum and
            weight_decay.

    Returns:
        SGD (with Nesterov momentum if momentum > 0) or AdamW.
    """
    model = unwrap(model)
    groups = {  # (part, gets weight decay) -> parameters
        ("backbone", True): [],
        ("backbone", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for name, p in model.named_parameters():
        part = "backbone" if name.startswith("body.") else "head"
        decay = p.ndim > 1  # no weight decay on biases / norm params
        groups[(part, decay)].append(p)
    pg = []
    for (part, decay), params in groups.items():
        if not params:
            continue  # skip empty groups
        pg.append(
            {
                "params": params,
                "weight_decay": tcfg["weight_decay"] if decay else 0.0,
                "part": part,  # read by LRSchedule.step
                "lr": tcfg["lr"],
            }
        )
    if tcfg["optimizer"] == "sgd":
        m = tcfg["momentum"]
        return torch.optim.SGD(pg, lr=tcfg["lr"], momentum=m, nesterov=m > 0)
    return torch.optim.AdamW(pg, lr=tcfg["lr"], betas=(0.9, 0.999))


class LRSchedule:
    """Sets the learning rate at every training step.

    The main rate is lr * base_factor, where base_factor rises linearly during warmup and
    then follows a cosine curve or step decay. The backbone rate is further multiplied by
    backbone_factor: 0 while the backbone is frozen, then ramping up to backbone_lr_mult.

    Example: lr 1e-3, warmup_epochs 1, backbone_lr_mult 0.1. Halfway through epoch 0 the
    head rate is 5e-4. After unfreezing, the backbone rate settles at a tenth of the
    head rate.

    Args:
        tcfg: The "train" section of the config.
        iters_per_epoch: Training steps per epoch, used to compute fractional epochs.
    """

    def __init__(self, tcfg: dict, iters_per_epoch: int):
        self.t = tcfg
        self.ipe = max(iters_per_epoch, 1)

    def base_factor(self, ep: float) -> float:
        """Returns the main learning-rate multiplier at fractional epoch ep."""
        t = self.t
        if ep < t["warmup_epochs"]:
            return max(ep / t["warmup_epochs"], 1e-3)  # linear warmup, never exactly 0
        if t["scheduler"] == "cosine":
            span = max(t["epochs"] - t["warmup_epochs"], 1e-6)
            prog = min((ep - t["warmup_epochs"]) / span, 1.0)  # 0 -> 1 over the post-warmup epochs
            return t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog))
        return t["gamma"] ** sum(ep >= m for m in t["milestones"])  # multistep: multiply by gamma at each milestone

    def backbone_factor(self, ep: float) -> float:
        """Returns the extra backbone multiplier at fractional epoch ep."""
        t = self.t
        start = t["freeze_backbone_epochs"]
        if ep < start:
            return 0.0  # backbone frozen
        ramp = t["unfreeze_warmup_epochs"]
        r = 1.0 if ramp <= 0 else min((ep - start) / ramp, 1.0)  # 0 -> 1 after unfreezing
        return t["backbone_lr_mult"] * max(r, 1e-3)

    def step(self, optimizer, epoch: int, it: int) -> float:
        """Updates every parameter group's learning rate for this step.

        Args:
            optimizer: Optimizer from build_optimizer.
            epoch: The current epoch, starting at 0.
            it: The step number within the epoch.

        Returns:
            The current head (main) learning rate, for logging.
        """
        ep = epoch + it / self.ipe  # fractional epoch, e.g. 2.5 halfway through epoch 2
        f = self.base_factor(ep)
        bf = self.backbone_factor(ep)
        for g in optimizer.param_groups:
            g["lr"] = self.t["lr"] * f * (bf if g.get("part") == "backbone" else 1.0)
        return self.t["lr"] * f


class ModelEMA:
    """Keeps an exponential moving average (EMA) of the model's weights.

    After each step the EMA copy moves a little towards the live weights:
    ema = d * ema + (1 - d) * live. The averaged weights are smoother and usually
    evaluate better than the live ones. d starts near 0 and ramps up to decay, so the EMA
    follows quickly at first (the YOLOv5 approach). Example: with tau 2000, d is about
    0.63 * decay after 2,000 updates.

    Args:
        model: The model to copy.
        decay: Final averaging factor d.
        tau: How many updates the ramp-up of d takes, roughly.
        updates: Number of updates already done, when resuming training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: float = 2000.0, updates: int = 0):
        self.module = copy.deepcopy(unwrap(model)).eval()  # separate copy, always in eval mode
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, updates

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Moves the EMA weights one step towards the live model's weights."""
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))  # ramped decay
        msd = unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)  # ema = d * ema + (1 - d) * live
            else:
                v.copy_(msd[k])  # integer buffers (e.g. BN counters) are copied as-is
