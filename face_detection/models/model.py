"""The RetinaFace detector network.

Data flows through four parts:
    1. Backbone (body): extracts feature maps C3, C4, C5 at stride 8, 16, 32.
    2. FPN (feature pyramid network): mixes each map with the coarser one above it, so
       the fine maps also know about large-scale context.
    3. SSH (single stage headless module): widens each map's view with 3x3, 5x5 and 7x7
       receptive fields, which helps with faces of different sizes.
    4. Heads: tiny 1x1 convolutions that predict, per anchor, face/background scores,
       box offsets and landmark offsets.

Module and weight names match biubug6/Pytorch_Retinaface (fpn.output1, ssh1.conv3X3,
ClassHead.0.conv1x1, ...). So biubug6 checkpoints load directly, and the FPN, SSH and
heads can be reused with a new backbone whenever out_channel is the same.

Input: a float tensor of shape (B, 3, H, W) in BGR order with pixel values 0 to 255, as
cv2.imread gives. Pixel normalisation happens inside the model, so training, evaluation
and ONNX export always normalise the same way.

Output: raw scores (logits, before softmax) as loc (B, P, 4), conf (B, P, 2) and
landm (B, P, 10), where P is the number of anchors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import NORM_PRESETS, build_backbone, conv_bn, conv_bn1x1, conv_bn_no_relu


class SSH(nn.Module):
    """Single stage headless context module.

    Runs three branches in parallel and stacks their outputs along channels:
        3x3 branch: one 3x3 conv, half of the output channels.
        5x5 branch: two 3x3 convs in a row (same view as one 5x5), a quarter of the channels.
        7x7 branch: three 3x3 convs in a row (same view as one 7x7), a quarter of the channels.
    Example: with out_channel 64 the branches give 32 + 16 + 16 = 64 channels.

    Args:
        in_channel: Input channels.
        out_channel: Output channels. Must be divisible by 4.
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0.1 if out_channel <= 64 else 0.0  # biubug6 uses leaky ReLU only for the small model
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, leaky=leaky)  # shared first step of 5x5 and 7x7
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4)

    def forward(self, x):
        """Returns the three branches stacked along channels, after a ReLU."""
        c3 = self.conv3X3(x)
        c5_1 = self.conv5X5_1(x)
        c5 = self.conv5X5_2(c5_1)
        c7 = self.conv7x7_3(self.conv7X7_2(c5_1))
        return F.relu(torch.cat([c3, c5, c7], dim=1))


class FPN(nn.Module):
    """Feature pyramid network over the three backbone maps.

    Each map is first brought to out_channels with a 1x1 conv. Then, from coarse to fine,
    each map gets the upsampled coarser map added to it, followed by a 3x3 conv. Example:
    the 40x40 map is upsampled to 80x80 and added to the 80x80 map.

    Args:
        in_channels_list: Channels of C3, C4 and C5, for example [64, 128, 256].
        out_channels: Channels of every output map.
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, leaky=leaky)  # C3 -> out_channels
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, leaky=leaky)  # C4 -> out_channels
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, leaky=leaky)  # C5 -> out_channels
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, feats):
        """Returns the three merged maps, finest first, for the inputs [C3, C4, C5]."""
        o1 = self.output1(feats[0])
        o2 = self.output2(feats[1])
        o3 = self.output3(feats[2])
        o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[-2:], mode="nearest"))  # add upsampled C5 into C4
        o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[-2:], mode="nearest"))  # add upsampled C4 into C3
        return [o1, o2, o3]


class _Head(nn.Module):
    """Shared code for the three prediction heads.

    A 1x1 conv predicts dim numbers for each of num_anchors anchors at every grid cell.
    The output is then reshaped from (B, A*dim, H, W) to (B, H*W*A, dim), in the same
    row, column, anchor order used by generate_priors.

    Args:
        inchannels: Input channels.
        num_anchors: Anchors per grid cell.
        dim: Numbers predicted per anchor: 2 for class, 4 for box, 10 for landmarks.
    """

    def __init__(self, inchannels, num_anchors, dim):
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Returns predictions of shape (B, H*W*num_anchors, dim)."""
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()  # (B, H, W, A*dim)
        return out.view(out.shape[0], -1, self.dim)  # (B, H*W*A, dim)


class ClassHead(_Head):
    """Predicts 2 scores per anchor: background and face."""

    def __init__(self, inchannels, num_anchors=2):
        super().__init__(inchannels, num_anchors, 2)


class BboxHead(_Head):
    """Predicts 4 box offsets per anchor."""

    def __init__(self, inchannels, num_anchors=2):
        super().__init__(inchannels, num_anchors, 4)


class LandmarkHead(_Head):
    """Predicts 10 landmark offsets per anchor (5 points, x and y)."""

    def __init__(self, inchannels, num_anchors=2):
        super().__init__(inchannels, num_anchors, 10)


HEAD_PREFIXES = ("ClassHead.", "BboxHead.", "LandmarkHead.")  # weight name prefixes of the heads


def resolve_norm(spec, body) -> dict:
    """Turns the input_norm config value into concrete mean, std and channel order.

    Example: "auto" with a ResNet backbone gives the ImageNet values with "rgb": True.

    Args:
        spec: "auto" or None (use the backbone's default), a NORM_PRESETS name, or a
            dict with "mean", "std" and "rgb".
        body: The backbone, used for its default_norm.

    Returns:
        A dict with "mean" (3 floats), "std" (3 floats) and "rgb" (bool).

    Raises:
        ValueError: If spec is an unknown preset name.
    """
    if spec in (None, "auto"):
        spec = body.default_norm
    if isinstance(spec, str):
        if spec not in NORM_PRESETS:
            raise ValueError(f"Unknown input_norm {spec!r}; use auto, {list(NORM_PRESETS)} or a dict")
        spec = NORM_PRESETS[spec]
    return {"mean": [float(v) for v in spec["mean"]], "std": [float(v) for v in spec["std"]], "rgb": bool(spec["rgb"])}


class RetinaFace(nn.Module):
    """The full RetinaFace detector: backbone, FPN, three SSH modules and three heads.

    Example:
        model = RetinaFace(cfg)
        loc, conf, landm = model(images)  # images: (B, 3, 640, 640), BGR, 0-255
        # loc: (B, 16800, 4), conf: (B, 16800, 2), landm: (B, 16800, 10)

    Args:
        cfg: The full config dict. Uses the "model" and "anchors" sections.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        out_ch = int(mcfg["out_channel"])  # channels inside FPN and SSH
        anchors = [len(m) for m in cfg["anchors"]["min_sizes"]]  # anchors per cell on each level
        self.body = build_backbone(mcfg)
        self.fpn = FPN(self.body.out_channels, out_ch)
        self.ssh1 = SSH(out_ch, out_ch)  # stride 8 level
        self.ssh2 = SSH(out_ch, out_ch)  # stride 16 level
        self.ssh3 = SSH(out_ch, out_ch)  # stride 32 level
        self.ClassHead = nn.ModuleList([ClassHead(out_ch, a) for a in anchors])  # one head per level
        self.BboxHead = nn.ModuleList([BboxHead(out_ch, a) for a in anchors])
        self.LandmarkHead = nn.ModuleList([LandmarkHead(out_ch, a) for a in anchors])
        self.norm = resolve_norm(mcfg.get("input_norm", "auto"), self.body)
        # non-persistent: keeps state_dict identical to biubug6 checkpoints
        self.register_buffer("pixel_mean", torch.tensor(self.norm["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(self.norm["std"]).view(1, 3, 1, 1), persistent=False)
        self.steps = list(cfg["anchors"]["steps"])
        self._init_new_layers()

    def _init_new_layers(self):
        """Gives the non-backbone layers sensible starting weights.

        These values are overwritten when a checkpoint is loaded, so they only matter
        when training the FPN, SSH or heads from scratch.
        """
        for mod in [self.fpn, self.ssh1, self.ssh2, self.ssh3]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        # Prior-probability init for the face logit (focal-loss style) speeds up early training
        # when heads are freshly initialised; harmless when a checkpoint overwrites them.
        for heads in (self.ClassHead, self.BboxHead, self.LandmarkHead):
            for h in heads:
                nn.init.normal_(h.conv1x1.weight, std=0.01)  # small weights: near-zero initial predictions
                nn.init.zeros_(h.conv1x1.bias)
        bias = -math.log((1 - 0.01) / 0.01)  # logit that makes the starting face probability about 1%
        for h in self.ClassHead:
            h.conv1x1.bias.data.view(-1, 2)[:, 1] = bias  # column 1 is the face score

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Converts raw BGR 0-255 pixels into the backbone's expected input.

        Args:
            x: Images of shape (B, 3, H, W), BGR, 0 to 255.

        Returns:
            Normalised images of the same shape, flipped to RGB if the backbone needs it.
        """
        if self.norm["rgb"]:
            x = x.flip(1)  # BGR -> RGB
        return (x - self.pixel_mean) / self.pixel_std

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns the three SSH output maps, finest first, for images x (B, 3, H, W)."""
        p = self.fpn(self.body(self.normalize(x)))
        return [self.ssh1(p[0]), self.ssh2(p[1]), self.ssh3(p[2])]

    def forward(self, x: torch.Tensor):
        """Runs the detector.

        Args:
            x: Images of shape (B, 3, H, W), BGR, 0 to 255.

        Returns:
            A tuple (loc, conf, landm) with shapes (B, P, 4), (B, P, 2) and (B, P, 10).
            Each is the three levels joined in level order, matching generate_priors.
        """
        f = self.features(x)
        loc = torch.cat([h(t) for h, t in zip(self.BboxHead, f)], dim=1)
        conf = torch.cat([h(t) for h, t in zip(self.ClassHead, f)], dim=1)
        landm = torch.cat([h(t) for h, t in zip(self.LandmarkHead, f)], dim=1)
        return loc, conf, landm

    @torch.no_grad()
    def check_strides(self, image_size: int = 640) -> None:
        """Checks that the backbone really outputs stride 8, 16 and 32 feature maps.

        Runs a blank image through the backbone and compares the map sizes with the
        anchor steps. Example: a 640x640 input must give 80x80, 40x40 and 20x20 maps.

        Args:
            image_size: Side length of the square test image.

        Raises:
            RuntimeError: If the feature map sizes do not match the anchor steps.
        """
        was_training = self.training
        self.eval()  # BatchNorm must not update its statistics on the blank image
        dev = self.pixel_mean.device
        feats = self.body(self.normalize(torch.zeros(1, 3, image_size, image_size, device=dev)))
        self.train(was_training)  # restore the previous mode
        got = [tuple(t.shape[-2:]) for t in feats]
        want = [(math.ceil(image_size / s),) * 2 for s in self.steps]
        if got != want:
            raise RuntimeError(f"Backbone feature sizes {got} != expected {want} for steps {self.steps}")


def key_group(k: str) -> str:
    """Returns which part of the model a weight name belongs to.

    Used when transferring and reporting checkpoint weights. Example:
    "ssh2.conv3X3.0.weight" returns "ssh".

    Args:
        k: A weight name from the state dict.

    Returns:
        One of "body", "fpn", "ssh", "heads" or "other".
    """
    if k.startswith("body."):
        return "body"
    if k.startswith("fpn."):
        return "fpn"
    if k.startswith("ssh"):
        return "ssh"
    if k.startswith(HEAD_PREFIXES):
        return "heads"
    return "other"
