"""Backbone networks for RetinaFace.

A backbone is the image feature extractor at the bottom of the detector, usually a
classification network with its final classifier removed. RetinaFace needs three of its
feature maps, at stride 8, 16 and 32. Stride 8 means each feature cell covers 8x8 image
pixels. These maps are called C3, C4 and C5.

Three backbone families are supported:
    mobilenet0.25: a tiny MobileNetV1. Its weights are named body.stage1.*, body.stage2.*
        and body.stage3.*.
    torchvision ResNets, for example resnet50: weights named body.conv1.*, body.bn1.*
        and body.layer1.* to body.layer4.*.
    timm:<name>, any timm model with stride 8, 16 and 32 features: weights named
        body.model.*.

The weight names match biubug6/Pytorch_Retinaface exactly, so its released checkpoints
(mobilenet0.25_Final.pth, Resnet50_Final.pth) load without renaming anything.

Every backbone class provides:
    out_channels: The channel count of C3, C4 and C5, for example [64, 128, 256].
    default_norm: How input pixels should be normalised (see NORM_PRESETS).
    stages(): Its layers in order from input to output, used to freeze early layers.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # per-channel RGB mean of ImageNet, 0-1 scale
IMAGENET_STD = (0.229, 0.224, 0.225)  # per-channel RGB standard deviation of ImageNet, 0-1 scale

# Input normalisation presets, on the 0-255 pixel scale. The model computes
# (pixel - mean) / std. "rgb": True means the model first flips BGR input to RGB.
NORM_PRESETS = {
    "retinaface_bgr": {"mean": [104.0, 117.0, 123.0], "std": [1.0, 1.0, 1.0], "rgb": False},  # biubug6 checkpoints
    "imagenet_rgb": {"mean": [m * 255 for m in IMAGENET_MEAN], "std": [s * 255 for s in IMAGENET_STD], "rgb": True},
}


# ----------------------------------------------------------------------------- shared blocks (biubug6 naming)


def conv_bn(inp, oup, stride=1, leaky=0.0):
    """Builds a 3x3 convolution, BatchNorm and LeakyReLU block.

    Args:
        inp: Input channels.
        oup: Output channels.
        stride: Convolution stride. 2 halves the height and width.
        leaky: LeakyReLU negative slope. 0 makes it a plain ReLU.

    Returns:
        An nn.Sequential with the three layers.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride=1):
    """Builds a 3x3 convolution and BatchNorm block with no activation.

    Args:
        inp: Input channels.
        oup: Output channels.
        stride: Convolution stride.

    Returns:
        An nn.Sequential with the two layers.
    """
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


def conv_bn1x1(inp, oup, stride=1, leaky=0.0):
    """Builds a 1x1 convolution, BatchNorm and LeakyReLU block.

    A 1x1 convolution changes the channel count without looking at neighbouring pixels.
    Example: the FPN uses it to bring every backbone map to the same channel count.

    Args:
        inp: Input channels.
        oup: Output channels.
        stride: Convolution stride.
        leaky: LeakyReLU negative slope.

    Returns:
        An nn.Sequential with the three layers.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    """Builds a MobileNet depthwise-separable block.

    It splits a normal convolution into two cheaper steps: a 3x3 convolution on each
    channel separately (depthwise), then a 1x1 convolution that mixes channels
    (pointwise). Each step is followed by BatchNorm and LeakyReLU, so the block has six
    layers at indices 0 to 5.

    Args:
        inp: Input channels.
        oup: Output channels.
        stride: Stride of the depthwise step.
        leaky: LeakyReLU negative slope.

    Returns:
        An nn.Sequential with the six layers.
    """
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),  # depthwise: one filter per channel
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),  # pointwise: mix channels
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


# ----------------------------------------------------------------------------- MobileNetV1 x0.25


class MobileNetV1Body(nn.Module):
    """MobileNetV1 at 0.25 width, exactly as in biubug6, without its classifier.

    "0.25 width" means every layer has a quarter of the original MobileNet's channels,
    which makes it very fast (about 0.4M parameters). Example: a 640x640 input gives
    feature maps of 80x80x64, 40x40x128 and 20x20x256.
    """

    out_channels = [64, 128, 256]
    default_norm = "retinaface_bgr"

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(  # input -> stride 8
            conv_bn(3, 8, 2, leaky=0.1),
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(  # stride 8 -> stride 16
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))  # stride 16 -> stride 32

    def forward(self, x):
        """Returns the [C3, C4, C5] feature maps for a batch of images of shape (B, 3, H, W)."""
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]

    def stages(self):
        """Returns the three stages in order, for freezing."""
        return [self.stage1, self.stage2, self.stage3]


# ----------------------------------------------------------------------------- torchvision ResNet family

TORCHVISION_RESNETS = {  # backbone names accepted as model.backbone
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
}


class TorchvisionResNetBody(nn.Module):
    """A torchvision ResNet without its pooling and classifier layers.

    C3, C4 and C5 are the outputs of layer2, layer3 and layer4. Example: resnet50 gives
    512, 1024 and 2048 channels, while resnet34 gives 128, 256 and 512.

    Args:
        name: A name from TORCHVISION_RESNETS, for example "resnet50".
        pretrained: If True, download torchvision's ImageNet weights.
    """

    default_norm = "imagenet_rgb"

    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        import torchvision

        ctor = getattr(torchvision.models, name)  # e.g. torchvision.models.resnet50
        m = ctor(weights="DEFAULT" if pretrained else None)
        # Keep only the layers we need, under their torchvision names (matches biubug6 key names).
        self.conv1, self.bn1, self.relu, self.maxpool = m.conv1, m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

        def last_ch(layer):
            blk = layer[-1]  # last residual block of the layer
            return (blk.bn3 if hasattr(blk, "bn3") else blk.bn2).num_features  # bottleneck has bn3, basic has bn2

        self.out_channels = [last_ch(self.layer2), last_ch(self.layer3), last_ch(self.layer4)]

    def forward(self, x):
        """Returns the [C3, C4, C5] feature maps for a batch of images of shape (B, 3, H, W)."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # stem: stride 4
        x = self.layer1(x)  # stride 4
        c3 = self.layer2(x)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        return [c3, c4, c5]

    def stages(self):
        """Returns the stem and the four layers in order, for freezing."""
        return [nn.ModuleList([self.conv1, self.bn1]), self.layer1, self.layer2, self.layer3, self.layer4]


# ----------------------------------------------------------------------------- timm


class TimmBody(nn.Module):
    """Any timm model that can return feature maps at stride 8, 16 and 32.

    Example: model.backbone "timm:mobilenetv4_conv_small" builds that timm model and
    picks its stride 8, 16 and 32 outputs. Input normalisation is read from the timm
    model's own config.

    Args:
        name: The timm model name, without the "timm:" prefix.
        pretrained: If True, download timm's pretrained weights.
        **kwargs: Extra arguments for timm.create_model, for example out_indices.

    Raises:
        ImportError: If timm is not installed.
        ValueError: If the model has no feature map at one of the three strides.
    """

    def __init__(self, name: str, pretrained: bool, **kwargs):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("timm backbones need `pip install timm`") from e
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True, **kwargs)
        info = self.model.feature_info
        reductions = info.reduction()  # stride of each feature map timm returns
        channels = info.channels()  # channels of each feature map
        self.pick = []  # which timm outputs are C3, C4, C5
        for r in (8, 16, 32):
            if r not in reductions:
                raise ValueError(
                    f"timm model {name!r} exposes feature strides {reductions}; RetinaFace needs 8, 16 and 32. "
                    "Pick another model or pass out_indices / output_stride via model.timm_kwargs."
                )
            self.pick.append(reductions.index(r))
        self.out_channels = [channels[i] for i in self.pick]
        self.default_norm = self._resolve_norm(name)

    def _resolve_norm(self, name):
        """Reads the model's mean and std from timm, falling back to ImageNet values."""
        try:
            from timm.data import resolve_model_data_config

            dc = resolve_model_data_config(self.model)
            mean, std = dc.get("mean", IMAGENET_MEAN), dc.get("std", IMAGENET_STD)
        except Exception:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        return {"mean": [m * 255 for m in mean], "std": [s * 255 for s in std], "rgb": True}  # 0-1 -> 0-255 scale

    def forward(self, x):
        """Returns the [C3, C4, C5] feature maps for a batch of images of shape (B, 3, H, W)."""
        feats = self.model(x)
        return [feats[i] for i in self.pick]

    def stages(self):
        """Returns the timm model's top-level blocks in order, for freezing."""
        return list(self.model.children())


# ----------------------------------------------------------------------------- factory


def build_backbone(model_cfg: dict) -> nn.Module:
    """Builds the backbone named in the model config.

    Pretrained weights come from one of two places. If model.backbone_weights is set,
    that file is loaded. Otherwise torchvision or timm downloads ImageNet weights when
    backbone_pretrained is True. Example: {"backbone": "resnet34"} builds a resnet34 with
    torchvision's ImageNet weights.

    Args:
        model_cfg: The "model" section of the config. Reads backbone,
            backbone_pretrained, backbone_weights and timm_kwargs.

    Returns:
        A MobileNetV1Body, TorchvisionResNetBody or TimmBody.

    Raises:
        ValueError: If the backbone name is not recognised.
    """
    name = model_cfg["backbone"]
    pretrained = bool(model_cfg.get("backbone_pretrained", True))
    if name in ("mobilenet0.25", "mobilenetv1_0.25", "mobilenet"):
        body = MobileNetV1Body()
    elif name in TORCHVISION_RESNETS:
        # Skip the ImageNet download when a backbone_weights file is given.
        body = TorchvisionResNetBody(name, pretrained and not model_cfg.get("backbone_weights"))
    elif name.startswith("timm:"):
        body = TimmBody(
            name[5:], pretrained and not model_cfg.get("backbone_weights"), **(model_cfg.get("timm_kwargs") or {})
        )
    else:
        raise ValueError(f"Unknown backbone {name!r}. Use mobilenet0.25, a torchvision ResNet name, or timm:<name>.")
    if model_cfg.get("backbone_weights"):
        load_backbone_weights(body, model_cfg["backbone_weights"])
    elif name.startswith("mobilenet") and pretrained:
        warnings.warn(  # MobileNetV1 x0.25 is not in torchvision, so there is nothing to download
            "mobilenet0.25 has no auto-download ImageNet weights. Set model.backbone_weights to "
            "mobilenetV1X0.25_pretrain.tar, or init.checkpoint to a full RetinaFace checkpoint."
        )
    return body


def _strip(sd: dict) -> dict:
    """Removes common wrapper prefixes from weight names.

    Example: "module.body.stage1.0.0.weight" becomes "stage1.0.0.weight".

    Args:
        sd: A state dict (weight name -> tensor).

    Returns:
        A new state dict with the prefixes removed.
    """
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod.", "body.", "backbone."):  # DataParallel, torch.compile, RetinaFace, misc
            if k.startswith(p):
                k = k[len(p) :]
        out[k] = v
    return out


def load_backbone_weights(body: nn.Module, path: str | Path) -> None:
    """Loads a backbone-only checkpoint, such as mobilenetV1X0.25_pretrain.tar.

    Only weights whose name and shape match the backbone are loaded. Example: the
    MobileNet ImageNet file also holds a classifier, which is skipped.

    Args:
        body: The backbone to load into.
        path: Path to the checkpoint file.

    Raises:
        RuntimeError: If fewer than 90% of the backbone's weights were found, which
            usually means the wrong file or the wrong architecture.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    sd = obj.get("state_dict", obj.get("model", obj)) if isinstance(obj, dict) else obj  # unwrap common layouts
    sd = _strip(sd)
    own = body.state_dict()
    if isinstance(body, TimmBody):  # timm checkpoints are keyed without the "model." prefix
        sd = {("model." + k if "model." + k in own else k): v for k, v in sd.items()}
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}  # matching name and shape
    body.load_state_dict(ok, strict=False)
    # Count real weights only; num_batches_tracked is a BatchNorm counter, not a weight.
    n_own = len([k for k in own if not k.endswith("num_batches_tracked")])
    n_ok = len([k for k in ok if not k.endswith("num_batches_tracked")])
    print(f"[backbone] loaded {n_ok}/{n_own} tensors from {path}")
    if n_ok < 0.9 * n_own:
        raise RuntimeError(f"Backbone weights {path} matched only {n_ok}/{n_own} tensors - wrong file or architecture?")
