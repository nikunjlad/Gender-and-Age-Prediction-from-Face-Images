"""Torch-side tests. Skipped automatically when torch is not installed.

They check biubug6 key compatibility, cross-backbone transfer, staged freezing, loss, and export.
No network: backbones are built without pretrained weights.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# These imports need torch, so they come after importorskip (hence noqa: E402).
from face_detection.models.anchors import generate_priors  # noqa: E402
from face_detection.models.checkpoint import read_checkpoint, save_checkpoint, transfer_weights  # noqa: E402
from face_detection.config import DEFAULTS, deep_merge  # noqa: E402
from face_detection.models.finetune import FreezeController, LRSchedule, ModelEMA, build_optimizer  # noqa: E402
from face_detection.models.loss import MultiBoxLoss  # noqa: E402
from face_detection.models.model import RetinaFace  # noqa: E402
from face_detection.models.postprocess import Detector, ExportWrapper  # noqa: E402
from face_detection.models.targets import match  # noqa: E402


def make_cfg(backbone="mobilenet0.25", out_channel=64, **model):
    """Returns a default config for the given backbone, with no pretrained download."""
    return deep_merge(
        DEFAULTS, {"model": {"backbone": backbone, "out_channel": out_channel, "backbone_pretrained": False, **model}}
    )


@pytest.fixture(scope="module")
def mnet():
    """A MobileNet RetinaFace model shared by the tests in this file."""
    torch.manual_seed(0)
    return RetinaFace(make_cfg())


def test_biubug6_key_names(mnet):
    """Weight names match biubug6's checkpoints for MobileNet and ResNet50, so they load directly."""
    keys = set(mnet.state_dict())
    for k in [
        "body.stage1.0.0.weight",
        "body.stage3.1.4.running_var",
        "fpn.output1.0.weight",
        "fpn.merge2.1.bias",
        "ssh1.conv3X3.0.weight",
        "ssh3.conv7x7_3.1.running_mean",
        "ClassHead.0.conv1x1.weight",
        "BboxHead.2.conv1x1.bias",
        "LandmarkHead.1.conv1x1.weight",
    ]:
        assert k in keys, k
    assert not any("pixel_mean" in k for k in keys)  # normalisation buffers are non-persistent
    r50 = RetinaFace(make_cfg("resnet50", 256))
    keys = set(r50.state_dict())
    for k in ["body.conv1.weight", "body.bn1.running_mean", "body.layer4.2.bn3.weight", "fpn.output3.0.weight"]:
        assert k in keys, k
    assert r50.state_dict()["fpn.output3.0.weight"].shape == (256, 2048, 1, 1)


def test_forward_shapes(mnet):
    """A forward pass returns one row per anchor for loc (4), conf (2) and landm (10)."""
    mnet.eval()
    x = torch.randint(0, 256, (2, 3, 640, 640)).float()
    loc, conf, landm = mnet(x)
    p = len(generate_priors((640, 640), DEFAULTS["anchors"]["min_sizes"], DEFAULTS["anchors"]["steps"]))
    assert loc.shape == (2, p, 4) and conf.shape == (2, p, 2) and landm.shape == (2, p, 10)
    mnet.check_strides(640)


@pytest.mark.parametrize("bb,oc", [("resnet18", 64), ("resnet34", 256), ("resnet50", 256)])
def test_torchvision_backbones(bb, oc):
    """ResNet18/34/50 backbones build with correct strides and default to ImageNet RGB normalisation."""
    m = RetinaFace(make_cfg(bb, oc)).eval()
    m.check_strides(320)
    assert m.norm["rgb"]  # auto -> imagenet_rgb


def test_same_backbone_transfer_is_complete(tmp_path, mnet):
    """A checkpoint with DataParallel 'module.' prefixes loads every weight back exactly."""
    path = tmp_path / "biubug6_style.pth"
    torch.save({"module." + k: v for k, v in mnet.state_dict().items()}, path)  # DataParallel-style keys
    sd, meta = read_checkpoint(path)
    fresh = RetinaFace(make_cfg())
    rep = transfer_weights(fresh, sd, require=["body", "fpn", "ssh", "heads"], min_fraction=1.0)
    assert not rep.mismatched and not rep.unexpected
    for k, v in mnet.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k


def test_cross_backbone_transfer_r50_to_r34():
    """Transferring R50 -> R34 keeps SSH and heads, and skips each FPN input conv together with its BatchNorm."""
    torch.manual_seed(0)
    src = RetinaFace(make_cfg("resnet50", 256)).state_dict()
    dst = RetinaFace(make_cfg("resnet34", 256))
    rep = transfer_weights(dst, src, exclude=["body."])
    assert rep.fraction("ssh") == 1.0 and rep.fraction("heads") == 1.0
    assert rep.fraction("body") == 0.0
    # laterals: conv mismatches -> its BN (same shape!) must NOT load either (atomic units)
    assert "fpn.output1.1.weight" not in rep.loaded and "fpn.output1.0.weight" not in rep.loaded
    assert "fpn.merge1.0.weight" in rep.loaded and "fpn.merge1.1.running_var" in rep.loaded
    assert torch.equal(dst.state_dict()["ssh2.conv5X5_1.0.weight"], src["ssh2.conv5X5_1.0.weight"])


def test_require_raises_on_wrong_checkpoint(mnet):
    """Requiring the heads to load fails loudly when the checkpoint's heads have a different shape."""
    r50 = RetinaFace(make_cfg("resnet50", 256))
    with pytest.raises(RuntimeError, match="Required groups"):
        transfer_weights(r50, mnet.state_dict(), require=["heads"])


def _batch(model, n=2, size=320):
    """Builds a small batch of random images with matched targets for loss tests."""
    a = DEFAULTS["anchors"]
    pr = generate_priors((size, size), a["min_sizes"], a["steps"])
    ts = []
    for i in range(n):
        boxes = np.array([[0.1, 0.1, 0.3, 0.35], [0.5, 0.4, 0.62, 0.55]], np.float32)
        lms = np.tile(np.array([0.2, 0.2], np.float32), 5)[None].repeat(2, 0)
        ts.append(match(boxes, lms, np.array([True, False]), pr).as_dict())
    tgt = {k: torch.from_numpy(np.stack([t[k] for t in ts])) for k in ts[0]}
    return torch.randint(0, 256, (n, 3, size, size)).float(), tgt


def test_loss_backward_and_freezing():
    """While frozen the backbone (weights and BN stats) does not change; after unfreezing it trains."""
    torch.manual_seed(0)
    cfg = make_cfg()
    cfg["train"].update(freeze_backbone_epochs=1, epochs=3, warmup_epochs=0.5)
    model = RetinaFace(cfg)
    crit = MultiBoxLoss()
    opt = build_optimizer(model, cfg["train"])
    sched = LRSchedule(cfg["train"], iters_per_epoch=2)
    fz = FreezeController(model, 1, 0, False)
    imgs, tgt = _batch(model)

    fz.apply(0)
    model.train()
    fz.set_train_mode()
    body_before = {k: v.clone() for k, v in model.body.state_dict().items()}
    sched.step(opt, 0, 1)
    assert all(g["lr"] == 0 for g in opt.param_groups if g["part"] == "backbone")
    loss, parts = crit(model(imgs), tgt)
    assert torch.isfinite(loss) and parts["landm"] > 0
    opt.zero_grad()
    loss.backward()
    opt.step()
    for k, v in model.body.state_dict().items():  # weights AND BN running stats untouched
        assert torch.equal(v, body_before[k]), k

    fz.apply(1)
    model.train()
    fz.set_train_mode()
    sched.step(opt, 1, 1)
    assert all(g["lr"] > 0 for g in opt.param_groups)
    loss, _ = crit(model(imgs), tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(not torch.equal(v, body_before[k]) for k, v in model.body.state_dict().items())


def test_ema_and_checkpoint_roundtrip(tmp_path, mnet):
    """A saved checkpoint with EMA weights reloads safely and reports its EMA source and config."""
    ema = ModelEMA(mnet)
    ema.update(mnet)
    cfg = make_cfg()
    p = tmp_path / "ck.pt"
    save_checkpoint(p, mnet, cfg, 0, ema, metrics={"ap50": 0.5}, best={"metric": "ap50", "value": 0.5, "epoch": 0})
    sd, meta = read_checkpoint(p)  # weights_only=True must work on our own format
    assert meta["source"] == "ema" and meta["cfg"]["model"]["backbone"] == "mobilenet0.25"


def test_detector_and_export_wrapper(mnet):
    """Detector handles odd image sizes and the export wrapper returns pixel boxes and 0-1 scores."""
    cfg = make_cfg()
    det = Detector(mnet, cfg, conf_thresh=0.0, keep_top_k=10)
    out = det(np.random.randint(0, 255, (300, 450, 3), dtype=np.uint8))  # non-multiple-of-32 input
    assert out.shape[1] == 15 and len(out) <= 10
    w = ExportWrapper(mnet, cfg, 320, 320).eval()
    b, s, lm = w(torch.zeros(1, 3, 320, 320))
    assert b.shape[-1] == 4 and lm.shape[-1] == 10 and s.shape == b.shape[:2]
    assert float(s.min()) >= 0 and float(s.max()) <= 1
