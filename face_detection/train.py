#!/usr/bin/env python3
"""Train or fine-tune RetinaFace with transfer learning.

Run from face_detection/ (PYTHONPATH must include the repo root):
    python train.py --config configs/resnet34_transfer.yaml
    python train.py --config configs/mobilenet025_finetune.yaml train.lr=5e-4 train.epochs=30
    python train.py --resume runs/<run>/last.pt

Each run writes to runs/<name>_<timestamp>/: config.yaml, results.csv (one row per
epoch), last.pt (latest), best.pt (best so far) and epoch_NNN.pt every save_every epochs.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from face_detection.models.anchors import generate_priors
from face_detection.processing.annotations import parse_label_file
from face_detection.models.checkpoint import init_from_config, save_checkpoint
from face_detection.config import load_config, save_config
from face_detection.processing.data import FaceDetectionDataset, worker_init_fn
from face_detection.models.finetune import FreezeController, LRSchedule, ModelEMA, build_optimizer
from face_detection.models.loss import MultiBoxLoss
from face_detection.models.model import RetinaFace
from face_detection.models.targets import TargetAssigner
from face_detection.processing.transforms import TrainTransform, ValTransform

from face_detection.metrics.ap import evaluate_ap
from face_detection.metrics.tracking import Meter, MlflowLogger


def seed_everything(seed: int) -> None:
    """Seeds Python, numpy and torch (CPU and GPU) so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate_loss(model, loader, criterion, device, amp_ctx) -> dict:
    """Computes the average loss over the validation set, without training.

    Args:
        model: The model to evaluate (usually the EMA copy).
        loader: Validation DataLoader.
        criterion: The MultiBoxLoss.
        device: Device to run on.
        amp_ctx: Function returning the mixed-precision context.

    Returns:
        A dict with the average "loss", "loc", "cls" and "landm".
    """
    model.eval()
    m = Meter()
    for imgs, tgt in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}
        with amp_ctx():
            preds = model(imgs)
        loss, parts = criterion(preds, tgt)
        m.add({"loss": loss.item(), **{k: v.item() for k, v in parts.items()}}, imgs.shape[0])
    return m.avg()


def main():
    """Parses arguments, builds everything, and runs the training loop."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config")
    ap.add_argument("--resume", help="resume from runs/<run>/last.pt (uses its embedded config)")
    ap.add_argument("--trust", action="store_true", help="allow weights_only=False for the init checkpoint")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=5e-4")
    args = ap.parse_args()

    # ---------------------------------------------------------------- config + run folder
    resume_obj = None
    if args.resume:
        resume_obj = torch.load(args.resume, map_location="cpu", weights_only=True)
        from face_detection.config import DEFAULTS, apply_overrides, deep_merge, validate

        cfg = apply_overrides(deep_merge(DEFAULTS, resume_obj["cfg"]), args.overrides)  # config saved in the checkpoint
        validate(cfg)
        run_dir = Path(args.resume).resolve().parent  # keep writing to the same run folder
    else:
        if not args.config:
            ap.error("--config is required unless --resume is given")
        cfg = load_config(args.config, args.overrides)
        if args.trust:
            cfg["init"]["trust_checkpoint"] = True
        run_dir = Path(cfg["train"]["output_dir"]) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")  # record exactly what this run used

    t, d, lc = cfg["train"], cfg["data"], cfg["loss"]
    seed_everything(t["seed"])
    device = torch.device(t["device"] if (t["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # pick the fastest conv algorithms for fixed input sizes
    print(f"run dir: {run_dir}  device: {device}")

    # ---------------------------------------------------------------- model + transfer
    model = RetinaFace(cfg)
    model.check_strides(d["image_size"])  # fail early if the backbone strides are wrong
    if resume_obj is None:
        rep = init_from_config(model, cfg)  # load starting weights from init.checkpoint
        if rep is None and not cfg["model"]["backbone_pretrained"] and not cfg["model"]["backbone_weights"]:
            print("[warn] training from scratch: no init.checkpoint, no pretrained backbone")
    print(f"input normalisation: {model.norm}")
    model.to(device)

    # ---------------------------------------------------------------- data
    S = d["image_size"]
    a = cfg["anchors"]
    priors = generate_priors((S, S), a["min_sizes"], a["steps"], a["clip"])  # same anchors for every training image
    assigner = TargetAssigner(priors, a["variance"], lc["pos_iou"], lc["min_gt_best_iou"])
    train_tf = TrainTransform(S, d["crop_scales"], d["min_face_px"], d["color_jitter"], d["hflip"], seed=t["seed"])
    train_ds = FaceDetectionDataset(d["train_labels"], d["train_root"], train_tf, assigner)
    print(f"train: {train_ds.describe()}  priors/image: {len(priors)}")
    loader_kw = dict(
        num_workers=d["workers"],
        pin_memory=device.type == "cuda",  # faster copies to the GPU
        persistent_workers=d["workers"] > 0,  # keep workers alive between epochs
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    if len(train_loader) == 0:
        raise ValueError(f"Fewer images ({len(train_ds)}) than batch_size ({t['batch_size']})")

    val_loader, ap_samples = None, []
    if d.get("val_labels"):
        val_ds = FaceDetectionDataset(d["val_labels"], d["val_root"], ValTransform(S), assigner)
        print(f"val:   {val_ds.describe()}")
        val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, **loader_kw)
        if t["ap_every"] > 0:
            ap_samples = parse_label_file(d["val_labels"])
            rng = random.Random(t["seed"])
            if t["ap_max_images"] and len(ap_samples) > t["ap_max_images"]:
                ap_samples = rng.sample(ap_samples, t["ap_max_images"])  # fixed random subset for speed

    # ---------------------------------------------------------------- optimisation
    criterion = MultiBoxLoss(lc["neg_pos_ratio"], lc["loc_weight"], lc["landm_weight"], lc["min_negatives_per_image"])
    optimizer = build_optimizer(model, t)
    sched = LRSchedule(t, len(train_loader))
    freezer = FreezeController(model, t["freeze_backbone_epochs"], t["freeze_stages"], t["freeze_backbone_bn"])
    amp_dtype = torch.bfloat16 if t["amp_dtype"] == "bfloat16" else torch.float16
    use_amp = bool(t["amp"]) and device.type == "cuda"
    try:
        # The gradient scaler prevents float16 gradients from rounding to zero; bfloat16 does not need it.
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    def amp_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    ema = ModelEMA(model, t["ema_decay"]) if t["ema"] else None
    start_epoch, best = 0, None
    if resume_obj is not None:  # restore everything needed to continue exactly where we stopped
        model.load_state_dict(resume_obj["model"])
        if ema is not None and resume_obj.get("ema"):
            ema.module.load_state_dict(resume_obj["ema"])
            ema.updates = resume_obj["epoch"] * len(train_loader)
        if resume_obj.get("optimizer"):
            optimizer.load_state_dict(resume_obj["optimizer"])
        if resume_obj.get("scaler"):
            scaler.load_state_dict(resume_obj["scaler"])
        start_epoch, best = resume_obj["epoch"] + 1, resume_obj.get("best")
        print(f"resumed at epoch {start_epoch}, best {best}")

    mlf = MlflowLogger(t["mlflow"], cfg, run_dir)
    csv_path = run_dir / "results.csv"  # one row per epoch
    csv_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_loc",
        "train_cls",
        "train_landm",
        "val_loss",
        "val_loc",
        "val_cls",
        "val_landm",
        "ap50",
        "recall",
        "time_s",
    ]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, t["epochs"]):
        msg = freezer.apply(epoch)  # freeze / unfreeze the backbone for this epoch
        if msg:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[epoch {epoch}] backbone {msg}; trainable params {n_train / 1e6:.2f}M")
        model.train()
        freezer.set_train_mode()  # keep frozen BatchNorms in eval mode
        meter, t0, lr = Meter(), time.time(), 0.0
        trainable = [p for p in model.parameters() if p.requires_grad]
        for it, (imgs, tgt) in enumerate(train_loader):
            lr = sched.step(optimizer, epoch, it)
            imgs = imgs.to(device, non_blocking=True).float()
            tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}
            with amp_ctx():
                preds = model(imgs)
            loss, parts = criterion(preds, tgt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} iter {it}: {parts}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if t["grad_clip"]:
                scaler.unscale_(optimizer)  # clip real gradient values, not scaled ones
                torch.nn.utils.clip_grad_norm_(trainable, t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            meter.add({"loss": loss.item(), **{k: v.item() for k, v in parts.items()}}, imgs.shape[0])
            if it % t["log_every"] == 0:
                m = meter.avg()
                print(
                    f"ep {epoch:3d} it {it:5d}/{len(train_loader)} lr {lr:.2e} "
                    f"loss {m['loss']:.4f} (loc {m['loc']:.4f} cls {m['cls']:.4f} landm {m['landm']:.4f})"
                )

        # ------------------------------------------------------------ end of epoch: evaluate
        tr = meter.avg()
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": tr["loss"],
            "train_loc": tr["loc"],
            "train_cls": tr["cls"],
            "train_landm": tr["landm"],
        }
        eval_model = ema.module if ema is not None else model  # evaluate the smoother EMA weights
        last = epoch == t["epochs"] - 1

        if val_loader is not None and ((epoch + 1) % t["val_every"] == 0 or last):
            v = validate_loss(eval_model, val_loader, criterion, device, amp_ctx)
            row.update(val_loss=v["loss"], val_loc=v["loc"], val_cls=v["cls"], val_landm=v["landm"])
            print(
                f"[val] epoch {epoch} loss {v['loss']:.4f} (loc {v['loc']:.4f} cls {v['cls']:.4f} landm {v['landm']:.4f})"
            )
        if ap_samples and ((epoch + 1) % t["ap_every"] == 0 or last):
            r = evaluate_ap(eval_model, cfg, ap_samples, d["val_root"], max_size=t["ap_max_size"])
            row.update(ap50=r["ap"], recall=r["recall"])
            print(
                f"[val] epoch {epoch} AP@{cfg['eval']['iou_thresh']}: {r['ap']:.4f} recall {r['recall']:.4f} "
                f"on {len(ap_samples)} images"
            )
        row["time_s"] = round(time.time() - t0, 1)

        # model selection: AP when available, else validation loss, else train loss
        if row.get("ap50") is not None:
            cand = ("ap50", row["ap50"])
        elif row.get("val_loss") is not None and not ap_samples:
            cand = ("neg_val_loss", -row["val_loss"])  # negated so higher is better
        elif val_loader is None:
            cand = ("neg_train_loss", -row["train_loss"])
        else:
            cand = None  # AP is configured but not computed this epoch
        improved = cand is not None and (best is None or best["metric"] != cand[0] or cand[1] > best["value"])
        if improved:
            best = {"metric": cand[0], "value": float(cand[1]), "epoch": epoch}

        # ------------------------------------------------------------ end of epoch: save
        metrics = {k: float(v) for k, v in row.items() if v is not None}
        save_checkpoint(run_dir / "last.pt", model, cfg, epoch, ema, optimizer, scaler, metrics, best)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, ema, None, None, metrics, best)
            print(f"[ckpt] new best {best['metric']}={best['value']:.4f} at epoch {epoch}")
        if t["save_every"] and (epoch + 1) % t["save_every"] == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, cfg, epoch, ema, None, None, metrics, best)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, csv_fields).writerow({k: row.get(k) for k in csv_fields})
        mlf.log(metrics, epoch)

    if (run_dir / "best.pt").exists():
        mlf.artifact(run_dir / "best.pt")
    mlf.close()
    print(f"done. best: {best}  checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
