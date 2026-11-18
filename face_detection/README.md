# RetinaFace transfer learning

Train RetinaFace on any backbone, starting from a pretrained RetinaFace checkpoint.

- **Backbones:** MobileNet-0.25, torchvision ResNets, or any `timm` model that produces stride 8, 16 and 32 feature maps.
- **Same-backbone fine-tuning:** biubug6 checkpoints load directly, because parameter names match `biubug6/Pytorch_Retinaface` exactly (`body.*`, `fpn.*`, `ssh*`, `ClassHead/BboxHead/LandmarkHead`).
- **Cross-backbone transfer:** the neck and heads (`fpn.merge*`, `ssh*`, the three heads) move onto a new backbone, provided `out_channel` matches.
- **Staged schedule:** a frozen-backbone warm-up, then unfreezing with a lower, ramped backbone learning rate. Partial stage freezing, frozen BatchNorm, EMA and AMP are all supported.

## Install

Dependencies are managed with uv at the repo root (`pyproject.toml`, `uv.lock`). direnv
activates `.venv` and puts the repo root on `PYTHONPATH`, which the package imports need
(`from face_detection.models.model import ...`). Without direnv, set it yourself:
`export PYTHONPATH=<repo root>`.

```bash
# from the repo root
uv sync                                   # install dependencies into .venv
pytest -q face_detection/tests            # numpy pipeline + torch model tests
ruff check face_detection                 # lint (incl. Google-style docstrings)
ruff format face_detection                # format (line length 120)
```

## Layout

Run the scripts from inside `face_detection/`; paths such as `configs/`, `data/` and
`results/` are relative to it.

```
train.py        train or fine-tune
evaluate.py     WIDER FACE benchmark -> results/<split>/<name>/
infer.py        detect faces in an image or a folder
config.py       defaults, YAML loading, dotted CLI overrides
configs/        one YAML file per scenario
models/         backbones, RetinaFace network, anchors, targets, loss, decoding, checkpoints, fine-tuning
processing/     label parsing, training Dataset, augmentation
metrics/        quick AP, WIDER benchmark metrics, training metric tracking (CSV / MLflow)
tools/          export_onnx, plot_pr, inspect_checkpoint, split_val_test (one-off)
tests/          pytest suite
data/  weights/  runs/  results/    datasets, checkpoints, training runs, evaluation outputs
```

## Pretrained checkpoints

Download them from the links in the [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) README and put them in `weights/`:

| file | what it is | out_channel |
|---|---|---|
| `mobilenet0.25_Final.pth` | full RetinaFace, MobileNet-0.25 | 64 |
| `Resnet50_Final.pth` | full RetinaFace, ResNet-50 | 256 |
| `mobilenetV1X0.25_pretrain.tar` | ImageNet MobileNet-0.25 backbone only | n/a |

Before you train, inspect a checkpoint or dry-run the transfer into a config:

```bash
python tools/inspect_checkpoint.py --checkpoint weights/Resnet50_Final.pth --config configs/resnet34_transfer.yaml
```

## Data

The label format is WIDER FACE / `retinaface_gt` (the same one biubug6 uses). It is one file per split:

```
# relative/path/to/image.jpg
x y w h  lx0 ly0 0.0  lx1 ly1 0.0  lx2 ly2 0.0  lx3 ly3 0.0  lx4 ly4 0.0  score
x y w h                                   <- boxes without landmarks are fine
# another/image.jpg                       <- no face lines = background-only image
```

- **Landmark order:** left eye, right eye, nose, left mouth corner, right mouth corner, with left and right as seen in the image.
- **Missing landmarks:** use `-1` for every value. Those faces then still train the box and class heads.
- **Your own data:** convert it to this format, set `data.*` in the config, and point `val_labels` at a held-out split.
- **WIDER FACE test split:** the official test labels are not public, so WIDER val was split locally into `val/` and `test/` with `tools/split_val_test.py`. `data/widerface/test/split_manifest.json` records how.

## Scenarios (configs/)

| config | backbone | initialised from | what transfers |
|---|---|---|---|
| `mobilenet025_finetune.yaml` | MobileNet-0.25 | `mobilenet0.25_Final.pth` | everything |
| `resnet50_finetune.yaml` | ResNet-50 | `Resnet50_Final.pth` | everything |
| `resnet34_transfer.yaml` | ResNet-34, ImageNet weights | `Resnet50_Final.pth` | `fpn.merge*`, `ssh*`, heads |
| `timm_mobilenetv4_transfer.yaml` | timm MobileNetV4-M, ImageNet weights | `mobilenet0.25_Final.pth` | `fpn.merge*`, `ssh*`, heads |

To use another timm model, change `model.backbone: timm:<name>`. Pick the source checkpoint by width: `out_channel: 64` uses the MobileNet checkpoint as source, and `256` uses the ResNet-50 one.

## How the transfer works

1. **Shape-matched copy.** Every tensor whose name and shape match is copied. Prefixes such as `module.` and `_orig_mod.` are stripped first. `init.rename` handles checkpoints from other repos.
2. **Atomic units.** Each conv+BN pair, residual block or head loads all-or-nothing. If a lateral conv `fpn.output1.0` mismatches because the backbone widths differ, its BatchNorm is re-initialised too, even though the BN shape happens to match.
3. **Explicit filters.** `init.exclude` and `init.include` are explicit. In cross-backbone configs, exclude `body.` because ResNet-50 and ResNet-34 share stem shapes and would otherwise silently mix.
4. **Fail-fast guard.** `init.require: [ssh, heads]` aborts training if those groups did not transfer at least `min_fraction`. This catches a wrong file or a wrong `out_channel` before you burn GPU hours.
5. **A report is printed for every run.** Example (illustrative numbers):

```
Weight transfer report
group           tensors                params
body          0/122          0.00M/21.28M (  0.0%)
fpn           6/15           1.18M/1.40M ( 84.2%)
ssh          36/36           1.77M/1.77M (100.0%)
heads        18/18           0.02M/0.02M (100.0%)
```

6. **Input normalisation is part of the model.** The model takes BGR images with values 0-255, straight from `cv2.imread`.
   - `input_norm: retinaface_bgr` is mean subtraction only, which is what biubug6 checkpoints expect.
   - `auto` picks the backbone's own pretraining statistics: ImageNet for torchvision, and the pretrained config for timm.
   - A warning is printed if a transferred backbone looks like it was trained with different normalisation.

## Training schedule

```
epoch:      0 ─ warmup ─┬─ freeze_backbone_epochs ─┬─ unfreeze_warmup ─┬────── cosine ──────▶
neck/heads: lr ramps up │ full lr                  │                   │ cosine decay
backbone:   frozen, BN in eval (stats untouched)   │ lr: 0 → lr×mult   │ cosine decay
```

Key knobs:

| knob | what it does | guidance |
|---|---|---|
| `freeze_backbone_epochs` | trains only the neck and heads at first | 3–5 epochs. This matters most when the laterals start fresh. |
| `backbone_lr_mult` | backbone LR relative to the neck/heads | 0.1 for a face-trained body. 0.2–0.3 for ImageNet bodies, which need more adaptation. |
| `freeze_stages` | keeps the earliest backbone stages frozen for the whole run | Good for small datasets. For timm models this counts top-level children; the frozen names are printed. |
| `freeze_backbone_bn` | keeps backbone BatchNorm statistics fixed | Use when the batch is below about 16 per GPU. |
| `loss.min_negatives_per_image` | sets a minimum number of hard negatives per image | Set above 0 if you add background-only images to suppress false positives. The original recipe takes none from images without faces. |

## Commands

```bash
# train / fine-tune
python train.py --config configs/resnet34_transfer.yaml
python train.py --config configs/mobilenet025_finetune.yaml train.lr=5e-4 train.epochs=30
python train.py --resume runs/<run>/last.pt

# benchmark on WIDER FACE: official Easy/Medium/Hard AP, COCO-style AP, P/R/F1
# writes results/<split>/<name>/metrics.json (incl. PR curves); predictions are cached
python evaluate.py --checkpoint runs/<run>/best.pt --name my_model --split val
# add --vis to also save every image with boxes to results/<split>/<name>/vis/ (Hard setting, IoU 0.5, --conf-op):
#   green = matched prediction (TP), red = false positive (FP), blue = label of a missed face (FN)
# biubug6 checkpoint as a baseline (checkpoints without an embedded config need --config):
python evaluate.py --checkpoint weights/Resnet50_Final.pth --config configs/resnet50_finetune.yaml \
    --name resnet50 --split test
# compare models: PR curves -> results/<split>/pr_curves.png
python tools/plot_pr.py test resnet50 mobilenet0.25

# run on images
python infer.py --checkpoint runs/<run>/best.pt --source samples/ --out out/ --conf 0.6

# ONNX with decoding baked in (fixed size; NMS left to TensorRT/DeepStream)
python tools/export_onnx.py --checkpoint runs/<run>/best.pt --height 640 --width 640 --out retinaface.onnx
trtexec --onnx=retinaface.onnx --fp16 --saveEngine=retinaface.plan
```

What each run directory under `runs/<name>_<timestamp>/` contains:

- `config.yaml`: the resolved config.
- `results.csv`: per-epoch losses, AP and timing.
- `last.pt`: written every epoch and holds everything needed to resume.
- `best.pt`: chosen by AP if `ap_every > 0`, otherwise by validation loss.
- `epoch_XXX.pt`: written every `save_every` epochs.

Checkpoints store both the raw and the EMA weights; inference uses the EMA weights by default. Set `train.mlflow: true` to log parameters and metrics to MLflow.

## Notes

- **Anchors.** Keep `anchors.min_sizes` and `steps` the same as the source checkpoint when you transfer heads. Changing them keeps the tensor shapes, so the heads still load, but their learned offsets then refer to different anchors.
- **Image size.** `image_size` can be any multiple of 8; biubug6 trains ResNet-50 at 840. `check_strides()` verifies at startup that the backbone's real feature-map sizes match the anchor grid. Timm models that use floor-mode downsampling need multiples of 32.
- **Multi-GPU.** Training runs on a single GPU. Unfreezing parameters partway through a run requires re-wrapping a DDP model, which isn't implemented here.
- **Licensing.** The biubug6 code is MIT-licensed. WIDER FACE is distributed for non-commercial research, so check its terms before shipping weights trained on it, and fine-tune on data you are licensed to use.
