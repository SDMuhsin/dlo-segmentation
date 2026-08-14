# DLO segmentation

Semantic segmentation of deformable linear objects (cables, wire harnesses) from RGB
images. A synthetic dataset is rendered from labelled 4096-point harness point clouds,
and a SegFormer-B5 is trained to label each pixel as background, wire or connector.

This file covers how to reproduce the August 2026 result from scratch: assets, rendered
dataset, trained checkpoint, metrics.

## Result

Held-out validation split, 1080 frames, three wire assemblies not used in training.
Plain argmax at native resolution.

| Metric | July | August |
|---|---|---|
| IoU, connector | 0.7625 | 0.8025 |
| IoU, wire | 0.8603 | 0.8784 |
| Mean IoU | 0.8735 | 0.8929 |
| Connector precision | 0.869 | 0.894 |
| Connector recall | 0.861 | 0.886 |

Two training-side changes; the dataset is the same one used for the July numbers.

1. Lovasz-Softmax auxiliary loss (`--lovasz-weight 0.5`). A differentiable surrogate for
   IoU, rather than scoring each pixel on its own.
2. Scale augmentation (`--aug-zoom 0.5,0.5,1.0`). With p=0.5, crop a random [0.5, 1.0]
   fraction and resize back, so connectors appear at up to twice their normal size.

## Setup

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

Everything was run on one NVIDIA A40 (48 GB). Training peaks around 19.3 GB, so a 24 GB
card is enough.

## Reproducing

### 0. Fetch the input assets (about 1.1 GB)

`data/` is gitignored, so a fresh clone has no inputs at all. One script pulls them from
the original public sources:

```bash
python scripts/download_assets.py --all      # roughly 15 min
python scripts/download_assets.py --verify   # report what is present
```

| Group | Goes to | From |
|---|---|---|
| 22 surface textures | `data/textures/` | ambientCG, CC0 |
| 11 photo backdrops | `data/textures/backgrounds*/` | Poly Haven, CC0 |
| 21 clutter point clouds | `data/objects/` | shipped in `assets/phase4_objects/`, CC0 |
| 24 harness point clouds | `data/set2/` | [PointWire](https://github.com/heti2000/cdlo-datasets), MIT |

None of these need an institutional account.

PointWire is published as one 50 GiB zip, but the renderer only reads three
subdirectories per harness. The script reads the archive index over HTTP range requests
and pulls just those members, which is about 0.9 GB of transfer rather than 50 GiB.

It fetches 24 harnesses, not all 40 in the archive. Splits are assigned by harness ID,
and the reference `data/` contains one incomplete harness (005, empty
`pointclouds_normed_4096`) that the renderer skips during discovery. Downloading all 40
would add source frames and give you a larger dataset than the one these numbers came
from.

The clutter point clouds cannot be re-downloaded. They were sampled from Poly Haven
meshes and the sampling code is gone, so the 21 files (2 MB, CC0) are committed under
`assets/phase4_objects/` and the script copies them into place.

One input is not covered by the script: the warm-start checkpoint at
`results/realism_campaign/p39_cutoutneg/segformer_b5_warmstart/epoch_10.pth`. See step 3.

### 1. Render (8,640 frames, about 40 min on 8 workers)

```bash
KIAT_OUTPUT_ROOT=data/render_3way_connscale3 \
KIAT_DATASET_MODE=phase13 \
KIAT_BG_DIR=data/textures/backgrounds_p4orig11 \
KIAT_DLO_UNICOLOR=1 \
KIAT_CONNECTOR_SCALE=3.0 \
python src/render_full_dataset.py --workers 8
```

1,440 source frames at 6 views each, so 8,640 RGB/depth/label images. 21 training
assemblies (1,260 sources) and 3 held out (180 sources). The holdout is at assembly
level, so no validation wire appears in training.

Set IDs are not contiguous. Train is 000, 002, 003, 004, 006 and so on up to 031;
validation is 032, 034, 035. Count sets rather than assuming a range.

The two variables that define this task:

- `KIAT_DLO_UNICOLOR=1` paints the whole harness a single wire colour, so colour is no
  longer a shortcut for finding the connector and the model has to use shape and context.
- `KIAT_CONNECTOR_SCALE=3.0` splats connector points with a disc about 3x the wire's
  on-screen width. This is the geometric cue that made the class learnable at all; it
  took connector IoU from 0.47 to 0.76 in July.

Keep `KIAT_BG_DIR` pointed at `backgrounds_p4orig11` (11 photos). The larger
`data/textures/backgrounds/` pool was tested and made real-camera performance much worse.

### 2. Stage and re-encode to three classes

```bash
# render layout -> flat RGB/Depth/Label + train.txt/test.txt, 6-class labels
python src/prepare_dformer_data.py \
    --src-root data/render_3way_connscale3 \
    --dst-root data/dformer_dataset_3way_connscale3_raw \
    --clean

# 6-class {bg,wire,endpoint,bifurcation,connector,noise} -> 3-class {bg,wire,connector}
python src/reencode_labels_3way.py \
    --src data/dformer_dataset_3way_connscale3_raw \
    --dst data/dformer_dataset_3way_connscale3
```

Re-encoding maps {0,5} to 0 (background), {1,2,3} to 1 (the whole cable body) and {4} to
2 (connector). RGB and depth are symlinked rather than copied.

You should end up with 7,560 lines in `train.txt` and 1,080 in `test.txt`.

### 3. Warm-start checkpoint

Training starts from a binary (background/wire) SegFormer-B5 built during the earlier
real-world work. It cannot be rebuilt from this repository: it was trained on third-party
real-cable data (MovingCables, CC-BY-SA) plus photographic object cutouts that are not
redistributed here. Either copy the checkpoint across, or drop `--init-checkpoint` in
step 4 and start from the ImageNet-pretrained `nvidia/mit-b5`. The two levers should
still help in that case, but the absolute numbers above will not match.

### 4. Train (30 epochs, 5 h 52 m on one A40)

```bash
bash scripts/train_3way_lovasz_ab.sh <gpu> 0.5 lovasz050_zoom --aug-zoom 0.5,0.5,1.0
```

Output goes to `results/realism_campaign/p3w_lovasz/lovasz050_zoom/`. The script pins
every other flag to the July baseline (`--num-classes 3 --class-weights 1,6,4
--dlo-weight 6.0 --batch-size 8 --lr 6e-5 --warmup-epochs 5 --seed 1234`), so the Lovasz
weight and the zoom flag are the only things that vary. Drop the trailing `--aug-zoom`
to train the Lovasz-only arm, which gives 0.7836 / 0.8791 / 0.8869.

### 5. Evaluate

```bash
python src/eval_3way_final.py \
    --ckpt results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth \
    --data-dir data/dformer_dataset_3way_connscale3 \
    --out results/realism_campaign/p3w_lovasz/eval_final.json
```

Prints pooled IoU and a per-set breakdown. Look at the per-set table as well as the
pooled figure: set 035 holds 65.5% of the connector pixels in validation, so a pooled
gain can come from one assembly on its own. All three should move (032 +0.035, 034
+0.046, 035 +0.038).

`--tta` adds horizontal-flip test-time augmentation, giving 0.8080 / 0.8812 / 0.8957.
Report it separately from the plain number.

### 6. Rebuild the comparison figure

```bash
python src/build_august_deck_figure.py
```

Writes `results/presentations/KIAT_CREFLE_UPDATE_AUGUST2026/slide_compare.png`, using the
validation frames where the two models disagree most on the connector class.

## Layout

```
src/                 rendering, dataset building, training, evaluation, analysis
  pcl_to_rgbd.py             point cloud to RGB-D rasteriser (splatting, views, depth)
  texture_mapping.py         per-point textures, background scene composition
  convert_to_video_dataset.py / render_full_dataset.py
                             render driver, per-set and multi-worker
  prepare_dformer_data.py    render layout to flat training dataset
  reencode_labels_3way.py    6-class to 3-class labels
  train_rgb_only_sota.py     SegFormer trainer, binary and three-way
  eval_3way_final.py         three-way evaluation, per-set breakdown, TTA
  diag_3way_connector_ceiling.py
                             error decomposition: confusion, rim bands, blob size vs recall
scripts/             asset download and training launchers
sbatch/              SLURM job scripts
assets/              small CC0 inputs that cannot be re-downloaded
dataset_generation/  point-cloud generation utilities
dataloading/         dataset loaders
data/                inputs and rendered datasets, gitignored
results/             checkpoints, metrics, figures, gitignored
```

## Things worth knowing before changing anything

Pooled metrics on this data are dominated by one assembly, so check every set.

Checkpoint metrics are noisy. A single good epoch has been read as a real gain here
before, so the numbers above are the mean of the last five evaluations.
`src/compare_3way_arms.py` does that comparison.

Every result in the table came from a matched run where only the lever under test
differed: same init, seed, data and schedule.

Dataset changes are where this project has been burnt. Several render changes that
improved the synthetic score made real-camera performance much worse. A synthetic gain
says nothing about real footage on its own.

All the numbers here are synthetic validation. Real-camera performance is measured
separately and these figures say nothing about it.
