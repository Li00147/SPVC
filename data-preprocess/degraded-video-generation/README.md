# SPVC Degraded Video Pair Creation Guide

This directory contains one entry-point script:

```text
data-preprocess/degraded-video-generation/
├── README.md
└── render_degradations.py
```

The script loads DriveStudio scene checkpoints, renders degraded RGB frames, and
uses DriveStudio's `save_videos` to save videos and individual frames. Pairing a
degraded output with a clean reference output from the same scene, the same set
of cameras, and the same time range produces the degraded/reference video pairs
required by SPVC.

## DriveStudio and OmniRe

DriveStudio, which is used in this workflow, is a 3D Gaussian Splatting codebase
for urban scene reconstruction and simulation. Its official OmniRe
implementation reconstructs dynamic urban scenes from driving logs and supports
multi-camera rendering. Degraded videos are produced by different training
states of these scene models or by different camera visibility settings, rather
than by an additional video generation model.

- DriveStudio repository: <https://github.com/ziyc/drivestudio>
- OmniRe project page: <https://ziyc.github.io/omnire/>
- OmniRe paper: <https://arxiv.org/abs/2408.16760>

Citation:

```bibtex
@article{chen2024omnire,
    title={OmniRe: Omni Urban Scene Reconstruction},
    author={Chen, Ziyu and Yang, Jiawei and Huang, Jiahui and Lutio, Riccardo de
        and Esturo, Janick Martinez and Ivanovic, Boris and Litany, Or
        and Gojcic, Zan and Fidler, Sanja and Pavone, Marco and Song, Li
        and Wang, Yue},
    journal={arXiv preprint arXiv:2408.16760},
    year={2024}
}
```

## Environment and Inputs

`render_degradations.py` dynamically imports the complete DriveStudio package.
Therefore, this directory does not include the DriveStudio source code, CUDA
extensions, datasets, or checkpoints. Run the script in an environment where
DriveStudio is already working:

```bash
cd /path/to/drivestudio
export PYTHONPATH=$(pwd)
python -c "from datasets.driving_dataset import DrivingDataset; from models.video_utils import render_images, save_videos"
```

Requirements:

1. `--drivestudio-root` must contain `datasets/driving_dataset.py`,
   `models/video_utils.py`, and the other DriveStudio dependencies;
2. `--checkpoint-root/<scene>/` must contain `config.yaml` and the corresponding
   checkpoint;
3. The data paths in `config.yaml` must point to preprocessed DriveStudio data;
4. To produce per-frame directories, set `logging.save_seperate_video` to `true`
   in the configuration.

The standard checkpoint root has the following structure:

```text
checkpoint-root/
└── <scene>/
    ├── config.yaml
    ├── checkpoint_01000.pth     # Underfitted model
    └── checkpoint_final.pth     # Standard 40,000-step model
```

Rendering cannot be run directly with only raw images or only a DriveStudio data
directory. Each scene must have the `config.yaml` saved alongside its checkpoint.

## Training the Three Types of Degradation Checkpoints

The following conventions describe the training strategies used in this project
to create degraded data. Training still uses DriveStudio's `tools/train.py` and
the OmniRe configuration; this directory only handles the subsequent rendering.

### 1. Underfitting: Train for 1,000 Steps

Use the standard OmniRe configuration and the full set of training cameras, but
set `trainer.optim.num_iters` to `1000`. After training, render with
`checkpoint_01000.pth`. This checkpoint retains an obvious underfitted state and
is used to create the underfitting degradation:

```bash
cd /path/to/drivestudio
export PYTHONPATH=$(pwd)
python tools/train.py \
  --config_file configs/omnire.yaml \
  --output_root /path/to/checkpoints \
  --project spvc_degradation \
  --run_name <dataset>_<scene>_underfitting \
  dataset=<dataset>/<num_cams>cams \
  data.scene_idx=<scene> \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  trainer.optim.num_iters=1000
```

Make sure the checkpoint-saving interval permits a checkpoint to be saved at
step 1,000. The standard configuration usually produces
`checkpoint_01000.pth`.

### 2. Cross-view: Train on Three Cameras and Render the Other Three

For a six-camera dataset, first train OmniRe using only three selected cameras,
then use the trained scene model to render the other three cameras that were not
used for training. For example, the training cameras can be `[0, 1, 2]` and the
rendering cameras `[3, 4, 5]`. Use a three-camera data configuration during
training:

```bash
python tools/train.py \
  --config_file configs/omnire.yaml \
  --output_root /path/to/checkpoints \
  --project spvc_degradation \
  --run_name <dataset>_<scene>_cross_view \
  dataset=<dataset>/3cams \
  data.scene_idx=<scene> \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  trainer.optim.num_iters=40000
```

Use `--cameras` to specify cameras that were not used for training. The following
example corresponds to training on `[0, 1, 2]` and rendering `[3, 4, 5]`; the
actual camera indices must match the dataset configuration:

```bash
python data-preprocess/degraded-video-generation/render_degradations.py \
  --mode cross-reference \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --cameras 3,4,5
```

The script's `cross-reference` mode is the cross-view rendering mode described
here. If `--cameras` is omitted, the script uses `[0, 3, 4]` by default. To
reproduce the setup that trains on three cameras and renders the other three,
explicitly pass the list of cameras excluded from training.

### 3. Random Mask: Add Random Occlusions to a Standard Checkpoint

Random mask does not retrain the model. First train for `40000` steps with the
standard OmniRe configuration and render using the resulting
`checkpoint_final.pth`. Then apply random rectangular, elliptical, or polygonal
occlusions to each saved RGB frame. A fixed `--seed` reproduces the same set of
occlusions:

```bash
python data-preprocess/degraded-video-generation/render_degradations.py \
  --mode random-mask \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --seed 0
```

This mode looks for `checkpoint_final.pth` by default. Random occlusions are
applied to the images already saved in `<scene>_rgbs/`. To obtain an occluded
MP4, re-encode these occluded PNGs; do not use the MP4 written before the
occlusions were applied.

## Rendering Degraded Videos and Images

### Underfitting

```bash
python data-preprocess/degraded-video-generation/render_degradations.py \
  --mode underfitting \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/underfitting_checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset>
```

By default, this mode reads `checkpoint_01000.pth` for each scene.

### Cross-view

```bash
python data-preprocess/degraded-video-generation/render_degradations.py \
  --mode cross-reference \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/cross_view_checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --cameras 3,4,5
```

### Random Mask

```bash
python data-preprocess/degraded-video-generation/render_degradations.py \
  --mode random-mask \
  --drivestudio-root /path/to/drivestudio \
  --checkpoint-root /path/to/normal_40000_checkpoints/<dataset> \
  --output-root /path/to/degraded_outputs \
  --dataset-name <dataset> \
  --seed 0
```

Use `--scenes 001 002` to process only specified scenes, or use `--checkpoint` to
specify a checkpoint with a non-default filename. `--device auto` uses a GPU when
CUDA is available.

The output directory has the following structure:

```text
degraded_outputs/
├── underfitting_data/<dataset>/<scene>_rgbs.mp4
├── underfitting_data/<dataset>/<scene>_rgbs/000_000.png
├── cross_reference_data/<dataset>/<scene>_rgbs.mp4
├── cross_reference_data/<dataset>/<scene>_rgbs/000_000.png
└── random_mask_data/<dataset>/<scene>_rgbs/000_000.png
```

When `logging.save_seperate_video=true`, `<scene>_rgbs/` is a directory of PNGs
saved for each timestamp and camera, with filenames in the form
`<timestamp>_<camera>.png`. If separate video saving is disabled in the
configuration, the output may contain only a tiled video and no frame directory
for random-mask processing.

## Creating Degraded–Reference Video Pairs

For each scene, keep all four of the following identical:

- Scene ID;
- Camera list and camera order;
- Start and end timestamps, and frame count;
- FPS, resolution, and video layout.

Use the degraded rendering result as the `control/degraded video`, and use either
the original sensor video or the standard 40,000-step OmniRe rendering result as
the `reference/GT video`. Pairing files with identical names is recommended:

```text
degraded/<mode>/<dataset>/<scene>_rgbs.mp4
reference/<dataset>/<scene>_rgbs.mp4
```

If different subsets of the six cameras are used for training or rendering,
first verify that the cameras and timelines in the two videos still correspond
one-to-one. The cross-view cameras excluded from training must match the same
cameras in the reference video.

The pairing workflow for random mask is:

1. Render with `checkpoint_final.pth` and save individual PNG frames;
2. Apply the script's built-in random occlusions to the PNGs in
   `<scene>_rgbs/`;
3. Encode the occluded PNGs as the control video;
4. Use a clean video with the same scene, cameras, and time window as the
   reference video.

For example, re-encode the occluded frames from a single camera as follows:

```bash
ffmpeg -framerate 12 \
  -i /path/to/random_mask_data/<dataset>/<scene>_rgbs/%03d_0.png \
  -c:v libx264 -pix_fmt yuv420p \
  /path/to/pairs/random_mask/<scene>_camera0.mp4
```

To preserve a tiled multi-camera layout, tile the camera PNGs for each timestamp
according to DriveStudio's `layout`, and then encode the resulting frames.

## Parameter Quick Reference

```text
--mode              cross-reference | underfitting | random-mask
--drivestudio-root  Root directory of the DriveStudio repository
--checkpoint-root   Checkpoint root directly containing each scene subdirectory
--output-root       Root directory for degraded video and image outputs
--dataset-name      nuscenes | pandaset | waymo
--checkpoint        Optional; overrides the default checkpoint filename
--scenes            Optional; processes only the specified scenes
--cameras           Optional; overrides the rendering camera list
--device            auto | cuda | cpu
--fps               Output FPS; default: 12
--seed              Random seed for random-mask; default: 0
```

This directory does not include checkpoints, raw data, or DriveStudio
dependencies. For release or reproduction, prepare these external resources in
accordance with the licenses of the relevant datasets and DriveStudio, and cite
OmniRe.
