# SPVC Benchmark Evaluation

This directory provides a shared evaluation workflow for the SPVC benchmark.
The same scripts support nuScenes, Waymo, PandaSet, and additional datasets that
follow the directory convention below.
Follow this guide to reproduce the quantitative results reported in our paper
for nuScenes, Waymo, and PandaSet.

## Test sets

We provide the [nuScenes test set](https://drive.google.com/drive/folders/1Pp_wnHtYVLn_qu_70Qc66QBfzUVdyRPL?usp=drive_link) with this benchmark release. For Waymo and
PandaSet, our evaluation uses the same test sets as
[ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving
Scene Representation](https://arxiv.org/html/2503.18438v1). Place these test
sets under `SPVC-Benchamrk/waymo/testset/` and
`SPVC-Benchamrk/pandaset/testset/`, respectively.

## Reproduce PanopticFix quantitative results

To reproduce the PanopticFix quantitative results reported in our paper,
download the [PanopticFix benchmark](https://drive.google.com/drive/folders/1TDutdH9oV3MrcSUOHsf84uDTv4aGsjDo?usp=drive_link)
and use the 48 test videos under `SPVC-PanopticFix-Benchmark/testset/`. Each MP4
is created from one scene-edit sequence and contains 25 frames at 12 FPS.
The corresponding reference images and videos are provided under
`SPVC-PanopticFix-Benchmark/gt/images/` and
`SPVC-PanopticFix-Benchmark/gt/videos/`. Run the PanopticFix setting on every
test video, preserve the input filename stem for the generated output, and
compute the paper's quantitative metrics using the evaluation workflow below.

## Benchmark layout

```text
spvc/
├── SPVC-Benchamrk/
│   ├── nuscenes/
│   │   ├── testset/
│   │   ├── gt/
│   │   │   ├── gt_images/
│   │   │   ├── gt_images_split/     # optional scene-grouped copy
│   │   │   └── gt_videos/
│   │   └── results/
│   │       ├── images/<scale>/
│   │       └── videos/<scale>/
│   ├── waymo/
│   │   ├── testset/
│   │   ├── gt/{gt_images,gt_videos}/
│   │   └── results/{images,videos}/
│   └── pandaset/
│       ├── testset/
│       ├── gt/{gt_images,gt_videos}/
│       └── results/{images,videos}/
├── SPVC-PanopticFix-Benchmark/
│   ├── testset/                    # 48 scene-edit test videos
│   └── gt/
│       ├── images/                 # reference frames
│       └── videos/                 # reference videos
└── evaluation/
    ├── README.md
    ├── evaluate.sh                  # shared FID/FVD entry point
    ├── cal_clip_metrics.py          # shared CLIP-F/CLIP-V entry point
    └── fvd/                         # FVD code and I3D checkpoints
```

Each dataset is self-contained. Put released inputs under `testset/`, reference
data under `gt/`, and method outputs under `results/`. The evaluator never reads
another dataset's directory.

## Result format

For a dataset name `<dataset>` and scale `<scale>`, place outputs in:

```text
SPVC-Benchamrk/<dataset>/results/videos/<scale>/
SPVC-Benchamrk/<dataset>/results/images/<scale>/
```

Use one generated video for each GT video and one generated frame for each GT
frame. File names should sort in corresponding temporal order. By default,
CLIP metrics group frames using the filename prefix before the first underscore;
use a stable scene or sequence ID in that position.

For example:

```text
SPVC-Benchamrk/nuscenes/results/videos/4.0/001_001_right_4.0_000_.mp4
SPVC-Benchamrk/nuscenes/results/images/4.0/001_001_right_4.0_000__frame_0000.png
```

If a method produces videos only, extract all frames with FFmpeg. Run this from
the `spvc/` directory after setting the dataset and scale:

```bash
DATASET=nuscenes
SCALE=4.0
mkdir -p "SPVC-Benchamrk/${DATASET}/results/images/${SCALE}"
for video in "SPVC-Benchamrk/${DATASET}/results/videos/${SCALE}"/*.mp4; do
    stem="$(basename "${video%.mp4}")"
    ffmpeg -i "$video" -start_number 0 \
        "SPVC-Benchamrk/${DATASET}/results/images/${SCALE}/${stem}_frame_%04d.png"
done
```

## Environment

A CUDA-capable environment is recommended. Install a PyTorch/torchvision build
compatible with the local CUDA version first, followed by the metric packages:

```bash
python -m pip install numpy==1.26.4 scipy==1.17.1 Pillow==12.1.0 \
    tqdm==4.67.2 einops av==16.1.0 pytorch-fid==0.3.0
python -m pip install git+https://github.com/openai/CLIP.git
```

The reference environment used PyTorch 2.10.0 and torchvision 0.25.0. FVD I3D
checkpoints are bundled under `evaluation/fvd/`. CLIP downloads the official
`ViT-B/32` checkpoint on first use unless `--clip-model` points to a local file.

## Run FID and FVD

From `spvc/evaluation/`, select any dataset folder and scale:

```bash
bash evaluate.sh --dataset nuscenes --scale 4.0
bash evaluate.sh --dataset waymo --scale 4.0
bash evaluate.sh --dataset pandaset --scale 4.0
```

By default, FID uses all top-level images and FVD uses all sorted MP4 files with
at most 40 frames per video. Limit the number of FVD videos when reproducing a
configuration that used a fixed subset:

```bash
bash evaluate.sh \
    --dataset nuscenes \
    --scale 4.0 \
    --fvd-data-num 56 \
    --fvd-max-frames 40
```

Run only one metric family when needed:

```bash
bash evaluate.sh --dataset nuscenes --scale 4.0 --skip-fvd
bash evaluate.sh --dataset nuscenes --scale 4.0 --skip-fid
```

For a dataset stored outside the standard tree, override its roots:

```bash
bash evaluate.sh \
    --dataset custom_name \
    --scale 4.0 \
    --results-root /path/to/results \
    --gt-root /path/to/gt
```

Use `bash evaluate.sh --help` for all options. The script reports FID, VideoGPT
FVD, and StyleGAN-V FVD.

## Run CLIP-F and CLIP-V

From `spvc/evaluation/`:

```bash
python cal_clip_metrics.py --dataset nuscenes --scales 4.0
python cal_clip_metrics.py --dataset waymo --scales 4.0
python cal_clip_metrics.py --dataset pandaset --scales 4.0
```

Explicit paths and model settings can be supplied when needed:

```bash
python cal_clip_metrics.py \
    --dataset custom_name \
    --gen-images-root /path/to/results/images \
    --gt-images-root /path/to/gt/gt_images \
    --scales 4.0 \
    --clip-model ViT-B/32 \
    --batch-size 64
```

Use `--grouping prefix` (the default) for per-scene/per-sequence averaging or
`--grouping all` to treat the complete image set as one group. CLIP-F compares
aligned generated/GT frame embeddings; CLIP-V compares adjacent generated-frame
embeddings.

## Preflight checks

From `spvc/`, set a dataset and scale, then verify its inputs:

```bash
DATASET=nuscenes
SCALE=4.0
find "SPVC-Benchamrk/${DATASET}/testset" -maxdepth 1 -type f -name '*.mp4' | wc -l
find "SPVC-Benchamrk/${DATASET}/gt/gt_images" -maxdepth 1 -type f | wc -l
find "SPVC-Benchamrk/${DATASET}/gt/gt_videos" -maxdepth 1 -type f -name '*.mp4' | wc -l
find "SPVC-Benchamrk/${DATASET}/results/images/${SCALE}" -maxdepth 1 -type f | wc -l
find "SPVC-Benchamrk/${DATASET}/results/videos/${SCALE}" -maxdepth 1 -type f -name '*.mp4' | wc -l
```

The generated and GT counts should match for each modality. `pytorch-fid` does
not scan subdirectories, so FID must point to the flat `gt/gt_images/` directory.
