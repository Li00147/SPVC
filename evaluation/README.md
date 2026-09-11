# SPVC Evaluation

This directory provides two batch-inference utilities and two independent
metric entry points:

- `eval_novel_views.py` generates NVS predictions using control-video,
  reference-video, structured-video, and camera-pose conditions;
- `eval_edit_scene.py` generates PanopticFix predictions;
- `evaluate.sh` evaluates NVS predictions;
- `evaluate_panoptic_fix.sh` evaluates `SPVC-PanopticFix-Benchmark` with its
  edit-aware pairing rules and adds FID-A.

The metric set includes:

- **CLIP-F**: frame-level semantic similarity between aligned generated and GT
  frames, averaged first within each group and then across groups;
- **CLIP-V**: temporal consistency between adjacent generated-frame CLIP
  embeddings, using the same group averaging;
- **FID**: Fréchet Inception Distance between generated and GT image sets;
- **FVD**: Fréchet Video Distance using the bundled VideoGPT and StyleGAN-V I3D
  implementations;
- **FID-A**: FID over YOLO-detected vehicle crops (PanopticFix only).

## Directory structure

```text
evaluation/
├── eval_novel_views.py                 # NVS batch inference
├── eval_edit_scene.py                  # PanopticFix batch inference
├── evaluate.sh                         # NVS metric entry point
├── evaluate_panoptic_fix.sh            # PanopticFix entry point
├── README.md
├── requirements.txt                    # metric-specific dependencies
├── clip_metrics/
│   └── calculate_clip_metrics.py       # NVS CLIP-F and CLIP-V
├── panoptic_fix/
│   ├── calculate_clip_metrics.py       # edit-aware CLIP-F and CLIP-V
│   └── crop_fid_a.py                   # YOLO vehicle crops for FID-A
└── fvd/
    ├── calculate_fvd.py                # FVD command-line entry point
    ├── README.md                        # local FVD layout and usage
    ├── README-upstream.md               # upstream implementation notes
    └── fvd/
        ├── styleganv/
        │   ├── fvd.py
        │   └── i3d_torchscript.pt
        └── videogpt/
            ├── fvd.py
            ├── pytorch_i3d.py
            └── i3d_pretrained_400.pt
```

FID is provided by the external `pytorch-fid` package, so it does not require a
local implementation directory.

## Installation

Install SPVC first, including a PyTorch/torchvision build compatible with the
machine's CUDA version. Then install the evaluation dependencies from the SPVC
repository root:

```bash
python -m pip install -r evaluation/requirements.txt
```

For either inference utility, point DiffSynth to the model base directory that
contains `PAI/Wan2.2-Fun-A14B-Control`:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/DiffSynth-Studio/models
```

The two FVD I3D checkpoints are already bundled under `evaluation/fvd/fvd/`.
OpenAI CLIP downloads `ViT-B/32` on first use unless `--clip-model` points to a
local checkpoint. On an offline machine, always pass the local CLIP checkpoint
to the NVS evaluator. The PanopticFix evaluator automatically uses
`ViT-B-32.pt` and `yolo11x.pt` from the repository root when those files are
present; the paths can also be supplied explicitly with `--clip-model` and
`--yolo-model`.

The evaluation code was prepared with Python 3.11, PyTorch 2.10.0, torchvision
0.25.0, and the package versions listed in `evaluation/requirements.txt`.

## NVS inference and evaluation

### Expected benchmark layout

The preferred benchmark root is `spvc/SPVC-Benchmark`:

```text
SPVC-Benchmark/
└── <dataset>/
    ├── testset/<scene>_<camera>_<direction>_<scale>_<clip>_.mp4
    ├── structured-condition/combined_<scene4>_<start3>_0_<direction>_<scale>.mp4
    ├── camera-pose-condition/<control-video-stem>.pt
    ├── gt/
    │   ├── gt_images/
    │   └── gt_videos/
    └── results/
        ├── images/<scale>/
        └── videos/<scale>/
```

Use one generated video for each GT video and one generated frame for each GT
frame. Files must sort in corresponding temporal order. Images must be directly
inside their image directories because `pytorch-fid` does not recursively scan
subdirectories.

The NVS evaluator automatically detects the legacy misspelled directory name
`SPVC-Benchamrk` when `SPVC-Benchmark` is absent. All paths can be overridden
explicitly.

### Generate NVS predictions

`eval_novel_views.py` uses the following four inputs for every selected test
clip:

```text
control video:        testset/<original-stem>.mp4
reference video:      gt/gt_videos/<scene>_<clip>.mp4
structured condition: structured-condition/combined_<scene4>_<clip*25>_0_<direction>_<scale>.mp4
camera pose:          camera-pose-condition/<original-stem>.pt
```

Here, `<scene4>` is the zero-padded four-digit scene ID and `<clip*25>` is the
zero-padded start frame. Camera-pose tensors must contain `--num-frames` finite
4 x 4 matrices (25 by default). The script validates every pairing before
loading the diffusion model and prints the four resolved condition paths for
each generated clip.

From the SPVC repository root, validate all condition pairings without loading
the model:

```bash
python evaluation/eval_novel_views.py --scenes all --dry-run
```

Generate all predictions, or select comma-separated scene IDs:

```bash
python evaluation/eval_novel_views.py --scenes all
python evaluation/eval_novel_views.py --scenes 001,062,365
```

By default, the script reads
`SPVC-Benchamrk/nuscenes/{testset,gt,structured-condition,camera-pose-condition}`
and writes `SPVC-Benchamrk/nuscenes/results`. Use `--benchmark-root` to select a
different dataset root. `--camera-pose-encoder-ckpt` defaults to the high-noise
checkpoint because that checkpoint contains the `relpose_ecam` weights.

### Run all NVS metrics

Run the evaluator from the SPVC repository root:

```bash
bash evaluation/evaluate.sh --dataset nuscenes --scale 4.0
```

This single command computes CLIP-F, CLIP-V, FID, VideoGPT FVD, and StyleGAN-V
FVD. The same interface works for other datasets:

```bash
bash evaluation/evaluate.sh --dataset waymo --scale 4.0
bash evaluation/evaluate.sh --dataset pandaset --scale 4.0
```

When invoking `evaluate.sh` from inside `evaluation/`, omit the `evaluation/`
prefix.

On a machine without Internet access, point `--clip-model` to the local OpenAI
CLIP checkpoint:

```bash
bash evaluation/evaluate.sh \
  --dataset nuscenes \
  --scale 4.0 \
  --device cuda:0 \
  --clip-model /path/to/ViT-B-32.pt
```

### Use a custom NVS benchmark location

Override the complete benchmark root:

```bash
bash evaluation/evaluate.sh \
  --dataset nuscenes \
  --scale 4.0 \
  --benchmark-root /path/to/SPVC-Benchmark
```

Or override the result and GT roots independently:

```bash
bash evaluation/evaluate.sh \
  --dataset custom \
  --scale 4.0 \
  --results-root /path/to/results \
  --gt-root /path/to/gt
```

For nonstandard layouts, pass the four leaf directories directly:

```bash
bash evaluation/evaluate.sh \
  --dataset custom \
  --generated-images-root /path/to/generated/images \
  --gt-images-root /path/to/reference/images \
  --generated-videos-root /path/to/generated/videos \
  --gt-videos-root /path/to/reference/videos
```

`--dataset` remains required as the name printed in the evaluation summary even
when all directories are overridden.

### Select NVS metrics and devices

All metrics run by default. Disable individual metric families as needed:

```bash
# CLIP-F, CLIP-V, and FID only
bash evaluation/evaluate.sh --dataset nuscenes --skip-fvd

# FVD only
bash evaluation/evaluate.sh --dataset nuscenes --skip-clip --skip-fid

# CLIP metrics only
bash evaluation/evaluate.sh --dataset nuscenes --skip-fid --skip-fvd
```

Choose a device and metric-specific settings:

```bash
bash evaluation/evaluate.sh \
  --dataset nuscenes \
  --device cuda:0 \
  --clip-model ViT-B/32 \
  --clip-batch-size 64 \
  --clip-grouping prefix \
  --fvd-methods videogpt,styleganv \
  --fvd-data-num 56 \
  --fvd-max-frames 40
```

`--clip-grouping prefix` groups frames by the filename component before the
first underscore and reports an unweighted mean over groups. Use
`--clip-grouping all` when the whole image directory should be treated as one
sequence. CLIP-F aligns generated and GT frames in sorted filename order;
CLIP-V compares consecutive generated frames.

Use `bash evaluation/evaluate.sh --help` for the complete option list.

## PanopticFix evaluation

PanopticFix uses a separate entry point because its generated frames contain
multiple edit variants per scene and its result directories do not have a
dataset/scale level:

```text
SPVC-PanopticFix-Benchmark/
├── gt/
│   ├── images/<scene>_<frame>_0.png
│   └── videos/<scene>_rep<id>.mp4
└── results/
    ├── images/<scene>_<edit>_frame_<frame>.png
    └── videos/<scene>_<edit>.mp4
```

Run CLIP-F, CLIP-V, FID, both FVD implementations, and FID-A:

```bash
bash evaluation/evaluate_panoptic_fix.sh --device cuda:0
```

Validate the paths and report file counts without loading any metric model:

```bash
bash evaluation/evaluate_panoptic_fix.sh --check-only
```

The PanopticFix CLIP implementation follows the edit-scene pairing rule: each
generated `(scene, edit)` clip is aligned with that scene's GT frames; edit
scores are averaged within each scene, followed by an unweighted average across
scenes. CLIP-V is also computed separately inside every edit clip, so no
cross-edit frame pair is included.

FID-A detects `car`, `truck`, `bus`, and `motorbike` instances with YOLO,
resizes every crop to 224 x 224, and computes FID between the generated and GT
vehicle crops. If `metrics/fid-a/edit-scene-gt` exists in the parent repository,
the script uses those reference crops. Otherwise it crops the benchmark GT
frames into a temporary directory. Temporary generated crops are removed when
evaluation exits.

Override local CLIP/YOLO checkpoints or precomputed FID-A GT crops when needed:

```bash
bash evaluation/evaluate_panoptic_fix.sh \
  --clip-model /path/to/ViT-B-32.pt \
  --yolo-model /path/to/yolo11x.pt \
  --fid-a-gt-crops /path/to/edit-scene-gt \
  --device cuda:0
```

Metrics can be disabled independently with `--skip-clip`, `--skip-fid`,
`--skip-fvd`, and `--skip-fid-a`. Use
`bash evaluation/evaluate_panoptic_fix.sh --help` for every option.

## Generate PanopticFix predictions

`eval_edit_scene.py` is a batch inference utility rather than a metric
implementation. It directly reads the 48 MP4 files under
`SPVC-PanopticFix-Benchmark/testset/`, applies SPVC, and writes generated videos
and frames under `SPVC-PanopticFix-Benchmark/results/`:

```bash
python evaluation/eval_edit_scene.py
```

The default inputs and checkpoints are:

```text
SPVC-PanopticFix-Benchmark/
├── testset/<scene>_<edit>.mp4
└── gt/videos/<scene>_rep<id>.mp4
checkpoints/
├── spvc_high_noise.safetensors
└── spvc_low_noise.safetensors
```

Run selected scenes with either zero-padded or integer IDs:

```bash
python evaluation/eval_edit_scene.py --scenes 002,062
python evaluation/eval_edit_scene.py --scenes 2,62
```

The script also accepts the legacy input representation in which each clip is a
subdirectory containing one camera stream named `000_000.png`, `001_000.png`,
and so on. Use `--input-dir`, `--gt-videos-dir`, `--output-root`, and the two
checkpoint options to override the bundled paths. Use `--disable-reference`
for an ablation without reference-video conditioning.

For compatibility with the original inference script, every edit clip from a
scene uses the first sorted `<scene>_*.mp4` file as its reference video. This
policy is printed at startup and should be changed to an explicit pairing
manifest if a different edit-to-reference correspondence is required.

## Extract frames from generated videos

FID and CLIP metrics require image frames. If a method outputs only MP4 files,
extract frames while preserving each video stem:

```bash
DATASET=nuscenes
SCALE=4.0
mkdir -p "SPVC-Benchmark/${DATASET}/results/images/${SCALE}"
for video in "SPVC-Benchmark/${DATASET}/results/videos/${SCALE}"/*.mp4; do
    stem="$(basename "${video%.mp4}")"
    ffmpeg -i "${video}" -start_number 0 \
        "SPVC-Benchmark/${DATASET}/results/images/${SCALE}/${stem}_frame_%04d.png"
done
```

## Input checks and interpretation

For NVS, verify that generated and GT image counts match and that generated and
GT video counts match. NVS CLIP prints a warning and compares only the shared
sorted prefix when group sizes differ. PanopticFix intentionally has more
generated frames than GT frames because every scene has multiple edit variants;
its edit-aware CLIP implementation handles that layout explicitly. FVD loads
sorted MP4 files and requires the generated and GT tensors to have identical
shapes after decoding.

Lower FID and FVD values are better. Higher CLIP-F and CLIP-V values are better.
FVD values from the two bundled implementations should be reported with their
method names because they use different I3D implementations and are not directly
interchangeable.

## Test-set sources

The released nuScenes test set is available from the project-provided
[Google Drive folder](https://drive.google.com/drive/folders/1Pp_wnHtYVLn_qu_70Qc66QBfzUVdyRPL?usp=drive_link).
The Waymo and PandaSet evaluation follows the test sets used by
[ReconDreamer++](https://arxiv.org/html/2503.18438v1).

For PanopticFix, use the 48 videos under `SPVC-PanopticFix-Benchmark/testset/`
and the corresponding references under `SPVC-PanopticFix-Benchmark/gt/`. Keep
the input filename stems for generated outputs so frames and videos remain
aligned after sorting.
