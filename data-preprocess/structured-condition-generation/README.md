# nuScenes Structured-Condition Generation

This directory generates the structured video condition used by SPVC as
`reference_combined_video`. Each output frame contains projected HD-map layers
and vehicle 3D bounding boxes. The RGB camera background is removed from the
final structured condition.

## Contents

```text
structured-condition-generation/
├── README.md
├── requirements.txt
├── proj_combined.py   # Generate 25-frame structured-condition videos
├── proj_map.py        # Render one map-projection diagnostic image
└── proj_bbox.py       # Render one vehicle-box diagnostic image
```

`proj_combined.py` is the dataset-generation entry point. The other two scripts
are optional diagnostics for checking nuScenes calibration, map projection, and
3D-box projection before a full run.

## Requirements

Install the Python dependencies from the SPVC repository root:

```bash
python -m pip install -r \
  data-preprocess/structured-condition-generation/requirements.txt
```

The generator also requires an `ffmpeg` executable. It is discovered from
`PATH` by default, or it can be selected explicitly with `--ffmpeg`.

The scripts use the official, unmodified `nuscenes-devkit`; no changes to
`site-packages` are required.

## nuScenes data

Pass the nuScenes data root explicitly. It must contain the selected metadata
version together with the samples, sweeps, and maps required by the devkit. The
12 Hz annotations used for SPVC are selected with:

```text
--version advanced_12Hz_trainval
```

For example:

```text
/path/to/nuscenes/
├── advanced_12Hz_trainval/
├── maps/
├── samples/
└── sweeps/
```

The custom annotation version is not distributed in this source directory.

## Generate structured-condition videos

From the SPVC repository root, first inspect the planned output for one scene,
one camera, and one clip:

```bash
python data-preprocess/structured-condition-generation/proj_combined.py \
  --data-root /path/to/nuscenes \
  --version advanced_12Hz_trainval \
  --output-dir /path/to/structured_conditions \
  --start-sequence 0 \
  --end-sequence 1 \
  --camera-indices 0 \
  --max-clips-per-sequence 1 \
  --dry-run
```

Remove `--dry-run` to render the clip. To process all scenes and all six
cameras:

```bash
python data-preprocess/structured-condition-generation/proj_combined.py \
  --data-root /path/to/nuscenes \
  --version advanced_12Hz_trainval \
  --output-dir /path/to/structured_conditions
```

Existing complete MP4 files are skipped, so the same command can resume an
interrupted run. Add `--overwrite` to regenerate them.

## Output naming and clip construction

The output scene ID is the trailing numeric component of the nuScenes scene
name: for example, `scene-0001` becomes `0001`. Every consecutive 25-frame
window produces one video; an incomplete window at the end of a scene is
discarded. Output names use zero-padded scene and start-frame IDs:

```text
combined_<scene:04d>_<start-frame:03d>_<camera>.mp4
```

Examples:

```text
combined_0001_000_0.mp4
combined_0001_025_0.mp4
combined_0012_100_3.mp4
```

The default output is H.264 video at 12 FPS, CRF 18, with 25 frames per clip.
Frame dimensions follow the figure returned by the nuScenes map renderer.

NVS benchmark conditions may additionally encode a requested view direction
and offset in the filename:

```text
combined_0001_000_0_right_4.0.mp4
```

Those suffixes identify benchmark conditions; `proj_combined.py` produces the
base structured render before benchmark-specific selection or renaming.

## Camera indices

The camera indices follow this fixed order:

```text
0 = CAM_FRONT
1 = CAM_FRONT_RIGHT
2 = CAM_BACK_RIGHT
3 = CAM_BACK
4 = CAM_BACK_LEFT
5 = CAM_FRONT_LEFT
```

Select one or more cameras with `--camera-indices`, for example:

```text
--camera-indices 0 3
```

## Rendered content

The structured condition includes these map layers:

```text
road_segment
ped_crossing
```

Only `vehicle.*` boxes are retained. Categories beginning with `animal`,
`human.`, `movable_object.`, or `static_object.` are excluded. The camera image
artist is removed after map projection, leaving only the map and 3D boxes.

## Parallel rendering

Independent processes may render different cameras or disjoint scene ranges.
Do not let multiple processes write the same scene-camera outputs.

The following example launches one process per camera:

```bash
DATA_ROOT=/path/to/nuscenes
OUTPUT_DIR=/path/to/structured_conditions
LOG_DIR=/path/to/structured_condition_logs
mkdir -p "${LOG_DIR}"

for camera in 0 1 2 3 4 5; do
  nohup python \
    data-preprocess/structured-condition-generation/proj_combined.py \
    --data-root "${DATA_ROOT}" \
    --version advanced_12Hz_trainval \
    --output-dir "${OUTPUT_DIR}" \
    --camera-indices "${camera}" \
    > "${LOG_DIR}/camera_${camera}.log" 2>&1 < /dev/null &
  echo "camera=${camera} pid=$!"
done
```

Use `--start-sequence` and `--end-sequence` to assign non-overlapping scene
ranges. `--end-sequence` is exclusive.

## Diagnostic renders

Render a single map overlay:

```bash
python data-preprocess/structured-condition-generation/proj_map.py \
  --data-root /path/to/nuscenes \
  --version advanced_12Hz_trainval \
  --sample-index 0 \
  --camera CAM_FRONT \
  --output /path/to/map_projection.png
```

Render vehicle boxes for the first frame of a scene:

```bash
python data-preprocess/structured-condition-generation/proj_bbox.py \
  --data-root /path/to/nuscenes \
  --version advanced_12Hz_trainval \
  --scene-index 0 \
  --camera CAM_FRONT \
  --output /path/to/vehicle_boxes.png
```

These diagnostic images retain the RGB background and are not direct model
inputs.

## Main generator options

```text
--data-root PATH              nuScenes dataset root (required)
--version NAME                Metadata version (default: advanced_12Hz_trainval)
--output-dir PATH             Output video directory (required)
--start-sequence N            Inclusive zero-based scene index (default: 0)
--end-sequence N              Exclusive zero-based scene index (default: all)
--camera-indices ...          One or more camera indices (default: all six)
--fps N                       Output FPS (default: 12)
--ffmpeg PATH                 ffmpeg executable (default: discovered from PATH)
--overwrite                   Replace existing complete MP4 files
--dry-run                     Print planned outputs without rendering
--max-clips-per-sequence N    Optional per-scene clip limit for testing
```

An in-progress clip is written with a `.part.mp4` suffix and atomically renamed
after `ffmpeg` succeeds. A failed or interrupted run may leave no completed file
for that clip; rerunning the same command safely regenerates it.
