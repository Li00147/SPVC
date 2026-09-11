#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPVC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_BENCHMARK_ROOT="${SPVC_ROOT}/SPVC-Benchmark"
if [[ ! -d "${DEFAULT_BENCHMARK_ROOT}" && -d "${SPVC_ROOT}/SPVC-Benchamrk" ]]; then
    DEFAULT_BENCHMARK_ROOT="${SPVC_ROOT}/SPVC-Benchamrk"
fi

DATASET=""
SCALE="4.0"
BENCHMARK_ROOT="${SPVC_BENCHMARK_ROOT:-${DEFAULT_BENCHMARK_ROOT}}"
RESULTS_ROOT=""
GT_ROOT=""
GENERATED_IMAGES_ROOT=""
GT_IMAGES_ROOT=""
GENERATED_VIDEOS_ROOT=""
GT_VIDEOS_ROOT=""

RUN_CLIP=1
RUN_FID=1
RUN_FVD=1

CLIP_MODEL="ViT-B/32"
CLIP_BATCH_SIZE="64"
CLIP_GROUPING="prefix"
FVD_DATA_NUM=""
FVD_MAX_FRAMES="40"
FVD_METHODS="videogpt,styleganv"
DEVICE=""
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluate.sh --dataset DATASET [options]

By default, this command computes CLIP-F, CLIP-V, FID, VideoGPT FVD, and
StyleGAN-V FVD for one dataset and scale.

Required:
  --dataset NAME                 Dataset folder under the benchmark root.

Dataset layout:
  <benchmark>/<dataset>/results/images/<scale>/
  <benchmark>/<dataset>/results/videos/<scale>/
  <benchmark>/<dataset>/gt/gt_images/
  <benchmark>/<dataset>/gt/gt_videos/

Path options:
  --scale VALUE                  Result scale subfolder (default: 4.0).
  --benchmark-root PATH          Benchmark root (default: ../SPVC-Benchmark;
                                 the legacy ../SPVC-Benchamrk is auto-detected).
  --results-root PATH            Override <benchmark>/<dataset>/results.
  --gt-root PATH                 Override <benchmark>/<dataset>/gt.
  --generated-images-root PATH   Override the generated image directory.
  --gt-images-root PATH          Override the GT image directory.
  --generated-videos-root PATH   Override the generated video directory.
  --gt-videos-root PATH          Override the GT video directory.

Metric options:
  --skip-clip                    Skip CLIP-F and CLIP-V.
  --skip-fid                     Skip FID.
  --skip-fvd                     Skip FVD.
  --device DEVICE                Torch device passed to all metric tools.
  --clip-model MODEL             CLIP model name/path (default: ViT-B/32).
  --clip-batch-size N            CLIP encoding batch size (default: 64).
  --clip-grouping MODE           prefix or all (default: prefix).
  --fvd-data-num N               Use the first N sorted videos (default: all).
  --fvd-max-frames N             Maximum frames per video (default: 40).
  --fvd-methods LIST             Comma-separated methods: videogpt,styleganv
                                 (default: both).
  -h, --help                     Show this message.

Environment:
  PYTHON_BIN                     Python executable (default: python).
  SPVC_BENCHMARK_ROOT            Alternative default benchmark root.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "[ERROR] Missing value for ${1}" >&2
        usage >&2
        exit 2
    fi
}

require_directory() {
    if [[ ! -d "${1}" ]]; then
        echo "[ERROR] Directory not found: ${1}" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            require_value "$@"
            DATASET="$2"
            shift 2
            ;;
        --scale)
            require_value "$@"
            SCALE="$2"
            shift 2
            ;;
        --benchmark-root)
            require_value "$@"
            BENCHMARK_ROOT="$2"
            shift 2
            ;;
        --results-root)
            require_value "$@"
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --gt-root)
            require_value "$@"
            GT_ROOT="$2"
            shift 2
            ;;
        --generated-images-root)
            require_value "$@"
            GENERATED_IMAGES_ROOT="$2"
            shift 2
            ;;
        --gt-images-root)
            require_value "$@"
            GT_IMAGES_ROOT="$2"
            shift 2
            ;;
        --generated-videos-root)
            require_value "$@"
            GENERATED_VIDEOS_ROOT="$2"
            shift 2
            ;;
        --gt-videos-root)
            require_value "$@"
            GT_VIDEOS_ROOT="$2"
            shift 2
            ;;
        --device)
            require_value "$@"
            DEVICE="$2"
            shift 2
            ;;
        --clip-model)
            require_value "$@"
            CLIP_MODEL="$2"
            shift 2
            ;;
        --clip-batch-size)
            require_value "$@"
            CLIP_BATCH_SIZE="$2"
            shift 2
            ;;
        --clip-grouping)
            require_value "$@"
            CLIP_GROUPING="$2"
            shift 2
            ;;
        --fvd-data-num)
            require_value "$@"
            FVD_DATA_NUM="$2"
            shift 2
            ;;
        --fvd-max-frames)
            require_value "$@"
            FVD_MAX_FRAMES="$2"
            shift 2
            ;;
        --fvd-methods)
            require_value "$@"
            FVD_METHODS="$2"
            shift 2
            ;;
        --skip-clip)
            RUN_CLIP=0
            shift
            ;;
        --skip-fid)
            RUN_FID=0
            shift
            ;;
        --skip-fvd)
            RUN_FVD=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${DATASET}" ]]; then
    echo "[ERROR] --dataset is required" >&2
    usage >&2
    exit 2
fi
if [[ ! "${DATASET}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] Invalid dataset name: ${DATASET}" >&2
    exit 2
fi
if [[ ! "${SCALE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] Invalid scale name: ${SCALE}" >&2
    exit 2
fi
if [[ "${RUN_CLIP}" -eq 0 && "${RUN_FID}" -eq 0 && "${RUN_FVD}" -eq 0 ]]; then
    echo "[ERROR] CLIP, FID, and FVD are all disabled" >&2
    exit 2
fi
if [[ ! "${CLIP_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --clip-batch-size must be a positive integer" >&2
    exit 2
fi
if [[ "${CLIP_GROUPING}" != "prefix" && "${CLIP_GROUPING}" != "all" ]]; then
    echo "[ERROR] --clip-grouping must be prefix or all" >&2
    exit 2
fi
if [[ -n "${FVD_DATA_NUM}" && ! "${FVD_DATA_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --fvd-data-num must be a positive integer" >&2
    exit 2
fi
if [[ ! "${FVD_MAX_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --fvd-max-frames must be a positive integer" >&2
    exit 2
fi

IFS=',' read -r -a FVD_METHOD_ARRAY <<< "${FVD_METHODS}"
if [[ "${#FVD_METHOD_ARRAY[@]}" -eq 0 ]]; then
    echo "[ERROR] --fvd-methods cannot be empty" >&2
    exit 2
fi
for method in "${FVD_METHOD_ARRAY[@]}"; do
    if [[ "${method}" != "videogpt" && "${method}" != "styleganv" ]]; then
        echo "[ERROR] Unsupported FVD method: ${method}" >&2
        exit 2
    fi
done

DATASET_ROOT="${BENCHMARK_ROOT}/${DATASET}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATASET_ROOT}/results}"
GT_ROOT="${GT_ROOT:-${DATASET_ROOT}/gt}"
GENERATED_IMAGES_ROOT="${GENERATED_IMAGES_ROOT:-${RESULTS_ROOT}/images/${SCALE}}"
GENERATED_VIDEOS_ROOT="${GENERATED_VIDEOS_ROOT:-${RESULTS_ROOT}/videos/${SCALE}}"

if [[ -z "${GT_IMAGES_ROOT}" ]]; then
    GT_IMAGES_ROOT="${GT_ROOT}/gt_images"
    if [[ ! -d "${GT_IMAGES_ROOT}" && -d "${GT_ROOT}/images" ]]; then
        GT_IMAGES_ROOT="${GT_ROOT}/images"
    fi
fi
if [[ -z "${GT_VIDEOS_ROOT}" ]]; then
    GT_VIDEOS_ROOT="${GT_ROOT}/gt_videos"
    if [[ ! -d "${GT_VIDEOS_ROOT}" && -d "${GT_ROOT}/videos" ]]; then
        GT_VIDEOS_ROOT="${GT_ROOT}/videos"
    fi
fi

if [[ "${RUN_CLIP}" -eq 1 || "${RUN_FID}" -eq 1 ]]; then
    require_directory "${GENERATED_IMAGES_ROOT}"
    require_directory "${GT_IMAGES_ROOT}"
fi
if [[ "${RUN_FVD}" -eq 1 ]]; then
    require_directory "${GENERATED_VIDEOS_ROOT}"
    require_directory "${GT_VIDEOS_ROOT}"
fi

echo "========== SPVC Benchmark Evaluation =========="
echo "Dataset:          ${DATASET}"
echo "Scale:            ${SCALE}"
echo "Generated images: ${GENERATED_IMAGES_ROOT}"
echo "GT images:        ${GT_IMAGES_ROOT}"
echo "Generated videos: ${GENERATED_VIDEOS_ROOT}"
echo "GT videos:        ${GT_VIDEOS_ROOT}"

if [[ "${RUN_CLIP}" -eq 1 ]]; then
    clip_command=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/clip_metrics/calculate_clip_metrics.py"
        --generated-images "${GENERATED_IMAGES_ROOT}"
        --gt-images "${GT_IMAGES_ROOT}"
        --clip-model "${CLIP_MODEL}"
        --batch-size "${CLIP_BATCH_SIZE}"
        --grouping "${CLIP_GROUPING}"
    )
    if [[ -n "${DEVICE}" ]]; then
        clip_command+=(--device "${DEVICE}")
    fi
    echo
    echo "[CLIP] Computing CLIP-F and CLIP-V"
    "${clip_command[@]}"
fi

if [[ "${RUN_FID}" -eq 1 ]]; then
    fid_command=(
        "${PYTHON_BIN}"
        -m pytorch_fid
        "${GENERATED_IMAGES_ROOT}"
        "${GT_IMAGES_ROOT}"
    )
    if [[ -n "${DEVICE}" ]]; then
        fid_command+=(--device "${DEVICE}")
    fi
    echo
    echo "[FID] ${GENERATED_IMAGES_ROOT} vs ${GT_IMAGES_ROOT}"
    "${fid_command[@]}"
fi

if [[ "${RUN_FVD}" -eq 1 ]]; then
    fvd_command=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/fvd/calculate_fvd.py"
        --generated-videos "${GENERATED_VIDEOS_ROOT}"
        --gt-videos "${GT_VIDEOS_ROOT}"
        --max-frames "${FVD_MAX_FRAMES}"
        --methods "${FVD_METHOD_ARRAY[@]}"
    )
    if [[ -n "${FVD_DATA_NUM}" ]]; then
        fvd_command+=(--num-videos "${FVD_DATA_NUM}")
    fi
    if [[ -n "${DEVICE}" ]]; then
        fvd_command+=(--device "${DEVICE}")
    fi
    echo
    echo "[FVD] ${GENERATED_VIDEOS_ROOT} vs ${GT_VIDEOS_ROOT}"
    "${fvd_command[@]}"
fi

echo
echo "Evaluation finished."
