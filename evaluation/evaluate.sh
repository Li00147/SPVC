#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPVC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET=""
SCALE="4.0"
BENCHMARK_ROOT="${SPVC_ROOT}/SPVC-Benchamrk"
RESULTS_ROOT=""
GT_ROOT=""
FVD_DATA_NUM=""
FVD_MAX_FRAMES="40"
RUN_FID=1
RUN_FVD=1
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluate.sh --dataset DATASET [options]

Required:
  --dataset NAME          Dataset folder under SPVC-Benchamrk (for example,
                          nuscenes, waymo, or pandaset).

Options:
  --scale VALUE           Scale subfolder to evaluate (default: 4.0).
  --benchmark-root PATH   Benchmark root containing dataset folders.
  --results-root PATH     Override <benchmark>/<dataset>/results.
  --gt-root PATH          Override <benchmark>/<dataset>/gt.
  --fvd-data-num N        Limit FVD to the first N sorted videos (default: all).
  --fvd-max-frames N      Maximum frames loaded per video (default: 40).
  --skip-fid              Do not calculate FID.
  --skip-fvd              Do not calculate FVD.
  -h, --help              Show this message.

Environment:
  PYTHON_BIN              Python executable to use (default: python).
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "[ERROR] Missing value for ${1}" >&2
        usage >&2
        exit 2
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
if [[ "${RUN_FID}" -eq 0 && "${RUN_FVD}" -eq 0 ]]; then
    echo "[ERROR] Both FID and FVD are disabled" >&2
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

DATASET_ROOT="${BENCHMARK_ROOT}/${DATASET}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATASET_ROOT}/results}"
GT_ROOT="${GT_ROOT:-${DATASET_ROOT}/gt}"

SCALE_IMAGES="${RESULTS_ROOT}/images/${SCALE}"
SCALE_VIDEOS="${RESULTS_ROOT}/videos/${SCALE}"
GT_IMAGES="${GT_ROOT}/gt_images"
GT_VIDEOS="${GT_ROOT}/gt_videos"

echo "========== SPVC Benchmark Evaluation =========="
echo "Dataset:      ${DATASET}"
echo "Scale:        ${SCALE}"
echo "Results root: ${RESULTS_ROOT}"
echo "GT root:      ${GT_ROOT}"

if [[ "${RUN_FID}" -eq 1 ]]; then
    for required_dir in "${SCALE_IMAGES}" "${GT_IMAGES}"; do
        if [[ ! -d "${required_dir}" ]]; then
            echo "[ERROR] Directory not found: ${required_dir}" >&2
            exit 1
        fi
    done

    echo "[FID] ${SCALE_IMAGES} vs ${GT_IMAGES}"
    "${PYTHON_BIN}" -m pytorch_fid "${SCALE_IMAGES}" "${GT_IMAGES}"
fi

if [[ "${RUN_FVD}" -eq 1 ]]; then
    for required_dir in "${SCALE_VIDEOS}" "${GT_VIDEOS}"; do
        if [[ ! -d "${required_dir}" ]]; then
            echo "[ERROR] Directory not found: ${required_dir}" >&2
            exit 1
        fi
    done

    fvd_command=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/fvd/calculate_fvd.py"
        --generated-videos "${SCALE_VIDEOS}"
        --gt-videos "${GT_VIDEOS}"
        --max-frames "${FVD_MAX_FRAMES}"
    )
    if [[ -n "${FVD_DATA_NUM}" ]]; then
        fvd_command+=(--num-videos "${FVD_DATA_NUM}")
    fi

    echo "[FVD] ${SCALE_VIDEOS} vs ${GT_VIDEOS}"
    "${fvd_command[@]}"
fi

echo "Evaluation finished."
