#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPVC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${SPVC_ROOT}/.." && pwd)"

BENCHMARK_ROOT="${PANOPTIC_FIX_BENCHMARK_ROOT:-${SPVC_ROOT}/SPVC-PanopticFix-Benchmark}"
GENERATED_IMAGES_ROOT=""
GT_IMAGES_ROOT=""
GENERATED_VIDEOS_ROOT=""
GT_VIDEOS_ROOT=""

RUN_CLIP=1
RUN_FID=1
RUN_FVD=1
RUN_FID_A=1
CHECK_ONLY=0

CLIP_MODEL="ViT-B/32"
if [[ -f "${REPOSITORY_ROOT}/ViT-B-32.pt" ]]; then
    CLIP_MODEL="${REPOSITORY_ROOT}/ViT-B-32.pt"
fi
CLIP_BATCH_SIZE="64"

YOLO_MODEL="yolo11x.pt"
if [[ -f "${REPOSITORY_ROOT}/yolo11x.pt" ]]; then
    YOLO_MODEL="${REPOSITORY_ROOT}/yolo11x.pt"
fi
FID_A_GT_CROPS=""
if [[ -d "${REPOSITORY_ROOT}/metrics/fid-a/edit-scene-gt" ]]; then
    FID_A_GT_CROPS="${REPOSITORY_ROOT}/metrics/fid-a/edit-scene-gt"
fi

FVD_DATA_NUM=""
FVD_MAX_FRAMES="25"
FVD_METHODS="videogpt,styleganv"
DEVICE=""
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/evaluate_panoptic_fix.sh [options]

Computes edit-aware CLIP-F/CLIP-V, FID, FVD, and FID-A for the
SPVC-PanopticFix-Benchmark. Unlike the NVS evaluator, no dataset or scale
subfolder is required.

Path options:
  --benchmark-root PATH          Benchmark root.
  --generated-images-root PATH   Generated frame directory.
  --gt-images-root PATH          GT frame directory.
  --generated-videos-root PATH   Generated MP4 directory.
  --gt-videos-root PATH          GT MP4 directory.
  --fid-a-gt-crops PATH          Optional precomputed GT vehicle crops. When
                                 omitted, GT crops are generated temporarily.

Metric options:
  --skip-clip                    Skip edit-aware CLIP-F and CLIP-V.
  --skip-fid                     Skip FID.
  --skip-fvd                     Skip FVD.
  --skip-fid-a                   Skip FID-A.
  --device DEVICE                Torch/Ultralytics device, e.g. cuda:0 or cpu.
  --clip-model MODEL             CLIP model name or local checkpoint.
  --clip-batch-size N            CLIP encoding batch size (default: 64).
  --yolo-model MODEL             YOLO checkpoint for FID-A (default: yolo11x.pt).
  --fvd-data-num N               Use the first N sorted videos (default: all).
  --fvd-max-frames N             Frames per video used for FVD (default: 25).
  --fvd-methods LIST             videogpt, styleganv, or both (default: both).
  --check-only                   Validate paths and report counts, then exit.
  -h, --help                     Show this message.

Environment:
  PYTHON_BIN                     Python executable (default: python).
  PANOPTIC_FIX_BENCHMARK_ROOT    Alternative default benchmark root.
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

count_images() {
    find "${1}" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
           -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l | tr -d ' '
}

count_videos() {
    find "${1}" -maxdepth 1 -type f -iname '*.mp4' | wc -l | tr -d ' '
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark-root)
            require_value "$@"
            BENCHMARK_ROOT="$2"
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
        --fid-a-gt-crops)
            require_value "$@"
            FID_A_GT_CROPS="$2"
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
        --yolo-model)
            require_value "$@"
            YOLO_MODEL="$2"
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
        --skip-fid-a)
            RUN_FID_A=0
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
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

if [[ "${RUN_CLIP}" -eq 0 && "${RUN_FID}" -eq 0 && "${RUN_FVD}" -eq 0 && "${RUN_FID_A}" -eq 0 ]]; then
    echo "[ERROR] CLIP, FID, FVD, and FID-A are all disabled" >&2
    exit 2
fi
if [[ ! "${CLIP_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --clip-batch-size must be a positive integer" >&2
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
for method in "${FVD_METHOD_ARRAY[@]}"; do
    if [[ "${method}" != "videogpt" && "${method}" != "styleganv" ]]; then
        echo "[ERROR] Unsupported FVD method: ${method}" >&2
        exit 2
    fi
done

GENERATED_IMAGES_ROOT="${GENERATED_IMAGES_ROOT:-${BENCHMARK_ROOT}/results/images}"
GT_IMAGES_ROOT="${GT_IMAGES_ROOT:-${BENCHMARK_ROOT}/gt/images}"
GENERATED_VIDEOS_ROOT="${GENERATED_VIDEOS_ROOT:-${BENCHMARK_ROOT}/results/videos}"
GT_VIDEOS_ROOT="${GT_VIDEOS_ROOT:-${BENCHMARK_ROOT}/gt/videos}"

if [[ "${RUN_CLIP}" -eq 1 || "${RUN_FID}" -eq 1 || "${RUN_FID_A}" -eq 1 ]]; then
    require_directory "${GENERATED_IMAGES_ROOT}"
    require_directory "${GT_IMAGES_ROOT}"
fi
if [[ "${RUN_FVD}" -eq 1 ]]; then
    require_directory "${GENERATED_VIDEOS_ROOT}"
    require_directory "${GT_VIDEOS_ROOT}"
fi
if [[ "${RUN_FID_A}" -eq 1 && -n "${FID_A_GT_CROPS}" ]]; then
    require_directory "${FID_A_GT_CROPS}"
fi

echo "========== SPVC PanopticFix Evaluation =========="
echo "Generated images: ${GENERATED_IMAGES_ROOT}"
echo "GT images:        ${GT_IMAGES_ROOT}"
echo "Generated videos: ${GENERATED_VIDEOS_ROOT}"
echo "GT videos:        ${GT_VIDEOS_ROOT}"
if [[ -d "${GENERATED_IMAGES_ROOT}" ]]; then
    echo "Generated image count: $(count_images "${GENERATED_IMAGES_ROOT}")"
fi
if [[ -d "${GT_IMAGES_ROOT}" ]]; then
    echo "GT image count:        $(count_images "${GT_IMAGES_ROOT}")"
fi
if [[ -d "${GENERATED_VIDEOS_ROOT}" ]]; then
    echo "Generated video count: $(count_videos "${GENERATED_VIDEOS_ROOT}")"
fi
if [[ -d "${GT_VIDEOS_ROOT}" ]]; then
    echo "GT video count:        $(count_videos "${GT_VIDEOS_ROOT}")"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "Check finished; no metrics were computed."
    exit 0
fi

if [[ "${RUN_CLIP}" -eq 1 ]]; then
    clip_command=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/panoptic_fix/calculate_clip_metrics.py"
        --generated-images "${GENERATED_IMAGES_ROOT}"
        --gt-images "${GT_IMAGES_ROOT}"
        --clip-model "${CLIP_MODEL}"
        --batch-size "${CLIP_BATCH_SIZE}"
    )
    if [[ -n "${DEVICE}" ]]; then
        clip_command+=(--device "${DEVICE}")
    fi
    echo
    echo "[CLIP] Computing edit-aware CLIP-F and CLIP-V"
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

FID_A_TEMP_DIR=""
cleanup_fid_a_temp() {
    if [[ -n "${FID_A_TEMP_DIR}" && -d "${FID_A_TEMP_DIR}" ]]; then
        rm -rf -- "${FID_A_TEMP_DIR}"
    fi
}
trap cleanup_fid_a_temp EXIT

if [[ "${RUN_FID_A}" -eq 1 ]]; then
    FID_A_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/spvc-panoptic-fid-a.XXXXXX")"
    FID_A_GENERATED_CROPS="${FID_A_TEMP_DIR}/generated"
    mkdir -p "${FID_A_GENERATED_CROPS}"

    crop_generated_command=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/panoptic_fix/crop_fid_a.py"
        --input-dir "${GENERATED_IMAGES_ROOT}"
        --output-dir "${FID_A_GENERATED_CROPS}"
        --model "${YOLO_MODEL}"
    )
    if [[ -n "${DEVICE}" ]]; then
        crop_generated_command+=(--device "${DEVICE}")
    fi
    echo
    echo "[FID-A] Detecting vehicles in generated frames"
    "${crop_generated_command[@]}"

    if [[ -z "${FID_A_GT_CROPS}" ]]; then
        FID_A_GT_CROPS="${FID_A_TEMP_DIR}/gt"
        mkdir -p "${FID_A_GT_CROPS}"
        crop_gt_command=(
            "${PYTHON_BIN}"
            "${SCRIPT_DIR}/panoptic_fix/crop_fid_a.py"
            --input-dir "${GT_IMAGES_ROOT}"
            --output-dir "${FID_A_GT_CROPS}"
            --model "${YOLO_MODEL}"
        )
        if [[ -n "${DEVICE}" ]]; then
            crop_gt_command+=(--device "${DEVICE}")
        fi
        echo "[FID-A] No precomputed GT crops found; detecting GT vehicles"
        "${crop_gt_command[@]}"
    else
        echo "[FID-A] Using GT crops: ${FID_A_GT_CROPS}"
    fi

    fid_a_command=(
        "${PYTHON_BIN}"
        -m pytorch_fid
        "${FID_A_GENERATED_CROPS}"
        "${FID_A_GT_CROPS}"
    )
    if [[ -n "${DEVICE}" ]]; then
        fid_a_command+=(--device "${DEVICE}")
    fi
    echo "[FID-A] ${FID_A_GENERATED_CROPS} vs ${FID_A_GT_CROPS}"
    "${fid_a_command[@]}"
fi

echo
echo "PanopticFix evaluation finished."
