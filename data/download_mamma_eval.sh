#!/bin/bash
#
# Download MammaEval datasets (GT, videos, masks, markers, previews) from the MPI server.
#
# Usage:
#   bash data/download_mamma_eval.sh --gt --videos --masks              # what to download
#   bash data/download_mamma_eval.sh --gt --videos --masks --ioi 01 02  # specific cameras
#   bash data/download_mamma_eval.sh --gt                               # only GT, all cameras
#   bash data/download_mamma_eval.sh --help
#
set -euo pipefail

# ===================================================================
# CLI ARGUMENT PARSING
# ===================================================================

# Anchor OUTPUT_DIR to the script's directory (the repo's data/ folder)
# rather than the caller's cwd, so files always land where the
# configs/examples/captures/*.json files expect them, regardless of
# where the user invokes the script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
DOWNLOAD_GT=0
DOWNLOAD_VIDEOS=0
DOWNLOAD_MASKS=0
DOWNLOAD_VIDEOS_CRF16=0
DOWNLOAD_VIDEOS_CRF24=0
DOWNLOAD_MARKERS=0
DOWNLOAD_PREVIEW=0
CLI_IOI=()

usage() {
    echo "Usage: bash data/download_mamma_eval.sh [OPTIONS]"
    echo ""
    echo "At least one of --gt, --videos, --masks, --videos-crf16, --videos-crf24, --markers, --preview is required."
    echo ""
    echo "Options:"
    echo "  --gt              Download ground-truth data"
    echo "  --videos          Download video files"
    echo "  --masks           Download mask files"
    echo "  --videos-crf16    Download CRF16 video files"
    echo "  --videos-crf24    Download CRF24 video files"
    echo "  --markers         Download marker files (eval_extra sequences only)"
    echo "  --preview         Download preview files (overlay_grid.mp4, masks_grid.mp4)"
    echo "  --ioi 01 02 ...   Download only these IOI cameras (zero-padded numbers)"
    echo "                    If omitted, all cameras are downloaded."
    echo "  --output DIR      Output directory (default: <repo>/data, where capture configs expect it)"
    echo "  -h, --help        Show this help message"
    exit 0
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt)             DOWNLOAD_GT=1; shift ;;
        --videos)         DOWNLOAD_VIDEOS=1; shift ;;
        --masks)          DOWNLOAD_MASKS=1; shift ;;
        --videos-crf16)   DOWNLOAD_VIDEOS_CRF16=1; shift ;;
        --videos-crf24)   DOWNLOAD_VIDEOS_CRF24=1; shift ;;
        --markers)        DOWNLOAD_MARKERS=1; shift ;;
        --preview)        DOWNLOAD_PREVIEW=1; shift ;;
        --ioi)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CLI_IOI+=("IOI_$1")
                shift
            done
            ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1 (use --help for usage)" >&2; exit 1 ;;
    esac
done

if [[ $DOWNLOAD_GT -eq 0 && $DOWNLOAD_VIDEOS -eq 0 && $DOWNLOAD_MASKS -eq 0 && \
      $DOWNLOAD_VIDEOS_CRF16 -eq 0 && $DOWNLOAD_VIDEOS_CRF24 -eq 0 && \
      $DOWNLOAD_MARKERS -eq 0 && $DOWNLOAD_PREVIEW -eq 0 ]]; then
    echo "Error: specify at least one of --gt, --videos, --masks, --videos-crf16, --videos-crf24, --markers, --preview" >&2
    exit 1
fi

# All cameras if --ioi not provided
if [[ ${#CLI_IOI[@]} -gt 0 ]]; then
    CAMERAS=("${CLI_IOI[@]}")
else
    CAMERAS=(
        IOI_01 IOI_02 IOI_03 IOI_04 IOI_05 IOI_06 IOI_07 IOI_08
        IOI_09 IOI_10 IOI_11 IOI_12 IOI_13 IOI_14 IOI_15 IOI_16
        IOI_17 IOI_18 IOI_19 IOI_20 IOI_21 IOI_22 IOI_23 IOI_24
        IOI_25 IOI_26 IOI_27 IOI_28 IOI_29 IOI_30 IOI_31 IOI_32
    )
fi

# Sequences to download (comment out sequences you don't need)
SEQUENCES=(
    # --- mamma_eval_singles (22 sequences, 16 cameras) ---
    mamma_eval_singles/230929_WhiteRabbit_CatchBall_50048_1
    mamma_eval_singles/230929_WhiteRabbit_CatchBall_50049_1
    mamma_eval_singles/230929_WhiteRabbit_CoffeeTable_50049_1
    mamma_eval_singles/230929_WhiteRabbit_CoffeeTable_50050_1
    mamma_eval_singles/230929_WhiteRabbit_LungeStretch_50048_1
    mamma_eval_singles/230929_WhiteRabbit_LungeStretch_50049_1
    mamma_eval_singles/230929_WhiteRabbit_LungeStretch_50050_1
    mamma_eval_singles/230929_WhiteRabbit_Stepper_50049_1
    mamma_eval_singles/230929_WhiteRabbit_Stepper_50050_1
    mamma_eval_singles/230929_WhiteRabbit_Stretches_50049_1
    mamma_eval_singles/230929_WhiteRabbit_SubjectCalib_50048_1
    mamma_eval_singles/230929_WhiteRabbit_SubjectCalib_50050_1
    mamma_eval_singles/230929_WhiteRabbit_WalkBarStool_50048_1
    mamma_eval_singles/230929_WhiteRabbit_WalkBarStool_50049_1
    mamma_eval_singles/230929_WhiteRabbit_WalkBarStool_50050_1
    mamma_eval_singles/230929_WhiteRabbit_WalkGreenChair_50048_1
    mamma_eval_singles/230929_WhiteRabbit_WalkGreenChair_50049_1
    mamma_eval_singles/230929_WhiteRabbit_WalkGreenChair_50050_1
    mamma_eval_singles/230929_WhiteRabbit_WalkLaptop_50048_1
    mamma_eval_singles/230929_WhiteRabbit_WalkLaptop_50048_2
    mamma_eval_singles/230929_WhiteRabbit_WalkLaptop_50049_3
    mamma_eval_singles/230929_WhiteRabbit_WalkLaptop_50050_1

    # --- mamma_eval_extra (12 sequences, 16 cameras) ---
    mamma_eval_extra/070525_BedlamLab_Eval_Solo_Dancing_00202_1
    mamma_eval_extra/070525_BedlamLab_Eval_Solo_Dancing_00219_1
    mamma_eval_extra/070525_BedlamLab_Eval_Solo_Dancing_00236_1
    mamma_eval_extra/070525_BedlamLab_Eval_Subject_Calib_00202_1
    mamma_eval_extra/070525_BedlamLab_Eval_Subject_Calib_00219_1
    mamma_eval_extra/070525_BedlamLab_Eval_Subject_Calib_00236_2_1
    mamma_eval_extra/070525_BedlamLab_Eval_Walking_00202_1
    mamma_eval_extra/070525_BedlamLab_Eval_Walking_00219_1
    mamma_eval_extra/070525_BedlamLab_Eval_Walking_00236_1
    mamma_eval_extra/070525_BedlamLab_Eval_WarmUp_00202_1
    mamma_eval_extra/070525_BedlamLab_Eval_WarmUp_00219_1
    mamma_eval_extra/070525_BedlamLab_Eval_WarmUp_00236_1

    # --- mamma_eval_dance (18 sequences, 32 cameras) ---
    mamma_eval_dance/250225_WestCoastSwing_Basic_Whip_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Improv02_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Improv05_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Improv08_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Improv12_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Improv22_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Left_Side_Pass_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Outside_Turn_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Outside_Turn_03675_03676_2_1
    mamma_eval_dance/250225_WestCoastSwing_Roll_In_Out_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Roll_In_Out_03675_03676_2_1
    mamma_eval_dance/250225_WestCoastSwing_Same_Side_Whip_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Same_Side_Whip_03675_03676_2_1
    mamma_eval_dance/250225_WestCoastSwing_Same_Side_Whip_03675_03676_3_1
    mamma_eval_dance/250225_WestCoastSwing_Sugar_Push_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Sugar_Tuck_03675_03676_2_1
    mamma_eval_dance/250225_WestCoastSwing_Sugar_Turn_03675_03676_1
    mamma_eval_dance/250225_WestCoastSwing_Underarm_Turn_03675_03676_1
)

# ===================================================================
# END OF SETTINGS
# ===================================================================

BASE_URL="https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"
REMOTE_ROOT="datasets"

urle () {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

mamma_is_error_response() {
    local file_path="$1"

    [[ -f "$file_path" ]] || return 0

    if head -c 256 "$file_path" | grep -Fqi "Error: File not found."; then
        return 0
    fi

    if head -c 256 "$file_path" | grep -Fqi "<!DOCTYPE html"; then
        return 0
    fi

    if head -c 256 "$file_path" | grep -Fqi "<html"; then
        return 0
    fi

    return 1
}

mamma_is_valid_download() {
    local file_path="$1"

    [[ -f "$file_path" && -s "$file_path" ]] || return 1
    mamma_is_error_response "$file_path" && return 1
    return 0
}

load_mamma_credentials() {
    if [[ -n "${MAMMA_USERNAME:-}" && -n "${MAMMA_PASSWORD:-}" ]]; then
        username=$(urle "$MAMMA_USERNAME")
        password=$(urle "$MAMMA_PASSWORD")
        return 0
    fi

    echo ""
    echo "You need to register at https://mamma.is.tue.mpg.de/"
    read -r -p "Username (MAMMA): " MAMMA_USERNAME
    read -r -s -p "Password (MAMMA): " MAMMA_PASSWORD
    echo ""

    username=$(urle "$MAMMA_USERNAME")
    password=$(urle "$MAMMA_PASSWORD")
}

# -------------------------------------------------------------------
# Credentials
# -------------------------------------------------------------------
load_mamma_credentials

# -------------------------------------------------------------------
# Download helper
# -------------------------------------------------------------------
download () {
    local sfile="$1"
    local outpath="${OUTPUT_DIR}/${sfile}"
    local remote_sfile="${REMOTE_ROOT}/${sfile}"

    if mamma_is_valid_download "$outpath"; then
        echo "  [skip] ${sfile}"
        return 0
    fi

    mkdir -p "$(dirname "$outpath")"
    if wget --post-data "username=$username&password=$password" \
            "${BASE_URL}&sfile=${remote_sfile}" \
            -O "$outpath" \
            --continue --no-verbose --show-progress 2>&1; then
        if mamma_is_valid_download "$outpath"; then
            echo "  [ok] ${sfile}"
            return 0
        fi
    fi

    rm -f "$outpath"
    echo "  [FAIL] ${sfile}"
    return 1
}

# -------------------------------------------------------------------
# Download
# -------------------------------------------------------------------
echo ""
echo "Sequences:  ${#SEQUENCES[@]}"
echo "Cameras:    ${#CAMERAS[@]}"
echo "Download:   $( [[ $DOWNLOAD_GT -eq 1 ]] && echo -n "gt " )$( [[ $DOWNLOAD_VIDEOS -eq 1 ]] && echo -n "videos " )$( [[ $DOWNLOAD_MASKS -eq 1 ]] && echo -n "masks " )$( [[ $DOWNLOAD_VIDEOS_CRF16 -eq 1 ]] && echo -n "videos-crf16 " )$( [[ $DOWNLOAD_VIDEOS_CRF24 -eq 1 ]] && echo -n "videos-crf24 " )$( [[ $DOWNLOAD_MARKERS -eq 1 ]] && echo -n "markers " )$( [[ $DOWNLOAD_PREVIEW -eq 1 ]] && echo -n "preview" )"
echo "Output:     ${OUTPUT_DIR}"
echo ""

downloaded=0
failed=0

for ds_seq in "${SEQUENCES[@]}"; do
    dataset="${ds_seq%%/*}"
    seq="${ds_seq#*/}"
    echo "=== ${ds_seq} ==="

    # Determine cameras for this sequence
    if [[ "$dataset" == "mamma_eval_singles" ]]; then
        # 16 cameras
        if [[ ${#CLI_IOI[@]} -gt 0 ]]; then
            seq_cameras=("${CLI_IOI[@]}")
        else
            seq_cameras=(IOI_01 IOI_02 IOI_03 IOI_04 IOI_05 IOI_06 IOI_07 IOI_08
                         IOI_09 IOI_10 IOI_11 IOI_12 IOI_13 IOI_14 IOI_15 IOI_16)
        fi
    else
        seq_cameras=("${CAMERAS[@]}")
    fi

    # --- GT ---
    if [[ $DOWNLOAD_GT -eq 1 ]]; then
        # Always download global.npz
        download "${ds_seq}/gt/global.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))

        # Per-camera GT
        for cam in "${seq_cameras[@]}"; do
            download "${ds_seq}/gt/${cam}.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Videos ---
    if [[ $DOWNLOAD_VIDEOS -eq 1 ]]; then
        for cam in "${seq_cameras[@]}"; do
            download "${ds_seq}/videos/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Masks ---
    if [[ $DOWNLOAD_MASKS -eq 1 ]]; then
        for cam in "${seq_cameras[@]}"; do
            download "${ds_seq}/masks/${cam}_masks.tar" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Videos CRF16 ---
    if [[ $DOWNLOAD_VIDEOS_CRF16 -eq 1 ]]; then
        for cam in "${seq_cameras[@]}"; do
            download "${ds_seq}/videos_crf16/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Videos CRF24 ---
    if [[ $DOWNLOAD_VIDEOS_CRF24 -eq 1 ]]; then
        for cam in "${seq_cameras[@]}"; do
            download "${ds_seq}/videos_crf24/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Markers (eval_extra only) ---
    if [[ $DOWNLOAD_MARKERS -eq 1 ]]; then
        if [[ "${ds_seq}" == mamma_eval_extra/* ]]; then
            download "${ds_seq}/markers/vicon_m37.npy"    && downloaded=$((downloaded+1)) || failed=$((failed+1))
            download "${ds_seq}/markers/baseline_m37.npy" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            download "${ds_seq}/markers/labels_m37.npy"   && downloaded=$((downloaded+1)) || failed=$((failed+1))
        fi
    fi

    # --- Preview ---
    if [[ $DOWNLOAD_PREVIEW -eq 1 ]]; then
        download "${ds_seq}/preview/overlay_grid.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        download "${ds_seq}/preview/masks_grid.mp4"   && downloaded=$((downloaded+1)) || failed=$((failed+1))
    fi

    echo ""
done

echo "Done! Downloaded: ${downloaded}, Failed: ${failed}"
