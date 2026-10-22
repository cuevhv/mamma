#!/bin/bash
#
# Download MAMMA markerless multi-people datasets (meta, pred, videos, previews) from the MPI server.
# Requires registration at https://mamma.is.tue.mpg.de/
#
# Usage:
#   bash data/download_mamma_multi_people.sh --meta --pred --videos              # what to download
#   bash data/download_mamma_multi_people.sh --videos --ioi 01 02               # specific cameras
#   bash data/download_mamma_multi_people.sh --meta                             # only metadata
#   bash data/download_mamma_multi_people.sh --help
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
DOWNLOAD_META=0
DOWNLOAD_PRED=0
DOWNLOAD_VIDEOS=0
DOWNLOAD_VIDEOS_CRF16=0
DOWNLOAD_VIDEOS_CRF24=0
DOWNLOAD_PREVIEW=0
CLI_IOI=()

usage() {
    echo "Usage: bash data/download_mamma_multi_people.sh [OPTIONS]"
    echo ""
    echo "At least one of --meta, --pred, --videos, --videos-crf16, --videos-crf24, --preview is required."
    echo ""
    echo "Options:"
    echo "  --meta            Download metadata"
    echo "  --pred            Download predictions"
    echo "  --videos          Download video files"
    echo "  --videos-crf16    Download CRF-16 video files"
    echo "  --videos-crf24    Download CRF-24 video files"
    echo "  --preview         Download preview overlay grid video"
    echo "  --ioi 01 02 ...   Download only these IOI cameras (zero-padded numbers)"
    echo "                    If omitted, all 32 cameras are downloaded."
    echo "  --output DIR      Output directory (default: <repo>/data, where capture configs expect it)"
    echo "  -h, --help        Show this help message"
    exit 0
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --meta)           DOWNLOAD_META=1; shift ;;
        --pred)           DOWNLOAD_PRED=1; shift ;;
        --videos)         DOWNLOAD_VIDEOS=1; shift ;;
        --videos-crf16)   DOWNLOAD_VIDEOS_CRF16=1; shift ;;
        --videos-crf24)   DOWNLOAD_VIDEOS_CRF24=1; shift ;;
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

if [[ $DOWNLOAD_META -eq 0 && $DOWNLOAD_PRED -eq 0 && $DOWNLOAD_VIDEOS -eq 0 && \
      $DOWNLOAD_VIDEOS_CRF16 -eq 0 && $DOWNLOAD_VIDEOS_CRF24 -eq 0 && \
      $DOWNLOAD_PREVIEW -eq 0 ]]; then
    echo "Error: specify at least one of --meta, --pred, --videos, --videos-crf16, --videos-crf24, --preview" >&2
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

# Sequences (34 multi-people sequences)
SEQUENCES_3P=(
    mamma_markerless_multiple_people/260216_MultiMama_3_accidental_bump_000111_1
    mamma_markerless_multiple_people/260216_MultiMama_3_assist_walk_101100_1
    mamma_markerless_multiple_people/260216_MultiMama_3_conga_line_100110_1
    mamma_markerless_multiple_people/260216_MultiMama_3_dance_hands_circle_101100_1
    mamma_markerless_multiple_people/260216_MultiMama_3_dance_shoulders_circle_001011_1
    mamma_markerless_multiple_people/260216_MultiMama_3_group_hug_101010_1
    mamma_markerless_multiple_people/260216_MultiMama_3_hands_stack_huddle_110001_1
    mamma_markerless_multiple_people/260216_MultiMama_3_pass_clap_001101_1
    mamma_markerless_multiple_people/260216_MultiMama_3_social_circle_001110_1
    mamma_markerless_multiple_people/260216_MultiMama_3_telephone_chain_101001_1
    mamma_markerless_multiple_people/260216_MultiMama_3_twister_easy_01_001101_1
)

SEQUENCES_4P=(
    mamma_markerless_multiple_people/260216_MultiMama_4_accidental_bump_011110_1
    mamma_markerless_multiple_people/260216_MultiMama_4_conga_line_011101_1
    mamma_markerless_multiple_people/260216_MultiMama_4_dance_hands_circle_010111_1
    mamma_markerless_multiple_people/260216_MultiMama_4_dance_shoulders_circle_110110_1
    mamma_markerless_multiple_people/260216_MultiMama_4_hands_stack_huddle_011110_1
    mamma_markerless_multiple_people/260216_MultiMama_4_pass_clap_111010_1
    mamma_markerless_multiple_people/260216_MultiMama_4_social_circle_111100_1
    mamma_markerless_multiple_people/260216_MultiMama_4_telephone_chain_010111_1
    mamma_markerless_multiple_people/260216_MultiMama_4_twister_easy_001111_1
)

SEQUENCES_5P=(
    mamma_markerless_multiple_people/260216_MultiMama_5_conga_line_111101_1
    mamma_markerless_multiple_people/260216_MultiMama_5_dance_hands_circle_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_dance_shoulders_circle_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_group_hug_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_group_photo_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_hands_stack_huddle_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_pass_clap_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_social_circle_111110_1
    mamma_markerless_multiple_people/260216_MultiMama_5_telephone_chain_111110_1
)

SEQUENCES_6P=(
    mamma_markerless_multiple_people/260216_MultiMama_6_dance_hands_circle_111111_1
    mamma_markerless_multiple_people/260216_MultiMama_6_dance_shoulders_circle_111111_1
    mamma_markerless_multiple_people/260216_MultiMama_6_pass_clap_111111_1
    mamma_markerless_multiple_people/260216_MultiMama_6_social_circle_111111_1
    mamma_markerless_multiple_people/260216_MultiMama_6_telephone_chain_111111_1
)

SEQUENCE_GROUPS=(
    "3:${SEQUENCES_3P[*]}"
    "4:${SEQUENCES_4P[*]}"
    "5:${SEQUENCES_5P[*]}"
    "6:${SEQUENCES_6P[*]}"
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

pred_indices_for_count() {
    local person_count="$1"
    local idx

    for (( idx = 0; idx < person_count; idx++ )); do
        printf '%02d\n' "$idx"
    done
}

# -------------------------------------------------------------------
# Download
# -------------------------------------------------------------------
echo ""
echo "Sequences:  $(( ${#SEQUENCES_3P[@]} + ${#SEQUENCES_4P[@]} + ${#SEQUENCES_5P[@]} + ${#SEQUENCES_6P[@]} ))"
echo "Cameras:    ${#CAMERAS[@]}"
echo "Download:   $( [[ $DOWNLOAD_META -eq 1 ]] && echo -n "meta " )$( [[ $DOWNLOAD_PRED -eq 1 ]] && echo -n "pred " )$( [[ $DOWNLOAD_VIDEOS -eq 1 ]] && echo -n "videos " )$( [[ $DOWNLOAD_VIDEOS_CRF16 -eq 1 ]] && echo -n "videos-crf16 " )$( [[ $DOWNLOAD_VIDEOS_CRF24 -eq 1 ]] && echo -n "videos-crf24 " )$( [[ $DOWNLOAD_PREVIEW -eq 1 ]] && echo -n "preview" )"
echo "Output:     ${OUTPUT_DIR}"
echo ""

downloaded=0
failed=0

for sequence_group in "${SEQUENCE_GROUPS[@]}"; do
    person_count="${sequence_group%%:*}"
    sequence_list="${sequence_group#*:}"

    for ds_seq in $sequence_list; do
        echo "=== ${ds_seq} ==="

        # --- Meta (global + per-camera) ---
        if [[ $DOWNLOAD_META -eq 1 ]]; then
            download "${ds_seq}/meta/global.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            for cam in "${CAMERAS[@]}"; do
                download "${ds_seq}/meta/${cam}.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            done
        fi

        # --- Pred (per-person, count defined by the sequence group) ---
        if [[ $DOWNLOAD_PRED -eq 1 ]]; then
            if ! mapfile -t pred_indices < <(pred_indices_for_count "$person_count"); then
                exit 1
            fi
            for pidx in "${pred_indices[@]}"; do
                download "${ds_seq}/pred/params_${pidx}.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            done
        fi

        # --- Videos (per-camera) ---
        if [[ $DOWNLOAD_VIDEOS -eq 1 ]]; then
            for cam in "${CAMERAS[@]}"; do
                download "${ds_seq}/videos/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            done
        fi

        # --- Videos CRF-16 (per-camera) ---
        if [[ $DOWNLOAD_VIDEOS_CRF16 -eq 1 ]]; then
            for cam in "${CAMERAS[@]}"; do
                download "${ds_seq}/videos_crf16/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            done
        fi

        # --- Videos CRF-24 (per-camera) ---
        if [[ $DOWNLOAD_VIDEOS_CRF24 -eq 1 ]]; then
            for cam in "${CAMERAS[@]}"; do
                download "${ds_seq}/videos_crf24/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
            done
        fi

        # --- Preview (global) ---
        if [[ $DOWNLOAD_PREVIEW -eq 1 ]]; then
            download "${ds_seq}/preview/overlay_grid.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        fi

        echo ""
    done
done

echo "Done! Downloaded: ${downloaded}, Failed: ${failed}"
