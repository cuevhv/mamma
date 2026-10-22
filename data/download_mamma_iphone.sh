#!/bin/bash
#
# Download MAMMA markerless iPhone capture datasets (meta, pred, videos, previews) from the MPI server.
# Requires registration at https://mamma.is.tue.mpg.de/
#
# Usage:
#   bash data/download_mamma_iphone.sh --meta --pred --videos              # what to download
#   bash data/download_mamma_iphone.sh --videos --cam A001 B001            # specific cameras
#   bash data/download_mamma_iphone.sh --meta --indoors                    # only indoor sequences
#   bash data/download_mamma_iphone.sh --help
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
DOWNLOAD_VIDEOS_LIGHT=0
DOWNLOAD_PREVIEW=0
DL_INDOORS=0
DL_OUTDOORS=0
CLI_CAMS=()

usage() {
    echo "Usage: bash data/download_mamma_iphone.sh [DATA_FLAGS] [LOCATION_FLAGS] [OPTIONS]"
    echo ""
    echo "At least one data flag is required."
    echo ""
    echo "Data flags (what to download):"
    echo "  --meta            Download metadata (camera params, sequence info)"
    echo "  --pred            Download SMPL-X predictions"
    echo "  --videos          Download video files (H.265, CRF 16)"
    echo "  --videos-light    Download lightweight video files (H.265, CRF 24)"
    echo "  --preview         Download preview overlay grid video"
    echo ""
    echo "Subset flags (which sequences):"
    echo "  --indoors         Indoor sequences only (16)"
    echo "  --outdoors        Outdoor sequences only (26)"
    echo "  If neither is specified, both subsets are downloaded."
    echo ""
    echo "Options:"
    echo "  --cam A001 B001   Download only these cameras"
    echo "                    If omitted, all 4 cameras (A001-D001) are downloaded."
    echo "  --output DIR      Output directory (default: <repo>/data, where capture configs expect it)"
    echo "  -h, --help        Show this help message"
    exit 0
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --meta)          DOWNLOAD_META=1; shift ;;
        --pred)          DOWNLOAD_PRED=1; shift ;;
        --videos)        DOWNLOAD_VIDEOS=1; shift ;;
        --videos-light)  DOWNLOAD_VIDEOS_LIGHT=1; shift ;;
        --preview)       DOWNLOAD_PREVIEW=1; shift ;;
        --indoors)       DL_INDOORS=1; shift ;;
        --outdoors)      DL_OUTDOORS=1; shift ;;
        --cam)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CLI_CAMS+=("$1")
                shift
            done
            ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1 (use --help for usage)" >&2; exit 1 ;;
    esac
done

if [[ $DOWNLOAD_META -eq 0 && $DOWNLOAD_PRED -eq 0 && $DOWNLOAD_VIDEOS -eq 0 && \
      $DOWNLOAD_VIDEOS_LIGHT -eq 0 && $DOWNLOAD_PREVIEW -eq 0 ]]; then
    echo "Error: specify at least one of --meta, --pred, --videos, --videos-light, --preview" >&2
    exit 1
fi

# Default: both indoors and outdoors
if [[ $DL_INDOORS -eq 0 && $DL_OUTDOORS -eq 0 ]]; then
    DL_INDOORS=1
    DL_OUTDOORS=1
fi

# All cameras if --cam not provided
if [[ ${#CLI_CAMS[@]} -gt 0 ]]; then
    CAMERAS=("${CLI_CAMS[@]}")
else
    CAMERAS=(A001 B001 C001 D001)
fi

# ===================================================================
# SEQUENCE DEFINITIONS
# ===================================================================

# Indoor sequences (16): 1 or 2 people each
# Format: person_count:sequence_path
INDOORS_SEQUENCES=(
    "1:mamma_markerless_iphones/indoors/crossing_arms"
    "1:mamma_markerless_iphones/indoors/grabbing_bucket"
    "2:mamma_markerless_iphones/indoors/grappling"
    "2:mamma_markerless_iphones/indoors/holding_person_lifted"
    "2:mamma_markerless_iphones/indoors/hugging"
    "1:mamma_markerless_iphones/indoors/jumping"
    "2:mamma_markerless_iphones/indoors/lean_on_shoulder_tap"
    "1:mamma_markerless_iphones/indoors/performing_splits"
    "2:mamma_markerless_iphones/indoors/person_lift_from_ground"
    "1:mamma_markerless_iphones/indoors/running"
    "1:mamma_markerless_iphones/indoors/sitting_cross_legged_on_ground"
    "1:mamma_markerless_iphones/indoors/sitting_on_stool"
    "1:mamma_markerless_iphones/indoors/squatting"
    "1:mamma_markerless_iphones/indoors/stacking_cones"
    "1:mamma_markerless_iphones/indoors/talking_on_phone"
    "1:mamma_markerless_iphones/indoors/warming_up"
)

# Outdoor sequences (26): 1 or 2 people each
OUTDOORS_SEQUENCES=(
    "2:mamma_markerless_iphones/outdoors/MANUAL_START_two_friends"
    "2:mamma_markerless_iphones/outdoors/MANUAL_START_bumping_to_eachother"
    "1:mamma_markerless_iphones/outdoors/crossing_arms"
    "2:mamma_markerless_iphones/outdoors/greeting_interaction"
    "1:mamma_markerless_iphones/outdoors/jumping_excercise"
    "1:mamma_markerless_iphones/outdoors/kicking_ball_around"
    "1:mamma_markerless_iphones/outdoors/long_jump"
    "1:mamma_markerless_iphones/outdoors/moving_cone_around"
    "2:mamma_markerless_iphones/outdoors/passing_ball"
    "1:mamma_markerless_iphones/outdoors/playing_with_ball"
    "2:mamma_markerless_iphones/outdoors/pushing_and_lifting_from_ground"
    "1:mamma_markerless_iphones/outdoors/run_and_kick_ball"
    "1:mamma_markerless_iphones/outdoors/running"
    "1:mamma_markerless_iphones/outdoors/running_2"
    "1:mamma_markerless_iphones/outdoors/running_jumping"
    "1:mamma_markerless_iphones/outdoors/sitting_on_stool"
    "1:mamma_markerless_iphones/outdoors/sitting_on_stool_2"
    "1:mamma_markerless_iphones/outdoors/sitting_on_stool_crossing_leg"
    "2:mamma_markerless_iphones/outdoors/spinning_together"
    "1:mamma_markerless_iphones/outdoors/squatting"
    "1:mamma_markerless_iphones/outdoors/squatting_2"
    "1:mamma_markerless_iphones/outdoors/stacking_cones"
    "1:mamma_markerless_iphones/outdoors/talking_on_the_phone"
    "2:mamma_markerless_iphones/outdoors/tapping_on_shoulder"
    "1:mamma_markerless_iphones/outdoors/warmup_2"
    "2:mamma_markerless_iphones/outdoors/wheelbarrow_walk"
)

# Build final list
ALL_SEQUENCES=()
[[ $DL_INDOORS -eq 1 ]]  && ALL_SEQUENCES+=("${INDOORS_SEQUENCES[@]}")
[[ $DL_OUTDOORS -eq 1 ]] && ALL_SEQUENCES+=("${OUTDOORS_SEQUENCES[@]}")

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
echo "Sequences:  ${#ALL_SEQUENCES[@]}"
echo "Cameras:    ${#CAMERAS[@]} (${CAMERAS[*]})"
echo "Location:   $( [[ $DL_INDOORS -eq 1 ]] && echo -n "indoors " )$( [[ $DL_OUTDOORS -eq 1 ]] && echo -n "outdoors" )"
echo "Download:   $( [[ $DOWNLOAD_META -eq 1 ]] && echo -n "meta " )$( [[ $DOWNLOAD_PRED -eq 1 ]] && echo -n "pred " )$( [[ $DOWNLOAD_VIDEOS -eq 1 ]] && echo -n "videos " )$( [[ $DOWNLOAD_VIDEOS_LIGHT -eq 1 ]] && echo -n "videos-light " )$( [[ $DOWNLOAD_PREVIEW -eq 1 ]] && echo -n "preview" )"
echo "Output:     ${OUTPUT_DIR}"
echo ""

downloaded=0
failed=0

for entry in "${ALL_SEQUENCES[@]}"; do
    person_count="${entry%%:*}"
    ds_seq="${entry#*:}"
    echo "=== ${ds_seq} ==="

    # --- Meta (global + per-camera) ---
    if [[ $DOWNLOAD_META -eq 1 ]]; then
        download "${ds_seq}/meta/global.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        for cam in "${CAMERAS[@]}"; do
            download "${ds_seq}/meta/${cam}.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Pred (per-person) ---
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

    # --- Videos Light (per-camera) ---
    if [[ $DOWNLOAD_VIDEOS_LIGHT -eq 1 ]]; then
        for cam in "${CAMERAS[@]}"; do
            download "${ds_seq}/videos_light/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Preview ---
    if [[ $DOWNLOAD_PREVIEW -eq 1 ]]; then
        download "${ds_seq}/preview/overlay_grid.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
    fi

    echo ""
done

echo "Done! Downloaded: ${downloaded}, Failed: ${failed}"
