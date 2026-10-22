#!/bin/bash
#
# Download MAMMA markerless dance datasets (meta, pred, videos) from the MPI server.
# Requires registration at https://mamma.is.tue.mpg.de/
#
# Usage:
#   bash data/download_mamma_dance.sh --meta --pred --videos              # what to download
#   bash data/download_mamma_dance.sh --videos --ioi 01 02               # specific cameras
#   bash data/download_mamma_dance.sh --meta                             # only metadata
#   bash data/download_mamma_dance.sh --help
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
DL_WESTCOASTSWING=0
DL_BACHATA=0
DL_BREAKDANCE=0
DL_BALLROOM=0
CLI_IOI=()

usage() {
    echo "Usage: bash data/download_mamma_dance.sh [DATA_FLAGS] [DANCE_FLAGS] [OPTIONS]"
    echo ""
    echo "At least one data flag and one dance flag are required."
    echo ""
    echo "Data flags (what to download):"
    echo "  --meta            Download metadata"
    echo "  --pred            Download predictions"
    echo "  --videos          Download video files"
    echo "  --videos-crf16    Download CRF16 video files"
    echo "  --videos-crf24    Download CRF24 video files"
    echo "  --preview         Download preview overlay video"
    echo ""
    echo "Dance flags (which sequences):"
    echo "  --westcoastswing  West Coast Swing sequences (19)"
    echo "  --bachata         Bachata sequences (32)"
    echo "  --breakdance      Breakdance sequences (17)"
    echo "  --ballroom        Ballroom sequences (55)"
    echo "  --all-dances      All dance types"
    echo ""
    echo "Options:"
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
        --westcoastswing) DL_WESTCOASTSWING=1; shift ;;
        --bachata)        DL_BACHATA=1; shift ;;
        --breakdance)     DL_BREAKDANCE=1; shift ;;
        --ballroom)       DL_BALLROOM=1; shift ;;
        --all-dances)     DL_WESTCOASTSWING=1; DL_BACHATA=1; DL_BREAKDANCE=1; DL_BALLROOM=1; shift ;;
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

if [[ $DOWNLOAD_META -eq 0 && $DOWNLOAD_PRED -eq 0 && $DOWNLOAD_VIDEOS -eq 0 && $DOWNLOAD_VIDEOS_CRF16 -eq 0 && $DOWNLOAD_VIDEOS_CRF24 -eq 0 && $DOWNLOAD_PREVIEW -eq 0 ]]; then
    echo "Error: specify at least one of --meta, --pred, --videos, --videos-crf16, --videos-crf24, --preview" >&2
    exit 1
fi

if [[ $DL_WESTCOASTSWING -eq 0 && $DL_BACHATA -eq 0 && $DL_BREAKDANCE -eq 0 && $DL_BALLROOM -eq 0 ]]; then
    echo "Error: specify at least one of --westcoastswing, --bachata, --breakdance, --ballroom, --all-dances" >&2
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

# ===================================================================
# SEQUENCE DEFINITIONS (by dance type)
# ===================================================================

WESTCOASTSWING_SEQUENCES=(
    mamma_markerless_dance/050825_WestCoastSwing_CutOff_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_InsideTurn_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_LeftSidePass_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_MonkeyBar_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_OneFootSpin_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Rides_03688_03689_1_1
    mamma_markerless_dance/050825_WestCoastSwing_SlingShot_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_SugarPush_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_SugarTuck_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Telemark_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_UnderArmTurn_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_Markers_1_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_Markers_2_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_NoMarkersLose_1_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_NoMarkersLose_2_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_NoMarkersLose_3_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_NoMarkersLose_4_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_NoMarkers_1_03688_03689_1
    mamma_markerless_dance/050825_WestCoastSwing_Improv_NoMarkers_3_03688_03689_1
)

BACHATA_SEQUENCES=(
    mamma_markerless_dance/140725_Bachata_BasicMotions_1_03684_03685_1
    mamma_markerless_dance/140725_Bachata_BasicMotions_2_03684_03686_1
    mamma_markerless_dance/140725_Bachata_BasicMotions_3_03684_03687_1
    mamma_markerless_dance/140725_Bachata_Improv_1_03684_03685_1
    mamma_markerless_dance/140725_Bachata_Improv_4_03684_03685_1
    mamma_markerless_dance/140725_Bachata_Improv_6_03684_03685_1
    mamma_markerless_dance/140725_Bachata_LadyStyle_Improv_1_03685_1
    mamma_markerless_dance/140725_Bachata_LadyStyle_Improv_2_03685_1
    mamma_markerless_dance/140725_Bachata_LadyStyle_Improv_3_03685_1
    mamma_markerless_dance/280425_Bachata_ArmsLengthSideTouchTurn_03679_03680_1
    mamma_markerless_dance/280425_Bachata_CirclePalms_03679_03680_1
    mamma_markerless_dance/280425_Bachata_HeadRoll_03679_03680_1
    mamma_markerless_dance/280425_Bachata_HipBump_03679_03680_1
    mamma_markerless_dance/280425_Bachata_Improv_1_03679_03680_1
    mamma_markerless_dance/280425_Bachata_Improv_2_03679_03680_1
    mamma_markerless_dance/280425_Bachata_Improv_3_03679_03680_1
    mamma_markerless_dance/280425_Bachata_Improv_4_03679_03680_1_1
    mamma_markerless_dance/280425_Bachata_Improv_4_03679_03680_1_2
    mamma_markerless_dance/280425_Bachata_Improv_5_03679_03680_1
    mamma_markerless_dance/280425_Bachata_LeanOnLongSide_03679_03680_1
    mamma_markerless_dance/280425_Bachata_LeanOnLongSide_03679_03680_1_1
    mamma_markerless_dance/280425_Bachata_ParallelSideSteps_03679_03680_1
    mamma_markerless_dance/280425_Bachata_ParallelSideSteps_03679_03680_1_1
    mamma_markerless_dance/280425_Bachata_ParallelSideSteps_03679_03680_2_1
    mamma_markerless_dance/280425_Bachata_ParallelSideTurn_03679_03680_1
    mamma_markerless_dance/280425_Bachata_ThrowUpCrossedArm_03679_03680_1
    mamma_markerless_dance/280425_Bachata_WaistCut_03679_03680_1
    mamma_markerless_dance/280425_Bachata_WaistCut_03679_03680_1_1
    mamma_markerless_dance/280425_Bachata_WalkZigZag_03679_03680_1
    mamma_markerless_dance/280425_Bachata_WalkZigZag_03679_03680_1_1
    mamma_markerless_dance/280425_Bachata_WalkZigZag_03679_03680_2_1
    mamma_markerless_dance/280425_Bachata_shoulderTangle_03679_03680_1
)

BREAKDANCE_SEQUENCES=(
    mamma_markerless_dance/140725_Breakdance_Improv_1_03684_1
    mamma_markerless_dance/140725_Breakdance_Improv_1_03686_1
    mamma_markerless_dance/140725_Breakdance_Improv_1_03684_03686_1
    mamma_markerless_dance/140725_Breakdance_Improv_2_03684_1
    mamma_markerless_dance/140725_Breakdance_Improv_2_03684_03686_1_1
    mamma_markerless_dance/140725_Breakdance_Improv_2_03684_03686_1_2
    mamma_markerless_dance/140725_Breakdance_Improv_3_03684_1
    mamma_markerless_dance/140725_Breakdance_Improv_3_03684_03686_1_1
    mamma_markerless_dance/140725_Breakdance_Improv_4_03684_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_1_03684_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_1_03686_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_1_03684_03686_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_2_03684_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_2_03684_03686_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_2_03686_1_1_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_3_03684_03686_1_1
    mamma_markerless_dance/140725_Breakdancemarkers_Improv_3_03684_03686_1_2
)

BALLROOM_SEQUENCES=(
    mamma_markerless_dance/250903_Ballroom_2StepsSpinTurn_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_3Step_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_BackHover_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_BackSideclose_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Backward3StepCurved_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_BackwardCurvedFeather_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ChangeOfDirection_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ChangeStep1_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ChasseFromPromenade_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ChasseToTheRight_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_CheckedNatural_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ClosedPromenade_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_CurvedFeather_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_DoubleReverse_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_FallawaySlipPivot_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_FallawaySlipPivot_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_FeatherStep_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_FiveStep_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_FollowayPromenade_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_HeelPull_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Hesitation_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_HoverCross_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Improv1_Foxtrot_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Improv1_Foxtrot_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_Improv1_Quickstep_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Improv1_Quickstep_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_Improv1_SlowWaltz_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Improv1_VienneseWaltz_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Improv1_VienneseWaltz_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_Improv2_Quickstep_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_Improv2_SlowWaltz_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Kick_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_NaturalRockTurn_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_NaturalTelemark_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_NaturalTurn_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_OutsidechangetoCP_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ProgressiveChasseLeft_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ReverseTurnOpenFinish_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ReverseTurnVW_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ReverseTurn_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ReverseWeave_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_RightLunge_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_RightLunge_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_Rocks_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_SambaWhisk_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ScatterChasse_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_SlowStep_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_SpanishDrag_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_SpinTurn_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_SpinTurn_03694_03695_1_1
    mamma_markerless_dance/250903_Ballroom_StepHopx2_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_SwivelSF_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Swivel_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_ThrowawayOversway_03694_03695_1
    mamma_markerless_dance/250903_Ballroom_Whisk_03694_03695_1
)

# Build the final sequence list from selected dance types
SEQUENCES=()
[[ $DL_WESTCOASTSWING -eq 1 ]] && SEQUENCES+=("${WESTCOASTSWING_SEQUENCES[@]}")
[[ $DL_BACHATA -eq 1 ]]        && SEQUENCES+=("${BACHATA_SEQUENCES[@]}")
[[ $DL_BREAKDANCE -eq 1 ]]     && SEQUENCES+=("${BREAKDANCE_SEQUENCES[@]}")
[[ $DL_BALLROOM -eq 1 ]]       && SEQUENCES+=("${BALLROOM_SEQUENCES[@]}")

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

dance_person_count() {
    local ds_seq="$1"
    local seq_name="${ds_seq##*/}"

    if [[ "$ds_seq" != mamma_markerless_dance/*Breakdance* && "$ds_seq" != mamma_markerless_dance/*Breakdancemarkers* ]]; then
        echo "2"
        return 0
    fi

    if [[ "$seq_name" =~ _[0-9]{5}(_[0-9]+){1,3}$ ]]; then
        echo "1"
        return 0
    fi

    if [[ "$seq_name" =~ _[0-9]{5}_[0-9]{5}(_[0-9]+){1,3}$ ]]; then
        echo "2"
        return 0
    fi

    echo "2"
    return 0
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
echo "Sequences:  ${#SEQUENCES[@]}"
echo "Cameras:    ${#CAMERAS[@]}"
echo "Dances:     $( [[ $DL_WESTCOASTSWING -eq 1 ]] && echo -n "westcoastswing " )$( [[ $DL_BACHATA -eq 1 ]] && echo -n "bachata " )$( [[ $DL_BREAKDANCE -eq 1 ]] && echo -n "breakdance " )$( [[ $DL_BALLROOM -eq 1 ]] && echo -n "ballroom" )"
echo "Download:   $( [[ $DOWNLOAD_META -eq 1 ]] && echo -n "meta " )$( [[ $DOWNLOAD_PRED -eq 1 ]] && echo -n "pred " )$( [[ $DOWNLOAD_VIDEOS -eq 1 ]] && echo -n "videos " )$( [[ $DOWNLOAD_VIDEOS_CRF16 -eq 1 ]] && echo -n "videos-crf16 " )$( [[ $DOWNLOAD_VIDEOS_CRF24 -eq 1 ]] && echo -n "videos-crf24 " )$( [[ $DOWNLOAD_PREVIEW -eq 1 ]] && echo -n "preview" )"
echo "Output:     ${OUTPUT_DIR}"
echo ""

downloaded=0
failed=0

for ds_seq in "${SEQUENCES[@]}"; do
    echo "=== ${ds_seq} ==="

    # --- Meta (global + per-camera) ---
    if [[ $DOWNLOAD_META -eq 1 ]]; then
        download "${ds_seq}/meta/global.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        for cam in "${CAMERAS[@]}"; do
            download "${ds_seq}/meta/${cam}.npz" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Pred (per-person, count inferred from sequence naming) ---
    if [[ $DOWNLOAD_PRED -eq 1 ]]; then
        if ! mapfile -t pred_indices < <(pred_indices_for_count "$(dance_person_count "$ds_seq")"); then
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

    # --- Videos CRF16 (per-camera) ---
    if [[ $DOWNLOAD_VIDEOS_CRF16 -eq 1 ]]; then
        for cam in "${CAMERAS[@]}"; do
            download "${ds_seq}/videos_crf16/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Videos CRF24 (per-camera) ---
    if [[ $DOWNLOAD_VIDEOS_CRF24 -eq 1 ]]; then
        for cam in "${CAMERAS[@]}"; do
            download "${ds_seq}/videos_crf24/${cam}.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done
    fi

    # --- Preview ---
    if [[ $DOWNLOAD_PREVIEW -eq 1 ]]; then
        download "${ds_seq}/preview/overlay_grid.mp4" && downloaded=$((downloaded+1)) || failed=$((failed+1))
    fi

    echo ""
done

echo "Done! Downloaded: ${downloaded}, Failed: ${failed}"
