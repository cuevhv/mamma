#!/bin/bash
#
# Download MAMMA synthetic training webdatasets from the MPI server.
# Requires registration at https://mamma.is.tue.mpg.de/
#
# ALTERNATIVE source. The default (faster) way is Hugging Face — see
# download_mamma_syn_hf.sh. Use this script if you prefer a MAMMA account.
#
# Usage:
#   bash data/download_mamma_syn_wd.sh --interactions                # multi-person datasets
#   bash data/download_mamma_syn_wd.sh --singles                     # single-person datasets
#   bash data/download_mamma_syn_wd.sh --hands                       # hand datasets
#   bash data/download_mamma_syn_wd.sh --interactions --singles      # combine
#   bash data/download_mamma_syn_wd.sh --all                         # everything
#   bash data/download_mamma_syn_wd.sh --help
#
set -euo pipefail

# ===================================================================
# CLI ARGUMENT PARSING
# ===================================================================

# Anchor OUTPUT_DIR to the script's directory (the repo's data/ folder)
# rather than the caller's cwd, so files always land in <repo>/data/mammasyn/
# regardless of where the user runs the script from. This matches the
# training `dataset_path` default (data/mammasyn) in
# landmarks/configs/train/models_2d/config.yaml, so each dataset lands at
# data/mammasyn/<dataset>/ exactly where the BEDLAM_WD loader expects it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/mammasyn"
DL_INTERACTIONS=0
DL_SINGLES=0
DL_HANDS=0

usage() {
    echo "Usage: bash data/download_mamma_syn_wd.sh [OPTIONS]"
    echo ""
    echo "At least one dataset group is required."
    echo ""
    echo "Dataset groups:"
    echo "  --interactions  Harmony4D, Inter-X, InteractionCouple, LatinDance10"
    echo "  --singles       BEDLAM, MoYo"
    echo "  --hands         InterHand"
    echo "  --all           All of the above"
    echo ""
    echo "Options:"
    echo "  --output DIR    Output directory (default: <repo>/data/mammasyn)"
    echo "  -h, --help      Show this help message"
    exit 0
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactions) DL_INTERACTIONS=1; shift ;;
        --singles)      DL_SINGLES=1; shift ;;
        --hands)        DL_HANDS=1; shift ;;
        --all)          DL_INTERACTIONS=1; DL_SINGLES=1; DL_HANDS=1; shift ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1 (use --help for usage)" >&2; exit 1 ;;
    esac
done

if [[ $DL_INTERACTIONS -eq 0 && $DL_SINGLES -eq 0 && $DL_HANDS -eq 0 ]]; then
    echo "Error: specify at least one of --interactions, --singles, --hands, --all" >&2
    exit 1
fi

# ===================================================================
# DATASET DEFINITIONS
# ===================================================================

INTERACTIONS_DATASETS=(
    harmony4d_train_1_NC_200_00_contact
    inter-x_train_close_1_NC_200_00_contact
    inter-x_train_close_1_NC_200_01_contact
    inter-x_train_close_1_NC_200_02_contact
    interactions_couple_1_C_200_00_contact
    interactions_couple_close_1_C_200_00_contact
    latindance10_1_NC_entire_dataset_00_contact
    latindance10_1_NC_entire_dataset_01_contact
    latindance10_1_NC_entire_dataset_02_contact
)

SINGLES_DATASETS=(
    b1_all_2-6_C_200_00
    b1_all_2-6_C_200_01
    BEDLAM_MASKS_WD
    moyo_4-6_C_200_00
)

HANDS_DATASETS=(
    interhand_2-6_NC_200_00
    interhand_2-6_NC_200_01
)

# Build the list of datasets to download
DATASETS=()
[[ $DL_INTERACTIONS -eq 1 ]] && DATASETS+=("${INTERACTIONS_DATASETS[@]}")
[[ $DL_SINGLES -eq 1 ]]      && DATASETS+=("${SINGLES_DATASETS[@]}")
[[ $DL_HANDS -eq 1 ]]        && DATASETS+=("${HANDS_DATASETS[@]}")

BASE_URL="https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"
# Remote layout keeps the training_webdataset/ segment; locally we strip it so
# datasets land directly under data/mammasyn/<dataset>/ (see OUTPUT_DIR above).
REMOTE_ROOT="datasets/training_webdataset"

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
            --no-check-certificate --continue --quiet --show-progress 2>&1; then
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
# Download summary
# -------------------------------------------------------------------
echo ""
echo "Datasets:   ${#DATASETS[@]}"
echo "Groups:     $( [[ $DL_INTERACTIONS -eq 1 ]] && echo -n "interactions " )$( [[ $DL_SINGLES -eq 1 ]] && echo -n "singles " )$( [[ $DL_HANDS -eq 1 ]] && echo -n "hands" )"
echo "Output:     ${OUTPUT_DIR}"
echo ""
echo "Note: each dataset contains many .tar and .json files."
echo "The script downloads the file list first, then fetches each file."
echo ""

# -------------------------------------------------------------------
# Download each dataset
# -------------------------------------------------------------------
# Each dataset dir contains sequence subdirs with .tar and .json files.
# We first download the tar_train_list.txt to know which files to get,
# then download each file individually.
# -------------------------------------------------------------------

downloaded=0
failed=0

for ds in "${DATASETS[@]}"; do
    echo "=== ${ds} ==="

    # Download the file list
    download "${ds}/tar_train_list.txt" && downloaded=$((downloaded+1)) || failed=$((failed+1))
    download "${ds}/train_data.txt" && downloaded=$((downloaded+1)) || true
    download "${ds}/get_dataset_list.sh" && downloaded=$((downloaded+1)) || true

    # If we got the tar list, use it to download all .tar files
    listfile="${OUTPUT_DIR}/${ds}/tar_train_list.txt"
    if [[ -f "$listfile" ]]; then
        while IFS= read -r tarpath; do
            [[ -z "$tarpath" ]] && continue
            download "${ds}/${tarpath}" && downloaded=$((downloaded+1)) || failed=$((failed+1))
        done < "$listfile"
    else
        echo "  [WARN] Could not retrieve file list for ${ds}, skipping."
    fi

    echo ""
done

echo "Done! Downloaded: ${downloaded}, Failed: ${failed}"
