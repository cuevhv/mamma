#!/bin/bash
#
# Download the MAMMA model files needed to run inference from the MPI server:
#   * the trained MammaNet landmark detector checkpoint, and
#   * the downsampled-SMPL-X-vertex matrix.
# Requires registration at https://mamma.is.tue.mpg.de/
#
# Same MAMMA account + same download.php wire format as the dataset scripts
# (download_mamma_dance.sh, download_mamma_iphone.sh, ...). The only
# difference is the remote top-level dir: the model files live under
# weights/ and assets/, not the datasets' "datasets/" root. Local and
# remote paths therefore differ, so download() takes both explicitly.
#
# Usage:
#   bash data/download_mamma_weights.sh --all                 # ckpt + verts
#   bash data/download_mamma_weights.sh --ckpt                # landmark detector only
#   bash data/download_mamma_weights.sh --verts               # downsampled verts only
#   bash data/download_mamma_weights.sh --help
#
set -euo pipefail

# ===================================================================
# CLI ARGUMENT PARSING
# ===================================================================

# Anchor OUTPUT_DIR to the script's directory (the repo's data/ folder)
# rather than the caller's cwd, so files always land where inference/env.py
# expects them (data/weights/ma_2d/... and data/body_models/...) regardless
# of where the user invokes the script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
DL_CKPT=0
DL_VERTS=0

usage() {
    echo "Usage: bash data/download_mamma_weights.sh [OPTIONS]"
    echo ""
    echo "At least one file group is required."
    echo ""
    echo "File groups:"
    echo "  --ckpt    MammaNet landmark detector (-> weights/ma_2d/mamma_mask_full_cvpr.ckpt, ~1.6 GB)"
    echo "  --verts   Downsampled SMPL-X vertices (-> body_models/downsampled_verts/verts_512.pkl, ~20 MB)"
    echo "  --all     Both of the above"
    echo ""
    echo "Options:"
    echo "  --output DIR    Output directory (default: <repo>/data, where the pipeline expects it)"
    echo "  -h, --help      Show this help message"
    exit 0
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)    DL_CKPT=1; shift ;;
        --verts)   DL_VERTS=1; shift ;;
        --all)     DL_CKPT=1; DL_VERTS=1; shift ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1 (use --help for usage)" >&2; exit 1 ;;
    esac
done

if [[ $DL_CKPT -eq 0 && $DL_VERTS -eq 0 ]]; then
    echo "Error: specify at least one of --ckpt, --verts, --all" >&2
    exit 1
fi

# ===================================================================
# END OF SETTINGS
# ===================================================================

BASE_URL="https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"

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
# download <remote_sfile> <local_relpath>
# The dataset scripts share one REMOTE_ROOT and mirror the remote layout
# locally; here the two files come from different remote dirs (weights/,
# assets/) and land in the pipeline's own data/ layout, so the remote sfile
# and the local destination are passed separately.
download () {
    local remote_sfile="$1"
    local local_relpath="$2"
    local outpath="${OUTPUT_DIR}/${local_relpath}"

    if mamma_is_valid_download "$outpath"; then
        echo "  [skip] ${local_relpath}"
        return 0
    fi

    mkdir -p "$(dirname "$outpath")"
    if wget --post-data "username=$username&password=$password" \
            "${BASE_URL}&sfile=${remote_sfile}" \
            -O "$outpath" \
            --continue --no-verbose --show-progress 2>&1; then
        if mamma_is_valid_download "$outpath"; then
            echo "  [ok] ${local_relpath}"
            return 0
        fi
    fi

    rm -f "$outpath"
    echo "  [FAIL] ${local_relpath}"
    return 1
}

# -------------------------------------------------------------------
# Download
# -------------------------------------------------------------------
echo ""
echo "Download:   $( [[ $DL_CKPT -eq 1 ]] && echo -n "ckpt " )$( [[ $DL_VERTS -eq 1 ]] && echo -n "verts" )"
echo "Output:     ${OUTPUT_DIR}"
echo ""

downloaded=0
failed=0

if [[ $DL_CKPT -eq 1 ]]; then
    download "weights/mamma_mask_full_cvpr.ckpt" \
             "weights/ma_2d/mamma_mask_full_cvpr.ckpt" \
        && downloaded=$((downloaded+1)) || failed=$((failed+1))
fi

if [[ $DL_VERTS -eq 1 ]]; then
    download "mamma_assets/verts_512.pkl" \
             "body_models/downsampled_verts/verts_512.pkl" \
        && downloaded=$((downloaded+1)) || failed=$((failed+1))
fi

echo ""
echo "Done! Downloaded: ${downloaded}, Failed: ${failed}"
