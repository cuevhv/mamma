#!/bin/bash
#
# Download the VPoser v2.05 variational human pose prior used *optionally*
# by ma_3d during SMPL-X fitting (the vposer_recon_loss manifold prior).
# Requires registration at https://smpl-x.is.tue.mpg.de/ — this is the
# SMPL-X account (the same gate as download_smplx_locked_head.sh, separate
# from the MAMMA account that the dataset scripts use).
#
# Same download.php wire format as the other SMPL-X/MAMMA scripts, with
# domain "smplx"; the file is a zip extracted in place after download.
#
# This asset is OPTIONAL — only configs that enable vposer_recon_loss need
# it (see optimization/.../config_*_occlusion_vposer.yaml). The default
# pipeline runs without it.
#
# Usage:
#   bash data/download_vposer.sh
#   bash data/download_vposer.sh --output /scratch/data
#   bash data/download_vposer.sh --help
#
set -euo pipefail

# ===================================================================
# CLI ARGUMENT PARSING
# ===================================================================

# Anchor OUTPUT_DIR to the script's directory (the repo's data/ folder)
# rather than the caller's cwd, so the model lands where the optim loss
# expects it (data/body_models/vposer/V02_05/, overridable via
# MAMMA_VPOSER_DIR) regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"

usage() {
    echo "Usage: bash data/download_vposer.sh [OPTIONS]"
    echo ""
    echo "Downloads V02_05.zip (VPoser v2.05, ~79 MB download / ~170 MB unpacked)"
    echo "from download.is.tue.mpg.de and extracts it into"
    echo "<output>/body_models/vposer/ (giving <output>/body_models/vposer/V02_05/)."
    echo ""
    echo "Optional asset: only needed by ma_3d configs that enable the"
    echo "vposer_recon_loss pose prior; the default pipeline runs without it."
    echo ""
    echo "Options:"
    echo "  --output DIR    Output directory (default: <repo>/data, where the pipeline expects it)"
    echo "  -h, --help      Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1 (use --help for usage)" >&2; exit 1 ;;
    esac
done

# ===================================================================
# END OF SETTINGS
# ===================================================================

REMOTE_SFILE="V02_05.zip"
BASE_URL="https://download.is.tue.mpg.de/download.php?domain=smplx&resume=1"

# The zip wraps the model in a top-level V02_05/ directory, so we extract
# into the vposer/ parent and the model lands at vposer/V02_05/ — exactly
# the path optimization/losses/losses.py loads by default.
VPOSER_DIR="${OUTPUT_DIR}/body_models/vposer"
EXPECTED_DIR="${VPOSER_DIR}/V02_05"
ZIP_PATH="${OUTPUT_DIR}/_vposer_download.zip"

urle () {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

is_error_response() {
    local file_path="$1"
    [[ -f "$file_path" ]] || return 0
    if head -c 256 "$file_path" | grep -Fqi "Error: File not found."; then return 0; fi
    if head -c 256 "$file_path" | grep -Fqi "<!DOCTYPE html"; then return 0; fi
    if head -c 256 "$file_path" | grep -Fqi "<html"; then return 0; fi
    return 1
}

is_valid_download() {
    local file_path="$1"
    [[ -f "$file_path" && -s "$file_path" ]] || return 1
    is_error_response "$file_path" && return 1
    return 0
}

load_smplx_credentials() {
    if [[ -n "${SMPLX_USERNAME:-}" && -n "${SMPLX_PASSWORD:-}" ]]; then
        username=$(urle "$SMPLX_USERNAME")
        password=$(urle "$SMPLX_PASSWORD")
        return 0
    fi

    echo ""
    echo "You need to register at https://smpl-x.is.tue.mpg.de/"
    read -r -p "Username (SMPL-X): " SMPLX_USERNAME
    read -r -s -p "Password (SMPL-X): " SMPLX_PASSWORD
    echo ""

    username=$(urle "$SMPLX_USERNAME")
    password=$(urle "$SMPLX_PASSWORD")
}

# Skip everything if the destination already looks populated.
if [[ -d "$EXPECTED_DIR" ]] && [[ -n "$(ls -A "$EXPECTED_DIR" 2>/dev/null)" ]]; then
    echo "  [skip] body_models/vposer/V02_05 (already populated)"
    echo ""
    echo "Done! Downloaded: 0, Failed: 0"
    exit 0
fi

load_smplx_credentials

echo ""
echo "Download:   ${REMOTE_SFILE} (domain=smplx, optional VPoser pose prior)"
echo "Output:     ${EXPECTED_DIR}"
echo ""

mkdir -p "$(dirname "$ZIP_PATH")"

if ! wget --post-data "username=$username&password=$password" \
          "${BASE_URL}&sfile=${REMOTE_SFILE}" \
          -O "$ZIP_PATH" \
          --no-check-certificate --continue --quiet --show-progress 2>&1; then
    rm -f "$ZIP_PATH"
    echo "  [FAIL] ${REMOTE_SFILE} (network / auth)"
    exit 1
fi

if ! is_valid_download "$ZIP_PATH"; then
    rm -f "$ZIP_PATH"
    echo "  [FAIL] ${REMOTE_SFILE} (server returned an error page — check credentials and SMPL-X license acceptance)"
    exit 1
fi

mkdir -p "$VPOSER_DIR"
if ! unzip -q -o "$ZIP_PATH" -d "$VPOSER_DIR"; then
    echo "  [FAIL] could not extract ${ZIP_PATH} into ${VPOSER_DIR}"
    exit 1
fi

# The release ships a top-level V02_05/ wrapper, so the model should now sit
# at vposer/V02_05/. If a future archive ever extracts the model files flat
# into vposer/ instead (snapshots/ + the .yaml config), tuck them into the
# V02_05/ dir the loss loads by default.
if [[ ! -d "$EXPECTED_DIR" ]] && [[ -d "$VPOSER_DIR/snapshots" ]]; then
    mkdir -p "$EXPECTED_DIR"
    shopt -s dotglob
    for item in "$VPOSER_DIR"/*; do
        [[ "$item" == "$EXPECTED_DIR" ]] && continue
        mv "$item" "$EXPECTED_DIR"/
    done
    shopt -u dotglob
fi

if [[ ! -d "$EXPECTED_DIR" ]] || [[ -z "$(ls -A "$EXPECTED_DIR" 2>/dev/null)" ]]; then
    echo "  [FAIL] extracted archive but ${EXPECTED_DIR} is missing/empty" >&2
    exit 1
fi

rm -f "$ZIP_PATH"

echo "  [ok] body_models/vposer/V02_05"
echo ""
echo "Done! Downloaded: 1, Failed: 0"
