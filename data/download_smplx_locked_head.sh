#!/bin/bash
#
# Download the SMPL-X locked-head body model used by ma_3d.
# Requires registration at https://smpl-x.is.tue.mpg.de/, the
# SMPL-X account (separate from the MAMMA account that the dataset
# scripts use).
#
# Same download.php wire format as the dataset scripts
# (download_mamma_dance.sh, ...), but the domain is "smplx" (not "mamma")
# and the file is a zip extracted in place after download.
#
# Usage:
#   bash data/download_smplx_locked_head.sh
#   bash data/download_smplx_locked_head.sh --output /scratch/data
#   bash data/download_smplx_locked_head.sh --help
#
set -euo pipefail

# ===================================================================
# CLI ARGUMENT PARSING
# ===================================================================

# Anchor OUTPUT_DIR to the script's directory (the repo's data/ folder)
# rather than the caller's cwd, so the model lands where inference/env.py
# expects it (data/body_models/smplx_locked_head/) regardless of where
# the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"

usage() {
    echo "Usage: bash data/download_smplx_locked_head.sh [OPTIONS]"
    echo ""
    echo "Downloads smplx_lockedhead_20230207.zip (~95 MB) from"
    echo "download.is.tue.mpg.de and extracts it into"
    echo "<output>/body_models/smplx_locked_head/."
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

REMOTE_SFILE="smplx_lockedhead_20230207.zip"
BASE_URL="https://download.is.tue.mpg.de/download.php?domain=smplx&resume=1"

DEST_DIR="${OUTPUT_DIR}/body_models/smplx_locked_head"
ZIP_PATH="${OUTPUT_DIR}/_smplx_locked_head_download.zip"

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
if [[ -d "$DEST_DIR" ]] && [[ -n "$(ls -A "$DEST_DIR" 2>/dev/null)" ]]; then
    echo "  [skip] body_models/smplx_locked_head (already populated)"
    echo ""
    echo "Done! Downloaded: 0, Failed: 0"
    exit 0
fi

load_smplx_credentials

echo ""
echo "Download:   ${REMOTE_SFILE} (domain=smplx)"
echo "Output:     ${DEST_DIR}"
echo ""

mkdir -p "$(dirname "$ZIP_PATH")"

if ! wget --post-data "username=$username&password=$password" \
          "${BASE_URL}&sfile=${REMOTE_SFILE}" \
          -O "$ZIP_PATH" \
          --continue --no-verbose --show-progress 2>&1; then
    rm -f "$ZIP_PATH"
    echo "  [FAIL] ${REMOTE_SFILE} (network / auth)"
    exit 1
fi

if ! is_valid_download "$ZIP_PATH"; then
    rm -f "$ZIP_PATH"
    echo "  [FAIL] ${REMOTE_SFILE} (server returned an error page; check credentials and SMPL-X license acceptance)"
    exit 1
fi

mkdir -p "$DEST_DIR"
if ! unzip -q -o "$ZIP_PATH" -d "$DEST_DIR"; then
    echo "  [FAIL] could not extract ${ZIP_PATH} into ${DEST_DIR}"
    exit 1
fi

# The MPI zip wraps the model in an extra top-level directory
# (smplx_lockedhead_20230207.zip → models_lockedhead/smplx/SMPLX_*.npz),
# but smplx.create() in ma_3d expects smplx/ directly under the
# --smplx-models path. Flatten one level if the wrapper is present.
if [[ ! -d "$DEST_DIR/smplx" ]]; then
    nested_smplx="$(find "$DEST_DIR" -mindepth 2 -maxdepth 2 -type d -name smplx -print -quit)"
    if [[ -n "$nested_smplx" ]]; then
        wrapper="$(dirname "$nested_smplx")"
        mv "$wrapper"/* "$DEST_DIR"/
        rmdir "$wrapper"
    fi
fi

rm -f "$ZIP_PATH"

echo "  [ok] body_models/smplx_locked_head"
echo ""
echo "Done! Downloaded: 1, Failed: 0"
