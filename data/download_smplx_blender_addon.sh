#!/bin/bash
#
# Download the SMPL-X Blender add-on (code + the locked-head .blend model) used
# by the SMPL-X exporter (ma_export) to produce rigged FBX / Alembic / BVH / USD.
# Requires registration at https://smpl-x.is.tue.mpg.de/ (the SMPL-X account).
#
# Same download.php wire format as download_smplx_locked_head.sh (domain=smplx).
# The zip is SELF-CONTAINED: add-on code + smplx_model_lh_*.blend + hand poses +
# regressors. We do NOT install it into Blender: the exporter imports it as a
# library and calls its operators (object.smplx_export_fbx / _alembic).
#
# If the zip is already in data/ (e.g. you downloaded it manually), this just
# extracts it (no credentials needed).
#
# Usage:
#   bash data/download_smplx_blender_addon.sh
#   bash data/download_smplx_blender_addon.sh --output /scratch/data
#   bash data/download_smplx_blender_addon.sh --help
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
REMOTE_SFILE="smplx_blender_addon-1.0.3-20260511.zip"

usage() {
    echo "Usage: bash data/download_smplx_blender_addon.sh [OPTIONS]"
    echo ""
    echo "Downloads ${REMOTE_SFILE} (~420 MB, includes the locked-head .blend)"
    echo "from download.is.tue.mpg.de and extracts it into"
    echo "<output>/blender_addon/smplx_blender_addon/."
    echo ""
    echo "Options:"
    echo "  --output DIR    Output directory (default: <repo>/data)"
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

BASE_URL="https://download.is.tue.mpg.de/download.php?domain=smplx&resume=1"
DEST_DIR="${OUTPUT_DIR}/blender_addon"
ADDON_DIR="${DEST_DIR}/smplx_blender_addon"
# Prefer a zip already sitting in data/ (manual download), else fetch to a temp.
PREEXISTING_ZIP="${OUTPUT_DIR}/${REMOTE_SFILE}"
ZIP_PATH="${PREEXISTING_ZIP}"

urle () {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

is_valid_download() {
    local f="$1"
    [[ -f "$f" && -s "$f" ]] || return 1
    head -c 256 "$f" | grep -Fqi "Error: File not found." && return 1
    head -c 256 "$f" | grep -Fqi "<!DOCTYPE html" && return 1
    head -c 256 "$f" | grep -Fqi "<html" && return 1
    return 0
}

# Skip if already extracted (look for the .blend model as the marker).
if compgen -G "${ADDON_DIR}/data/*.blend" >/dev/null 2>&1; then
    echo "  [skip] blender_addon/smplx_blender_addon (already extracted)"
    echo "Done! Downloaded: 0, Failed: 0"
    exit 0
fi

if [[ -f "$PREEXISTING_ZIP" ]] && is_valid_download "$PREEXISTING_ZIP"; then
    echo "  [ok] using existing zip: ${PREEXISTING_ZIP} (no download needed)"
else
    echo ""
    echo "You need to register at https://smpl-x.is.tue.mpg.de/"
    if [[ -n "${SMPLX_USERNAME:-}" && -n "${SMPLX_PASSWORD:-}" ]]; then
        username=$(urle "$SMPLX_USERNAME"); password=$(urle "$SMPLX_PASSWORD")
    else
        read -r -p "Username (SMPL-X): " SMPLX_USERNAME
        read -r -s -p "Password (SMPL-X): " SMPLX_PASSWORD; echo ""
        username=$(urle "$SMPLX_USERNAME"); password=$(urle "$SMPLX_PASSWORD")
    fi
    ZIP_PATH="${OUTPUT_DIR}/${REMOTE_SFILE}"
    echo "Download:   ${REMOTE_SFILE} (domain=smplx)"
    if ! wget --post-data "username=$username&password=$password" \
              "${BASE_URL}&sfile=${REMOTE_SFILE}" -O "$ZIP_PATH" \
              --continue --no-verbose --show-progress 2>&1; then
        rm -f "$ZIP_PATH"; echo "  [FAIL] ${REMOTE_SFILE} (network / auth)" >&2; exit 1
    fi
    if ! is_valid_download "$ZIP_PATH"; then
        rm -f "$ZIP_PATH"
        echo "  [FAIL] server returned an error page; check credentials and SMPL-X license acceptance" >&2
        exit 1
    fi
fi

mkdir -p "$DEST_DIR"
if ! unzip -q -o "$ZIP_PATH" -d "$DEST_DIR"; then
    echo "  [FAIL] could not extract ${ZIP_PATH} into ${DEST_DIR}" >&2; exit 1
fi

if ! compgen -G "${ADDON_DIR}/data/*.blend" >/dev/null 2>&1; then
    echo "  [FAIL] extracted, but no .blend model found under ${ADDON_DIR}/data/" >&2; exit 1
fi

echo "  [ok] blender_addon/smplx_blender_addon (code + .blend model)"
echo ""
echo "Done! Downloaded: 1, Failed: 0"
