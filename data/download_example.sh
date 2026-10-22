#!/usr/bin/env bash
#
# Download the MAMMA demo sequence (pushing_and_lifting_from_ground) from
# the public MPI-IS server. No login required: the assets live under
# /mamma/assets/download/ on download.is.tue.mpg.de and are served as
# plain HTTPS GET. After this finishes you can run the example pipeline:
#
#     bash gui/scripts/dev.sh        # GUI route, pick mamma_example in the wizard
#   OR
#     python -m inference --task configs/examples/presets/quick.yaml \
#         --capture configs/examples/captures/mamma_example.json
#
# Usage:
#   bash data/download_example.sh                        # all 4 cameras to ./data/mamma_example/
#   bash data/download_example.sh --cam A001             # just one camera
#   bash data/download_example.sh --output /scratch/data # alt output dir
#
set -euo pipefail

# ===================================================================
# Defaults
# ===================================================================

BASE_URL="https://download.is.tue.mpg.de/mamma/assets/download/dataset_examples/pushing_and_lifting_from_ground/videos_light"
# Anchor the default output to the SCRIPT's directory (which is the
# repo's data/ folder) rather than the caller's cwd, so the script
# works whether you run it from the repo root or from anywhere else.
# The in-repo capture config
# (configs/examples/captures/mamma_example.json) resolves its
# `capture_root` to <repo>/data/mamma_example, so this path matches
# what the GUI wizard and `python -m inference` will look for.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/mamma_example/pushing_and_lifting_from_ground/videos"
DEFAULT_CAMERAS=(A001 B001 C001 D001)

OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
CAMERAS=("${DEFAULT_CAMERAS[@]}")

# ===================================================================
# CLI parsing
# ===================================================================

usage() {
    cat <<EOF
Usage: bash data/download_example.sh [OPTIONS]

Downloads the MAMMA demo sequence (pushing_and_lifting_from_ground,
light-encode mp4s) from the public MPI-IS server. ~56 MB total for all
four cameras. No registration / login required.

Options:
  --cam A001 [B001 ...]   Cameras to download (default: A001 B001 C001 D001).
  --output DIR            Output directory (default: $DEFAULT_OUTPUT_DIR).
  -h, --help              Show this message.

The default --output matches what configs/examples/captures/mamma_example.json
expects, so the pipeline finds the videos with no extra wiring.
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cam)
            shift
            CAMERAS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CAMERAS+=("$1")
                shift
            done
            if [[ ${#CAMERAS[@]} -eq 0 ]]; then
                echo "Error: --cam needs at least one camera name" >&2
                exit 1
            fi
            ;;
        --output)
            [[ $# -ge 2 ]] || { echo "Error: --output needs a directory" >&2; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help) usage ;;
        *)
            echo "Unknown option: $1 (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

# ===================================================================
# Sanity: detect HTML error responses (so a future server-side rename
# or outage produces a clean failure, not a silently-bogus mp4 file).
# Mirrors the pattern from the iPhone download script.
# ===================================================================

is_html_error_response() {
    local f="$1"
    [[ -s "$f" ]] || return 0
    if head -c 256 "$f" | grep -Fqi "<!DOCTYPE html"; then return 0; fi
    if head -c 256 "$f" | grep -Fqi "<html";          then return 0; fi
    if head -c 256 "$f" | grep -Fqi "Error: File not found."; then return 0; fi
    return 1
}

is_valid_download() {
    local f="$1"
    [[ -s "$f" ]] || return 1
    if is_html_error_response "$f"; then return 1; fi
    return 0
}

# ===================================================================
# Required tools
# ===================================================================

if ! command -v wget >/dev/null 2>&1; then
    echo "Error: 'wget' not found. Install wget or use a different downloader." >&2
    exit 1
fi

# ===================================================================
# Download
# ===================================================================

mkdir -p "$OUTPUT_DIR"

echo "MAMMA example: pushing_and_lifting_from_ground (light videos)"
echo "  Cameras: ${CAMERAS[*]}"
echo "  Output:  $OUTPUT_DIR"
echo "  Source:  $BASE_URL/"
echo ""

ok=0
fail=0

for cam in "${CAMERAS[@]}"; do
    target="$OUTPUT_DIR/${cam}.mp4"
    url="$BASE_URL/${cam}.mp4"

    if is_valid_download "$target"; then
        echo "  [skip] ${cam}.mp4 (already present, $(stat -c%s "$target" 2>/dev/null || echo "?") bytes)"
        ok=$((ok+1))
        continue
    fi

    # --continue lets a partially-downloaded file resume across retries.
    if wget --continue --no-verbose --show-progress \
            -O "$target" "$url"; then
        if is_valid_download "$target"; then
            echo "  [ok]   ${cam}.mp4 ($(stat -c%s "$target") bytes)"
            ok=$((ok+1))
            continue
        fi
    fi

    rm -f "$target"
    echo "  [FAIL] ${cam}.mp4 (see ${url})"
    fail=$((fail+1))
done

echo ""
echo "Done. OK: ${ok}, Failed: ${fail}"
exit $(( fail > 0 ))
