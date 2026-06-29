#!/bin/bash
#
# Download MAMMA synthetic training webdatasets from Hugging Face.
# This is the DEFAULT, fastest way to fetch MammaSyn (xet transfers, parallel,
# resumable). For the MAMMA-account alternative, see download_mamma_syn_wd.sh.
#
# One-time setup:
#   1. Create a free Hugging Face account.
#   2. Accept the license at https://huggingface.co/datasets/Intelligent-Systems/MammaSyn
#      (access is granted automatically on acceptance).
#   3. Authenticate: `hf auth login` (or `export HF_TOKEN=hf_...`).
#      The `hf` CLI ships with huggingface_hub; it's in the project `mamma`
#      conda env. To install elsewhere: `pip install -U huggingface_hub`.
#
# Usage:
#   bash data/download_mamma_syn_hf.sh --interactions               # multi-person datasets
#   bash data/download_mamma_syn_hf.sh --singles                    # single-person datasets
#   bash data/download_mamma_syn_hf.sh --hands                      # hand datasets
#   bash data/download_mamma_syn_hf.sh --interactions --singles     # combine
#   bash data/download_mamma_syn_hf.sh --all                        # everything (~6.0 TiB)
#   bash data/download_mamma_syn_hf.sh --hands --dry-run            # show what would download
#   bash data/download_mamma_syn_hf.sh --help
#
# Tip: for very fast links you can opt into high-performance transfers with
#   export HF_XET_HIGH_PERFORMANCE=1
# (left off by default — it raises concurrency/memory and can saturate links).
#
set -euo pipefail

# ===================================================================
# CLI ARGUMENT PARSING
# ===================================================================

# Anchor OUTPUT_DIR to the script's directory (the repo's data/ folder) so
# files always land in <repo>/data/mammasyn/ regardless of where the user runs
# the script from. This matches the training `dataset_path` default
# (data/mammasyn) and the MAMMA-account script, so each dataset is reachable at
# data/mammasyn/<dataset>/ exactly where the BEDLAM_WD loader expects it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/mammasyn"
DL_INTERACTIONS=0
DL_SINGLES=0
DL_HANDS=0
DRY_RUN=0

usage() {
    echo "Usage: bash data/download_mamma_syn_hf.sh [OPTIONS]"
    echo ""
    echo "At least one dataset group is required."
    echo ""
    echo "Dataset groups:"
    echo "  --interactions  Harmony4D, Inter-X, InteractionCouple, LatinDance10 (~2.71 TiB)"
    echo "  --singles       BEDLAM, MoYo (~2.67 TiB)"
    echo "  --hands         InterHand (~0.638 TiB)"
    echo "  --all           All of the above (~6.0 TiB)"
    echo ""
    echo "Options:"
    echo "  --output DIR    Output directory (default: <repo>/data/mammasyn)"
    echo "  --dry-run       Show what would be downloaded without fetching"
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
        --dry-run)      DRY_RUN=1; shift ;;
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
# These lists mirror exactly what is published on Hugging Face (the GUI parses
# these arrays to show per-group counts). The download itself uses per-group
# globs, so the lists don't need to be kept in lock-step with the shards.

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

HF_REPO="Intelligent-Systems/MammaSyn"
# `hf` reproduces the repo path under --local-dir (MammaSyn-<Group>/<dataset>/...)
# with no way to strip the group prefix, so we download each dataset into a
# hidden, resumable staging dir beside the output and then *move* it into place
# at data/mammasyn/<dataset>/. The result is a real directory tree (no symlinks)
# matching the MAMMA-account script and the training configs. Staging lives
# next to OUTPUT_DIR (same filesystem) so the move is an instant rename, and is
# removed once everything succeeds.
STAGING="$(dirname "$OUTPUT_DIR")/.mammasyn_hf"

# `hf download` pre-creates the --local-dir tree even on --dry-run, so for a
# dry run we point it at a throwaway dir (cleaned on exit) instead of STAGING.
DL_LOCAL_DIR="$STAGING"
if [[ $DRY_RUN -eq 1 ]]; then
    DL_LOCAL_DIR="$(mktemp -d)"
    trap 'rm -rf "$DL_LOCAL_DIR"' EXIT
fi

# Selected groups → HF top-level directory (for the dry-run preview + summary).
SELECTED_GROUPS=()
[[ $DL_INTERACTIONS -eq 1 ]] && SELECTED_GROUPS+=("interactions:MammaSyn-Interactions:${#INTERACTIONS_DATASETS[@]}")
[[ $DL_SINGLES -eq 1 ]]      && SELECTED_GROUPS+=("singles:MammaSyn-Singles:${#SINGLES_DATASETS[@]}")
[[ $DL_HANDS -eq 1 ]]        && SELECTED_GROUPS+=("hands:MammaSyn-Hands:${#HANDS_DATASETS[@]}")

# Flat list of "<HF group dir>/<dataset>" entries to download (real runs).
ENTRIES=()
[[ $DL_INTERACTIONS -eq 1 ]] && for ds in "${INTERACTIONS_DATASETS[@]}"; do ENTRIES+=("MammaSyn-Interactions/${ds}"); done
[[ $DL_SINGLES -eq 1 ]]      && for ds in "${SINGLES_DATASETS[@]}";      do ENTRIES+=("MammaSyn-Singles/${ds}");      done
[[ $DL_HANDS -eq 1 ]]        && for ds in "${HANDS_DATASETS[@]}";        do ENTRIES+=("MammaSyn-Hands/${ds}");        done

# -------------------------------------------------------------------
# Preflight: hf CLI present, logged in, and has access to the gated repo
# -------------------------------------------------------------------
preflight() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "Error: the 'hf' CLI was not found." >&2
        echo "Activate the project env (conda activate mamma), or install it:" >&2
        echo "    pip install -U huggingface_hub" >&2
        exit 1
    fi
    if ! hf auth whoami >/dev/null 2>&1; then
        echo "Error: not logged in to Hugging Face." >&2
        echo "Run 'hf auth login' (or 'export HF_TOKEN=hf_...') and retry." >&2
        exit 1
    fi
    # Cheap gated-access probe. Listing/fetching a gated repo requires access,
    # so this fails clearly if the license hasn't been accepted yet.
    local probe=(download "$HF_REPO" --repo-type dataset --include "README.md" --local-dir "$DL_LOCAL_DIR")
    [[ $DRY_RUN -eq 1 ]] && probe+=(--dry-run)
    if ! hf "${probe[@]}" >/dev/null 2>&1; then
        echo "Error: no access to ${HF_REPO}." >&2
        echo "Accept the license at https://huggingface.co/datasets/${HF_REPO}" >&2
        echo "then retry (access is granted automatically on acceptance)." >&2
        exit 1
    fi
}

# -------------------------------------------------------------------
# Download one dataset into staging, then move it to data/mammasyn/<dataset>/
# -------------------------------------------------------------------
download_one() {
    local hf_dir="$1" ds="$2"
    local dest="${OUTPUT_DIR}/${ds}"
    # Skip if a real directory is already in place (a prior successful run).
    if [[ -d "$dest" && ! -L "$dest" ]]; then
        echo "  [skip] ${ds} (already present)"
        return 0
    fi
    # Replace any leftover symlink from an older version of this script.
    [[ -L "$dest" ]] && rm -f "$dest"
    echo "  [..]   ${ds}"
    hf download "$HF_REPO" --repo-type dataset --include "${hf_dir}/${ds}/*" --local-dir "$STAGING"
    mkdir -p "$OUTPUT_DIR"
    rm -rf "$dest"
    # Instant rename (staging shares OUTPUT_DIR's filesystem).
    mv "${STAGING}/${hf_dir}/${ds}" "$dest"
    echo "  [ok]   ${ds}"
}

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
preflight

total_datasets=0
for entry in "${SELECTED_GROUPS[@]}"; do
    total_datasets=$(( total_datasets + ${entry##*:} ))
done

echo ""
echo "Repo:       ${HF_REPO}"
echo "Datasets:   ${total_datasets}"
echo "Groups:     $( [[ $DL_INTERACTIONS -eq 1 ]] && echo -n "interactions " )$( [[ $DL_SINGLES -eq 1 ]] && echo -n "singles " )$( [[ $DL_HANDS -eq 1 ]] && echo -n "hands" )"
echo "Output:     ${OUTPUT_DIR}"
[[ $DRY_RUN -eq 1 ]] && echo "Mode:       DRY RUN (nothing will be written)"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    for entry in "${SELECTED_GROUPS[@]}"; do
        IFS=':' read -r gid hfdir _count <<< "$entry"
        echo "=== ${gid} (${hfdir}) ==="
        hf download "$HF_REPO" --repo-type dataset --include "${hfdir}/*" --local-dir "$DL_LOCAL_DIR" --dry-run
        echo "  [dry-run] would place each dataset at ${OUTPUT_DIR}/<dataset>/"
        echo ""
    done
    echo "Dry run complete."
    exit 0
fi

for entry in "${ENTRIES[@]}"; do
    download_one "${entry%/*}" "${entry##*/}"
done

# Everything succeeded — drop the staging scratch (resume metadata + empties).
rm -rf "$STAGING"

echo "Done! Datasets are available under ${OUTPUT_DIR}/<dataset>/."
