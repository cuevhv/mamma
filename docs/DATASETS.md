# MAMMA datasets

The MAMMA project releases several dataset collections. The **MammaSyn**
synthetic training data is published on Hugging Face — the default, fastest
download (see [MammaSyn (training)](#mammasyn-training) below). The captures
and evaluation data are fetched with the account-based scripts under
[`data/`](../data/): register a free account on the project page
(<https://mamma.is.tue.mpg.de/>) — sign up, click the confirmation link in
the email you receive, then run any account-based script and supply your
credentials when prompted (or export them once per session):

```bash
export MAMMA_USERNAME='your_email'
export MAMMA_PASSWORD='your_password'
```

All scripts download into `<repo>/data/` by default (anchored to the
script's location, so they work regardless of cwd), which matches
the layout the shipped capture configs in
[`configs/examples/captures/`](../configs/examples/captures/) expect.
Use `--output DIR` to override.

> **Note:** the project team is still verifying that all data is
> ready for download — broken links / missing files are being
> corrected on a rolling basis.

---

## Scripts

| Script | Dataset | Minutes | FPS |
|---|---|---|---|
| [`data/download_mamma_dance.sh`](../data/download_mamma_dance.sh) | **Markerless Dance** (captured with MAMMA) — 123 dance sequences (West Coast Swing, Bachata, Breakdance, Ballroom), 32 cameras | 67 | 30 |
| [`data/download_mamma_multi_people.sh`](../data/download_mamma_multi_people.sh) | **Markerless Multi-People** (captured with MAMMA) — 34 interaction sequences with 3–6 people, 32 cameras | 8 | 30 |
| [`data/download_mamma_iphone.sh`](../data/download_mamma_iphone.sh) | **Markerless iPhone Captures** (captured with MAMMA) — 42 sequences (16 indoors + 26 outdoors), 4 iPhone cameras | 10 | 30 |
| [`data/download_mamma_eval.sh`](../data/download_mamma_eval.sh) | **MammaEval-Singles**, **MammaEval-Dance**, **MammaEval-Extra** — 52 evaluation sequences (22 singles + 18 dance + 12 extra), 16 or 32 cameras | 24 | 30 |
| [`data/download_mamma_syn_hf.sh`](../data/download_mamma_syn_hf.sh) | **MammaSyn-Interactions**, **MammaSyn-Singles**, **MammaSyn-Hands** — synthetic training data, each scene rendered from 8 views, WebDataset format. From [Hugging Face](https://huggingface.co/datasets/Intelligent-Systems/MammaSyn) by default; [`download_mamma_syn_wd.sh`](../data/download_mamma_syn_wd.sh) is the MAMMA-account alternative. | varies | 6 |

All datasets use **IOI** Victorem/Volucam cameras except the
**iPhone** captures (4 iPhones).

Run any script with `--help` to see the full flag list.

---

## Data types

Each script lets you pick which asset types to download:

| Flag | Description | Total size | Available in |
|---|---|---|---|
| `--gt` | Ground-truth SMPL-X parameters (MoSh++) and camera parameters (intrinsics, extrinsics). | 34 GB | Eval |
| `--meta` | Sequence metadata (frame count, FPS) and camera parameters (intrinsics, extrinsics). | 10 MB | Markerless |
| `--pred` | SMPL-X body parameters (poses, shape, translation) estimated by MAMMA. | 197 MB | Markerless |
| `--videos*` | Multi-view videos. See encoding tables below. | 5.6 TB | All |
| `--masks` | Per-frame binary segmentation masks per person (SAM). Tar archives of PNGs, one per camera. | 20 GB | Eval |
| `--markers` | Vicon markers (37 held-out) + MoSh++ baseline marker predictions with labels. | 38 MB | Eval-Extra |
| `--preview` | Pre-rendered grid videos: SMPL-X mesh overlay + mask overlay. For quick visual inspection. | 14 GB | All |

---

## Video encoding

All video variants contain the same frames at the same resolution
and 30 fps; they differ in codec / compression.

### IOI captures (Victorem & Volucam)

4K (4112×3008) for MammaEval-Singles, 2K (~2048×1504) for all
other IOI datasets.

| Flag | Codec | Settings | Total size |
|---|---|---|---|
| – | PNG | Source images (not available for download) | ~13 TB |
| `--videos` | H.264 | CRF 5, preset `veryslow`, `yuv444p` | 5.6 TB |
| `--videos-crf16` | H.265 | CRF 16, preset `slow`, `yuv444p` (~36× smaller) | 155 GB |
| `--videos-crf24` | H.265 | CRF 24, preset `slow`, `yuv444p` (~250× smaller) | 22 GB |

### iPhone captures

4K (3840×2160).

| Flag | Codec | Settings | Total size |
|---|---|---|---|
| – | MOV | `.MOV` with embedded timecode, 60 fps (not available for download) | ~500 GB |
| `--videos` | H.265 | CRF 16, preset `veryslow`, `yuv444p` | 17 GB |
| `--videos-light` | H.265 | CRF 24, preset `slow`, `yuv444p` (~9× smaller) | 2 GB |

---

## Usage

Use `--ioi 01 02 …` to restrict to specific IOI cameras.

### Markerless Dance

Requires at least one data flag and one dance style (`--westcoastswing`,
`--bachata`, `--breakdance`, `--ballroom`, or `--all-dances`):

```bash
# Metadata, predictions, and videos for all dances
# (use --videos-crf16 or --videos-crf24 for smaller downloads)
bash data/download_mamma_dance.sh --meta --pred --videos --all-dances

# Only bachata predictions for specific cameras
bash data/download_mamma_dance.sh --pred --bachata --ioi 01 05 10

# Preview visualizations
bash data/download_mamma_dance.sh --preview --all-dances
```

### Markerless Multi-People

```bash
# Metadata, predictions, and videos
# (use --videos-crf16 or --videos-crf24 for smaller downloads)
bash data/download_mamma_multi_people.sh --meta --pred --videos
```

### Markerless iPhone

Use `--indoors` and/or `--outdoors` to select a subset (both by
default). `--cam A001 B001` restricts to specific cameras (all 4
by default).

```bash
# Metadata, predictions, and videos for all sequences
# (use --videos-light for smaller downloads)
bash data/download_mamma_iphone.sh --meta --pred --videos

# Only indoor sequences
bash data/download_mamma_iphone.sh --meta --pred --videos --indoors

# Preview visualizations for outdoor sequences
bash data/download_mamma_iphone.sh --preview --outdoors
```

### MammaEval

GT SMPL-X params were created from MoSh++ and 73 Vicon markers
(FrontWaist10Fingers Vicon template). MammaEval-Singles and
MammaEval-Dance use the v_template (created from 3D body scans);
MammaEval-Extra uses predicted betas from MoSh++.

`--markers` is only available for MammaEval-Extra sequences.

```bash
# Ground truth and videos
# (use --videos-crf16 or --videos-crf24 for smaller downloads)
bash data/download_mamma_eval.sh --gt --videos

# Ground truth, masks, and Vicon markers (37 held-out) for specific cameras
bash data/download_mamma_eval.sh --gt --masks --markers --ioi 01 02 03
```

<a id="mammasyn-training"></a>
### MammaSyn (training)

Synthetic SMPL-X renders (WebDataset `.tar` shards), grouped as:

- `--interactions` — Harmony4D, Inter-X, InteractionCouple, LatinDance10 (~2.71 TiB)
- `--singles` — BEDLAM, MoYo (~2.67 TiB)
- `--hands` — InterHand (~0.64 TiB)
- `--all` — all of the above (~6.0 TiB)

#### Default: Hugging Face

MammaSyn is a **gated** Hugging Face dataset
(<https://huggingface.co/datasets/Intelligent-Systems/MammaSyn>). One-time setup:

1. Create a free Hugging Face account.
2. On the dataset page, click **Agree and access** to accept the license —
   access is granted automatically.
3. Authenticate: `hf auth login` (or `export HF_TOKEN=hf_...`). The `hf` CLI
   ships with `huggingface_hub`, included in the project `mamma` env.

Then download one or more groups:

```bash
bash data/download_mamma_syn_hf.sh --interactions
bash data/download_mamma_syn_hf.sh --singles
bash data/download_mamma_syn_hf.sh --hands
bash data/download_mamma_syn_hf.sh --all
bash data/download_mamma_syn_hf.sh --hands --dry-run   # preview, fetch nothing
```

Or call the `hf` CLI directly (note this keeps the `MammaSyn-<Group>/` prefix —
the script strips it for you):

```bash
hf download Intelligent-Systems/MammaSyn --repo-type dataset \
    --include "MammaSyn-Hands/*" --local-dir ./MammaSyn
```

`hf` always recreates the repo path (`MammaSyn-<Group>/<dataset>/…`) under
`--local-dir` and can't drop the group prefix, so the script downloads into a
hidden, resumable staging dir (`data/.mammasyn_hf/`, beside `data/mammasyn/`) and
then **moves** each dataset into place as a real directory at
`data/mammasyn/<dataset>/` — matching the loader and the account-based script.
Staging is removed once the download completes. For very fast links you can opt
into `export HF_XET_HIGH_PERFORMANCE=1` (off by default). In the GUI, the
*MAMMA Datasets → MammaSyn (Hugging Face)* panel does the same and shows your
login status, an optional pasted token, and per-file progress.

#### Alternative: MAMMA account

The same datasets are also served from the MAMMA download server for users who
prefer a MAMMA account (see the top of this page for credential setup):

```bash
bash data/download_mamma_syn_wd.sh --interactions
bash data/download_mamma_syn_wd.sh --singles
bash data/download_mamma_syn_wd.sh --hands
bash data/download_mamma_syn_wd.sh --all
```

---

## Notes

- Existing valid files are skipped on rerun.
- To download only specific sequences, open the script and comment
  out unwanted entries in the `SEQUENCES` array near the top.
- MAMMA uses **SMPL-X with removed head bun** (NPZ, 392 MB,
  neutral gender). Download it from
  <https://smpl-x.is.tue.mpg.de/>.
- SMPL-X parameters such as `gender`, `flat_hand_mean`, and
  whether a `v_template` is used may vary per dataset. These are
  stored in `gt/global.npz` (eval datasets) or
  `pred/params_XX.npz` (markerless datasets).

---

## Visualizing MAMMA SMPL-X predictions in Blender

The Markerless Dance and Multi-People datasets include SMPL-X
MAMMA predictions in `pred/params_XX.npz` (one file per person).
To view them:

1. Download the **SMPL-X Blender add-on** from
   <https://smpl-x.is.tue.mpg.de/> (requires registration).
2. In Blender: **Edit → Preferences → Add-ons → Install** and
   select the downloaded `.zip`.
3. Load a `params_XX.npz` file to create an animated SMPL-X mesh
   for that person. Set **Hand Pose Reference** to "Relaxed" and
   orientation format to "AMASS".
