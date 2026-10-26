# Running on an HTCondor cluster

MAMMA can submit a pipeline run to a remote compute cluster as an HTCondor DAG, from the GUI or the
command line, and track it from your machine. This is optional: by default every step runs locally.

The feature is built for **HTCondor** clusters reached **over SSH**. It works only if you have an
account on such a cluster; it does not create or rent compute anywhere.

## Requirements

- **HTCondor.** The cluster schedules jobs with HTCondor, and its login node provides
  `condor_submit_dag`, `condor_q` and `condor_rm` (or `condor_submit_dag_bid` for sites that use
  bidding). Other schedulers (Slurm, PBS, LSF, Kubernetes) are not supported.
- **SSH access.** You can log in to the cluster's login node from the machine running MAMMA, with
  your SSH keys or ssh-agent. Clusters that need a password or 2FA work through a one-time
  `python -m inference cluster connect <profile>` session in a terminal. MAMMA never asks for or
  stores a password.
- **A shared workspace.** A folder of yours on the cluster's shared filesystem, visible from the
  login node and every execute node. MAMMA copies its code there (rsync) and writes run configs,
  logs and outputs under it.
- **The pipeline environment on the execute nodes.** Usually the MAMMA Apptainer image (see
  [docker/README.md](../docker/README.md)); a conda env or Docker image the nodes can use also
  works. The image's CUDA extensions need GPUs with compute capability 8.0 or newer.
- **Data on the cluster.** The code sync skips `data/` and `output/`. Put the model weights and body
  models in `<workspace>/repo/data/` (same layout as in [INSTALL.md](INSTALL.md)), and point the
  capture at footage and calibration paths that exist on the cluster.
- **`python3` on the login node**, for MAMMA's small job wrapper (standard library only).

## Set up a cluster profile

A profile stores how to reach the cluster and how to run there. Profiles live on your machine in
`~/.config/mamma/clusters/`, never in the repository.

- **GUI:** New Task, Global section, Engine: *cluster (HTCondor, over SSH)*, then add a profile.
- **CLI:**

```bash
python -m inference cluster add my-cluster --host login.example.edu --username you \
    --workspace-root /shared/you/mamma-workspace \
    --engine apptainer --sif-path /shared/containers/mamma.sif --flavor standard
python -m inference cluster test my-cluster      # checks SSH, HTCondor, workspace and environment
```

Use `--flavor standard` (plain `condor_submit_dag`) unless your site submits through
`condor_submit_dag_bid`. A profile's `resources` map sets per-step requests: GPUs, memory,
`max_jobs`, `min_cuda_capability` / `max_cuda_capability`, and `exclude_machines` for broken hosts.

## Submit and follow a run

- **GUI:** choose the profile as the engine and start the task. The Tasks board shows each step and
  sequence as the cluster runs it; logs and outputs open from the GUI when the workspace is also
  mounted locally.
- **CLI:**

```bash
python -m inference submit-remote --profile my-cluster \
    --cfg configs/examples/presets/full.yaml --capture my_capture.json --watch
```

`--dry-run` builds the DAG on the cluster without submitting it. With ma_masks camera fan-out
(`ma_masks.fanout_cameras`), each camera runs as its own cluster job and a final job runs the rig vote
and the collage.
