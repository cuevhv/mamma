"""Cluster execution backend.

The cluster counterpart of :mod:`inference.engines`. Where the local runner
Popens :func:`inference.engines.build_command`, this package renders the same
``(step, sequence)`` matrix into HTCondor submit artifacts (``.sh`` / ``.sub``
/ ``.dag``) and submits them via ``condor_submit_dag[_bid]``.

Nothing about per-step argv is derived here: each DAG node re-enters the
runner for a single cell (``python -m inference run-cell``), which builds the
exact command a local run would. See :mod:`inference.cluster.htcondor`.
"""
# Keep this import light: only the cluster-side emitter + status map, which run
# on the login / execute node (minimal python, no third-party deps). The
# client-side modules (transport, profile, remote, monitor) pull yaml / the
# local env and are imported explicitly by the client (submit-remote, GUI), not
# eagerly here, so `python -m inference submit-cluster|run-cell` stays stdlib-only.
from . import htcondor, scheduler, status_map  # noqa: F401
