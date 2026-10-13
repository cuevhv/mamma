"""Fan-out cells must be able to narrow the camera set: step flags win over
the injected global camera list under argparse last-occurrence semantics."""
from inference import env
from inference.steps import get_builder


def _argv():
    env.bootstrap_env()
    b = get_builder(
        "ma_masks",
        {"script": "run_ma_masks.py", "repo_path": "segmentation",
         "flags": ["--sam_version", "sam2",
                   "--cam_names", "IOI_01", "IOI_09", "--cam_init", "IOI_01"]},
        {"out_dir": "/out", "dataset_name": "d",
         "cam_names": [f"IOI_{i:02d}" for i in range(1, 34)]},
        "tag")
    return b.python_argv("seqA")


def test_flags_cam_names_come_after_global_injection():
    argv = _argv()
    positions = [i for i, a in enumerate(argv) if a == "--cam_names"]
    assert len(positions) == 2
    last = positions[-1]
    # The LAST occurrence (which argparse honors) is the flags' subset.
    assert argv[last + 1:last + 3] == ["IOI_01", "IOI_09"]
    assert argv[last + 3].startswith("--")
    # The global 33-camera list is still present, earlier, as the default.
    first = positions[0]
    assert argv[first + 1:first + 34] == [f"IOI_{i:02d}" for i in range(1, 34)]


def test_ma_2d_flags_cam_names_override_global_injection():
    """Same last-wins contract for ma_2d: a fan-out cell's --cam_names in the
    flags must land after the injected global rig."""
    env.bootstrap_env()
    b = get_builder(
        "ma_2d",
        {"script": "run_ma_2d.py", "repo_path": "landmarks",
         "flags": ["--cam_names", "IOI_05"]},
        {"out_dir": "/out", "dataset_name": "d",
         "cam_names": [f"IOI_{i:02d}" for i in range(1, 34)]},
        "tag")
    argv = b.python_argv("seqA")
    positions = [i for i, a in enumerate(argv) if a == "--cam_names"]
    assert len(positions) == 2
    assert argv[positions[-1] + 1] == "IOI_05"
    assert argv[positions[0] + 1:positions[0] + 34] == \
        [f"IOI_{i:02d}" for i in range(1, 34)]


def test_cell_log_paths_are_namespaced_per_cell():
    """Concurrent fan-out cells sharing one <seq>.err clobber each other's
    tracebacks (run 163); each cell gets its own pair."""
    from inference.runner import _log_paths
    g = {"jobs_log_dir": "/logs", "username": "u"}
    out, err = _log_paths(g, "163", "ma_masks", "seqA", cell="IOI_02")
    assert out.endswith("/163/ma_masks/seqA.IOI_02.out")
    assert err.endswith("/163/ma_masks/seqA.IOI_02.err")
    # Plain (non-fan-out) runs keep the historical name.
    out, err = _log_paths(g, "163", "ma_masks", "seqA")
    assert out.endswith("/163/ma_masks/seqA.out")
    assert err.endswith("/163/ma_masks/seqA.err")
