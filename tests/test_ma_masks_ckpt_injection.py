"""SAM2 checkpoint injection must mirror the subprocess's actual default.

run_ma_masks.py's argparse defaults --sam_version to sam2, so a config with
NO version flag runs sam2 and needs the checkpoint injected. The old
fallback assumed sam3_prompt and skipped injection: fine locally where the
cwd-relative default path exists, a guaranteed FileNotFoundError on cluster
nodes (gs_off study post-mortem, 2026-08-15)."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference.steps.ma_masks import _preset_supplies_sam_checkpoint, _resolve_sam_version


def test_bare_flags_resolve_to_sam2():
    # must match run_ma_masks.py's argparse default
    assert _resolve_sam_version([]) == "sam2"


def test_explicit_version_wins_and_last_occurrence():
    assert _resolve_sam_version(["--sam_version", "sam3_prompt"]) == "sam3_prompt"
    assert _resolve_sam_version(["--sam_version=sam3"]) == "sam3"
    assert _resolve_sam_version(
        ["--sam_version", "sam2", "--sam_version", "sam3_prompt_light"]
    ) == "sam3_prompt_light"


def test_preset_supplied_checkpoint_detection():
    assert _preset_supplies_sam_checkpoint(["--sam_checkpoint", "/x.pt"])
    assert _preset_supplies_sam_checkpoint(["--sam_checkpoint=/x.pt"])
    assert not _preset_supplies_sam_checkpoint(["--sam_version", "sam2"])
