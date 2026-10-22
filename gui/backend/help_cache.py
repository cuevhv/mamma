"""Cached `python <step>.py --help` parser.

Surfaces the available CLI flags per pipeline step to the GUI so users
editing a preset can see "what else could I add?" without leaving the
form. The actual catalogue is whatever argparse prints — no separate
maintained list to drift from the scripts.

Cost model:
* First request for a given step: spawn the script's `--help`, parse
  the output, write JSON cache. Typically 1-5 s because each step
  script imports torch/cv2/etc. just to print help.
* Subsequent requests: read JSON from disk (~5 ms) if the script file
  hasn't been modified since the cache was written.
* Cache invalidates automatically by comparing the script's mtime
  against ``scriptMtime`` recorded in the cache file. Edit a script
  -> next request re-parses.

The parser is a regex-based scrape of argparse's default text output.
It covers the common shapes (``--flag``, ``--flag VAL``, ``-x, --flag
VAL``, multi-line help) and falls through with empty results when the
output is too exotic to grok — the route then exposes ``rawHelp`` so
the user can still read it as plain text.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # gui/backend -> mamma_release
_CACHE_DIR = Path(os.environ.get("MAMMA_DATA_DIR", str(_REPO_ROOT / "gui" / "var"))) / "help_cache"

# Canonical (repo_subdir, script_basename) per step. Mirrors the
# defaults in scripts/generate_task_configs.py — kept hardcoded here
# rather than parsed from a preset because the user-saved preset may
# carry overrides that point elsewhere (e.g. a debug fork), and for
# flag-listing we always want the canonical script.
STEP_SCRIPTS: dict[str, tuple[str, str]] = {
    "ma_cap":   ("capture",       "run_ma_cap.py"),
    "ma_masks": ("segmentation",  "run_ma_masks.py"),
    "ma_2d":    ("landmarks",     "run_ma_2d.py"),
    "ma_3d":    ("optimization",  "run_ma_3d.py"),
    "ma_vis":   ("visualization", "run_ma_vis.py"),
}


def get_flags(step_name: str, *, force_refresh: bool = False) -> dict:
    """Return parsed argparse spec for a step.

    Shape::

        {
            "step":         "ma_cap",
            "scriptPath":   "capture/run_ma_cap.py",
            "scriptMtime":  1779700000.0,
            "parsedAt":     1779700050.0,
            "rawHelp":      "<full --help stdout>",
            "flags": [
                {"name": "--export_gt", "valueHint": None, "help": "..."},
                {"name": "--start", "valueHint": "START", "help": "..."},
                ...
            ],
            "warning":      None,   # or a short message when parse was empty
        }

    Raises:
        ValueError: ``step_name`` is not one of the known steps.
        FileNotFoundError: The script file doesn't exist on disk.
        RuntimeError: The ``--help`` subprocess timed out or errored.
    """
    if step_name not in STEP_SCRIPTS:
        raise ValueError(f"unknown step: {step_name!r}")
    repo_path, script_name = STEP_SCRIPTS[step_name]
    script_abs = _REPO_ROOT / repo_path / script_name
    if not script_abs.is_file():
        raise FileNotFoundError(f"step script missing: {script_abs}")

    cache_file = _CACHE_DIR / f"{step_name}.json"
    script_mtime = script_abs.stat().st_mtime

    if not force_refresh and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("scriptMtime") == script_mtime:
                # Re-parse the stored help so parser fixes apply without a cache bust.
                cached["flags"] = parse_argparse_help(cached.get("rawHelp", "")) or cached["flags"]
                return cached
        except (OSError, ValueError):
            pass  # corrupt cache -> re-parse

    raw = _run_help(script_abs, repo_path)
    flags = parse_argparse_help(raw)
    result: dict = {
        "step": step_name,
        "scriptPath": str(script_abs.relative_to(_REPO_ROOT)),
        "scriptMtime": script_mtime,
        "parsedAt": time.time(),
        "rawHelp": raw,
        "flags": flags,
        "warning": None if flags else "Parser produced zero flags; check rawHelp.",
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2))
    except OSError as e:
        # Cache write failure is non-fatal — return live result, log it.
        print(f"---> WARN: help_cache write failed for {step_name}: {e}")
    return result


def _run_help(script_abs: Path, repo_path: str) -> str:
    """Spawn ``python <script> --help`` and return stdout.

    We use ``sys.executable`` rather than `conda run -n <env>` because
    the Flask process is already running inside the right conda env
    (dev.sh / prod.sh activate ``mamma`` before launching us). Skipping
    the conda-run shim saves the ~1s env-activation overhead on every
    cache miss.

    cwd is the step's ``repo_path`` so any same-package relative imports
    in the script resolve the same way they would during a real run.
    """
    cmd = [sys.executable, script_abs.name, "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT / repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"--help timed out after 30s for {script_abs.name} — script "
            f"probably hangs on import. Check the script directly."
        ) from e
    if proc.returncode != 0:
        # argparse exits with 0 on --help; non-zero means the script
        # blew up during import. Surface the stderr so the user can fix.
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise RuntimeError(
            f"`python {script_abs.name} --help` exited rc={proc.returncode}. "
            f"Last stderr:\n{tail}"
        )
    return proc.stdout


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Section header that introduces the options block in argparse output.
# argparse renamed "optional arguments:" -> "options:" in Python 3.10;
# we accept both so this works against any vendored env.
_OPTIONS_HEADER_RE = re.compile(r'^\s*(?:options|optional arguments)\s*:\s*$', re.MULTILINE)

# A "header" line in the options block — leading 2 spaces then a dash.
# argparse hard-codes the 2-space indent for action entries.
_FLAG_HEADER_RE = re.compile(r'^  -')


def parse_argparse_help(text: str) -> list[dict]:
    """Turn argparse stdout into ``[{name, valueHint, help}, ...]``.

    Best-effort. Returns ``[]`` when the format is too exotic to parse
    — the caller surfaces ``rawHelp`` in that case so the user still
    sees something useful.
    """
    m = _OPTIONS_HEADER_RE.search(text)
    if not m:
        return []
    # Restrict to the options block — stop at the next bare section
    # header (a line starting in column 0 that ends with ':').
    block = text[m.end():]
    after = re.search(r'^\S.*:\s*$', block, re.MULTILINE)
    if after:
        block = block[:after.start()]

    flags: list[dict] = []
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _FLAG_HEADER_RE.match(line):
            i += 1
            continue
        # Strip the leading 2-space indent, then split at the first run
        # of 2+ spaces (the "gap" before the help column).
        header_then_help = line[2:]
        parts = re.split(r'  +', header_then_help, maxsplit=1)
        header = parts[0]
        help_text = parts[1].strip() if len(parts) > 1 else ""

        # Walk continuation lines (indented further than the flag header).
        i += 1
        while i < len(lines):
            cont = lines[i]
            if not cont.strip():
                break
            if _FLAG_HEADER_RE.match(cont):
                break  # next flag
            help_text = (help_text + " " + cont.strip()).strip()
            i += 1

        # Parse the header: each comma-separated form may carry a
        # metavar (e.g. "-j JOBS, --jobs JOBS"). Pick the long form
        # for the canonical name; keep the metavar as a value hint.
        # Commas inside a {a,b,c} choices metavar do not separate forms.
        forms = [p.strip() for p in re.split(r",(?![^{]*\})", header)]
        long_form = next((p for p in forms if p.startswith("--")), forms[-1])
        # Split flag name from metavar by first whitespace.
        name, _, metavar = long_form.partition(" ")
        flags.append({
            "name": name,
            "valueHint": metavar.strip() or None,
            "help": help_text or None,
        })

    return flags
