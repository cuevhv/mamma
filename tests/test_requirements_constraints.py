"""requirements/constraints.txt must agree with the requirement files it is used with."""
import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import Specifier
from packaging.utils import canonicalize_name

REQ = Path(__file__).resolve().parents[1] / "requirements"
FILES = ("requirements.txt", "requirements_no_build_isolation.txt", "requirements-vposer.txt", "requirements-gui.txt")
# sam2/sam3 ask for iopath>=0.1.10, detectron2 (a later layer) installs 0.1.9: no single pin fits both.
UNPINNED = {"iopath"}


def _lines(name):
    for line in (REQ / name).read_text().splitlines():
        line = line.split(" #")[0].strip()
        if line and not line.startswith(("#", "-")):
            yield line


def _pins():
    pins = {}
    for line in _lines("constraints.txt"):
        r = Requirement(line)
        (spec,) = r.specifier
        assert spec.operator == "==", line
        pins[canonicalize_name(r.name)] = spec.version
    return pins


def test_pins_satisfy_requirements():
    pins = _pins()
    for name in FILES:
        for line in _lines(name):
            if line.startswith("git+"):
                continue
            r = Requirement(line)
            if r.url:
                continue
            if canonicalize_name(r.name) in UNPINNED:
                continue
            v = pins.get(canonicalize_name(r.name))
            assert v is not None, f"{name}: {line} has no pin in constraints.txt"
            # An exact requirement may carry a local label (torch==2.5.1+cpu), which ==2.5.1 accepts.
            exact = [s.version for s in r.specifier if s.operator == "=="]
            cand = exact[0] if exact else v
            ok = Specifier(f"=={v}").contains(cand, prereleases=True) and r.specifier.contains(cand, prereleases=True)
            assert ok, f"{name}: {line} excludes the pin =={v}"


def test_git_requirements_pinned_to_a_commit():
    for name in FILES:
        for line in _lines(name):
            if "git+" in line:
                assert re.search(r"@[0-9a-f]{40}$", line), f"{name}: {line} is not pinned to a commit"
