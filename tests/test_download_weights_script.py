"""data/download_mamma_weights.sh drives wget with TLS verification on.

A stub ``wget`` on PATH records its argv and writes a fake payload, so the
script runs end to end without network access.
"""
import os
import stat
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "data" / "download_mamma_weights.sh"


def test_wget_keeps_certificate_verification(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_log = tmp_path / "wget_argv.txt"
    stub = bin_dir / "wget"
    stub.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" >> "{argv_log}"\n'
        'while [[ $# -gt 0 ]]; do [[ "$1" == "-O" ]] && out="$2"; shift; done\n'
        'printf "payload" > "$out"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    out_dir = tmp_path / "data"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               MAMMA_USERNAME="user", MAMMA_PASSWORD="pass")
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "--verts", "--output", str(out_dir)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out_dir / "body_models/downsampled_verts/verts_512.pkl").read_text() == "payload"
    argv = argv_log.read_text().splitlines()
    assert argv, "wget was not called"
    assert "--no-check-certificate" not in argv
    assert any(a.startswith("https://") for a in argv)
