"""Transport: how the local client reaches the machine that runs the jobs.

One thin seam so the submit / sync / poll logic is written once and tested
without a cluster:

* :class:`SshTransport` — shells out to OpenSSH ``ssh`` / ``rsync`` / ``scp``,
  relying on the user's existing ``~/.ssh/config`` and agent (the lean auth
  path: no stored secrets, no paramiko). A persistent ControlMaster is used so
  one authentication serves every subsequent call.
* :class:`LocalTransport` — runs everything on the same box. Used by tests and
  when the client already runs on the login node.

The interface is deliberately tiny: run a command, move a file each way, tail a
growing file, mirror a directory, mkdir. Everything else composes from these.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Transport:
    """Abstract transport. Subclasses implement the primitives."""

    # --- primitives -------------------------------------------------------
    def run(self, argv: List[str]) -> CommandResult:
        """Run ``argv`` on the remote and capture output."""
        raise NotImplementedError

    def run_script(self, script: str) -> CommandResult:
        """Run a shell snippet on the remote (login shell)."""
        raise NotImplementedError

    def put(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def get(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError

    def sync_dir(self, local_dir: str, remote_dir: str,
                 excludes: Optional[List[str]] = None,
                 dry_run: bool = False) -> CommandResult:
        raise NotImplementedError

    # --- composed helpers -------------------------------------------------
    def mkdirs(self, remote_dir: str) -> CommandResult:
        return self.run(["mkdir", "-p", remote_dir])

    def exists(self, remote_path: str) -> bool:
        return self.run(["test", "-e", remote_path]).ok

    def tail(self, remote_path: str, offset: int) -> Tuple[str, int]:
        """Return ``(new_text, new_byte_offset)`` from ``offset`` to EOF.

        Missing file (no output yet) yields ``("", offset)``.
        """
        res = self.run(["tail", "-c", f"+{offset + 1}", "--", remote_path])
        if not res.ok:
            return "", offset
        text = res.stdout
        return text, offset + len(text.encode("utf-8", "replace"))

    def close(self) -> None:  # optional
        pass

    # --- authentication status (SshTransport overrides) --------------------
    # MAMMA never handles credentials itself. These hooks only OBSERVE the
    # user's SSH state so the GUI/CLI can say "connected" or point at the
    # one-time interactive login (``connect_argv`` run in a real terminal,
    # where OpenSSH does its own prompting: LDAP password, OTP, ...).

    def auth_ok(self) -> bool:
        """True when transport calls succeed without any interactive prompt."""
        return True

    def master_alive(self) -> bool:
        """True when a shared authenticated session (ControlMaster) is up."""
        return False

    def connect_argv(self, persist: str = "8h") -> Optional[List[str]]:
        """Foreground command that authenticates interactively, or ``None``
        when the transport needs no authentication (local)."""
        return None

    def close_master(self) -> None:
        """Tear down the shared session, if one exists."""


class LocalTransport(Transport):
    """Everything runs on the local box (tests; or client on the login node)."""

    def run(self, argv: List[str]) -> CommandResult:
        p = subprocess.run(argv, capture_output=True, text=True)
        return CommandResult(p.returncode, p.stdout, p.stderr)

    def run_script(self, script: str) -> CommandResult:
        p = subprocess.run(["bash", "-lc", script], capture_output=True, text=True)
        return CommandResult(p.returncode, p.stdout, p.stderr)

    def put(self, local_path: str, remote_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(remote_path)) or ".", exist_ok=True)
        shutil.copyfile(local_path, remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
        shutil.copyfile(remote_path, local_path)

    def sync_dir(self, local_dir: str, remote_dir: str,
                 excludes: Optional[List[str]] = None,
                 dry_run: bool = False) -> CommandResult:
        os.makedirs(remote_dir, exist_ok=True)
        argv = ["rsync", "-a", "--delete", "--itemize-changes"]
        if dry_run:
            argv.append("--dry-run")
        for e in (excludes or []):
            argv.append(f"--exclude={e}")
        argv += [local_dir.rstrip("/") + "/", remote_dir.rstrip("/") + "/"]
        return self.run(argv)

    def tail(self, remote_path: str, offset: int) -> Tuple[str, int]:
        if not os.path.exists(remote_path):
            return "", offset
        with open(remote_path, "rb") as f:
            f.seek(offset)
            data = f.read()
        return data.decode("utf-8", "replace"), offset + len(data)


class SshTransport(Transport):
    """OpenSSH-backed transport with a persistent ControlMaster.

    Auth is delegated entirely to the user's SSH setup (agent / ``~/.ssh/config``
    / keys). ``ssh_opts`` are extra raw tokens appended to every ssh invocation
    (e.g. ``["-J", "bastion"]`` for a jump host).
    """

    def __init__(self, host: str, username: str = "", port: int = 22,
                 ssh_opts: Optional[List[str]] = None,
                 control_dir: Optional[str] = None):
        self.host = host
        self.username = username
        self.port = int(port)
        self.ssh_opts = list(ssh_opts or [])
        self._target = f"{username}@{host}" if username else host
        # A stable multiplexing socket keyed by target: one auth, many calls.
        cdir = control_dir or os.path.join(
            tempfile.gettempdir(), "mamma-ssh")
        os.makedirs(cdir, exist_ok=True)
        key = hashlib.sha1(f"{self._target}:{self.port}".encode()).hexdigest()[:16]
        self.control_path = os.path.join(cdir, f"cm-{key}.sock")

    # --- ssh command assembly --------------------------------------------
    def _ssh_opts(self) -> List[str]:
        opts = ["-p", str(self.port),
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={self.control_path}",
                "-o", "ControlPersist=120"]
        opts += self.ssh_opts
        return opts

    # --- authentication ----------------------------------------------------
    # MAMMA never handles credentials. ``connect_argv`` is run by the USER in
    # a real terminal, where OpenSSH itself does the prompting (LDAP password,
    # OTP, ...); the authenticated ControlMaster then lives on
    # ``control_path`` for ``persist`` of idle time and every ssh/rsync call
    # here rides it. The remaining hooks only observe that state.

    def _master_ctl_argv(self, op: str) -> List[str]:
        return ["ssh", "-o", f"ControlPath={self.control_path}",
                "-O", op, self._target]

    def _auth_probe_argv(self) -> List[str]:
        # BatchMode forbids prompts, so this succeeds only via keys/agent or
        # a live master; ConnectTimeout keeps the GUI status check snappy.
        return ["ssh", *self._ssh_opts(),
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                self._target, "true"]

    def connect_argv(self, persist: str = "8h") -> List[str]:
        opts = ["-p", str(self.port),
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={self.control_path}",
                "-o", f"ControlPersist={persist}"]
        opts += self.ssh_opts
        return ["ssh", *opts, self._target, "true"]

    def master_alive(self) -> bool:
        return subprocess.run(self._master_ctl_argv("check"),
                              capture_output=True, text=True).returncode == 0

    def close_master(self) -> None:
        subprocess.run(self._master_ctl_argv("exit"),
                       capture_output=True, text=True)

    def auth_ok(self) -> bool:
        return subprocess.run(self._auth_probe_argv(),
                              capture_output=True, text=True).returncode == 0

    def _ssh_base(self) -> List[str]:
        return ["ssh", *self._ssh_opts(), self._target]

    def _rsync_e(self) -> str:
        return "ssh " + " ".join(shlex.quote(t) for t in self._ssh_opts())

    # --- primitives -------------------------------------------------------
    def run(self, argv: List[str]) -> CommandResult:
        remote_cmd = shlex.join(argv)
        p = subprocess.run(self._ssh_base() + [remote_cmd],
                           capture_output=True, text=True)
        return CommandResult(p.returncode, p.stdout, p.stderr)

    def run_script(self, script: str) -> CommandResult:
        p = subprocess.run(self._ssh_base() + [script],
                           capture_output=True, text=True)
        return CommandResult(p.returncode, p.stdout, p.stderr)

    def put(self, local_path: str, remote_path: str) -> None:
        self.mkdirs(os.path.dirname(remote_path) or ".")
        p = subprocess.run(
            ["rsync", "-a", "-e", self._rsync_e(),
             local_path, f"{self._target}:{remote_path}"],
            capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"put failed: {p.stderr.strip()}")

    def get(self, remote_path: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
        p = subprocess.run(
            ["rsync", "-a", "-e", self._rsync_e(),
             f"{self._target}:{remote_path}", local_path],
            capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"get failed: {p.stderr.strip()}")

    def sync_dir(self, local_dir: str, remote_dir: str,
                 excludes: Optional[List[str]] = None,
                 dry_run: bool = False) -> CommandResult:
        self.mkdirs(remote_dir)
        argv = ["rsync", "-az", "--delete", "--itemize-changes",
                "-e", self._rsync_e()]
        if dry_run:
            argv.append("--dry-run")
        for e in (excludes or []):
            argv.append(f"--exclude={e}")
        argv += [local_dir.rstrip("/") + "/", f"{self._target}:{remote_dir.rstrip('/')}/"]
        p = subprocess.run(argv, capture_output=True, text=True)
        return CommandResult(p.returncode, p.stdout, p.stderr)

    def close(self) -> None:
        # Tear down the master connection (best-effort).
        subprocess.run(["ssh", "-O", "exit",
                        "-o", f"ControlPath={self.control_path}", self._target],
                       capture_output=True, text=True)
