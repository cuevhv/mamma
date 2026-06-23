import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inference import engines


class _DummyBuilder:
    step_name = "dummy"
    conda_env = "mamma"

    def build_argv(self, seq_name: str):
        return ["run_dummy.py", "--seq_name", seq_name]

    def host_cwd(self) -> str:
        return "/tmp/dummy"


class RunCondaTests(unittest.TestCase):
    def setUp(self):
        self.builder = _DummyBuilder()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.stdout_log = str(Path(self.tmpdir.name) / "stdout.log")
        self.stderr_log = str(Path(self.tmpdir.name) / "stderr.log")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_conda_uses_conda_when_available(self):
        seen = {}

        def fake_run(cmd, cwd, out_f, err_f):
            seen["cmd"] = cmd
            seen["cwd"] = cwd
            return 0

        with mock.patch.object(engines.shutil, "which", return_value="/usr/bin/conda"):
            with mock.patch.object(engines, "_run", side_effect=fake_run):
                rc = engines.run_conda(
                    self.builder,
                    "seq01",
                    self.stdout_log,
                    self.stderr_log,
                )

        self.assertEqual(rc, 0)
        self.assertEqual(seen["cwd"], "/tmp/dummy")
        self.assertEqual(
            seen["cmd"],
            [
                "conda",
                "run",
                "-n",
                "mamma",
                "--no-capture-output",
                "--live-stream",
                "python",
                "run_dummy.py",
                "--seq_name",
                "seq01",
            ],
        )

    def test_run_conda_falls_back_to_current_python_when_conda_missing(self):
        seen = {}

        def fake_run(cmd, cwd, out_f, err_f):
            seen["cmd"] = cmd
            seen["cwd"] = cwd
            return 0

        with mock.patch.object(engines.shutil, "which", return_value=None):
            with mock.patch.object(engines, "_run", side_effect=fake_run):
                rc = engines.run_conda(
                    self.builder,
                    "seq01",
                    self.stdout_log,
                    self.stderr_log,
                )

        self.assertEqual(rc, 0)
        self.assertEqual(seen["cwd"], "/tmp/dummy")
        self.assertEqual(
            seen["cmd"],
            [sys.executable, "run_dummy.py", "--seq_name", "seq01"],
        )


if __name__ == "__main__":
    unittest.main()