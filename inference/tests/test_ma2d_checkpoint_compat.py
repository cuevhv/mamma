import re
import unittest
from pathlib import Path


class Ma2dCheckpointCompatTests(unittest.TestCase):
    def test_run_ma_2d_explicitly_disables_weights_only_for_trusted_checkpoint(self):
        src = Path("landmarks/run_ma_2d.py").read_text(encoding="utf-8")
        self.assertRegex(
            src,
            re.compile(r"torch\.load\([^\n]*weights_only\s*=\s*False|torch\.load\([\s\S]*weights_only\s*=\s*False", re.MULTILINE),
        )


if __name__ == "__main__":
    unittest.main()