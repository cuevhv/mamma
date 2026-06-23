import tempfile
import unittest
from pathlib import Path
from unittest import mock

from landmarks.utils import video_utils


class VideoUtilsFfmpegTests(unittest.TestCase):
    def test_create_video_uses_resolved_ffmpeg_exe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "img_0001.jpg").write_bytes(b"x")
            out = folder / "out.mp4"
            seen = {}

            def fake_run(cmd, check, capture_output, text):
                seen["cmd"] = cmd
                return None

            with mock.patch.object(video_utils, "FFMPEG_EXE", "/tmp/fake-ffmpeg"):
                with mock.patch("subprocess.run", side_effect=fake_run):
                    ok = video_utils.create_video_from_images(str(folder), str(out))

        self.assertTrue(ok)
        self.assertEqual(seen["cmd"][0], "/tmp/fake-ffmpeg")


if __name__ == "__main__":
    unittest.main()