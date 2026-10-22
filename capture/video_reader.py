"""In-memory video frame reader for MP4 files.

No frames are written to disk. SAM receives the MP4 path directly;
The person detector and CLIP receive PIL Images from this reader.

Usage::

    from capture.video_reader import VideoFrameReader, cam_data_from_video

    reader = VideoFrameReader("/path/to/IOI_09.mp4", start=10, end=50)
    pil_img = reader.read_pil(0)   # frame 0 within the range (global frame 10)
    print(len(reader))             # 40 frames

    cam_data = cam_data_from_video("/path/to/IOI_09.mp4", start=10, end=50)
"""
import logging
import os
import threading
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class VideoFrameReader:
    """Random-access frame reader for MP4 videos. No disk I/O for frames.

    Tuned for the usual in-order playback pattern. A single ``cv2.VideoCapture``
    is held open and advanced lazily: a forward request walks the decoder to the
    target with ``grab()`` (which decodes but skips the pixel copy), and only a
    request for an *earlier* frame pays a ``CAP_PROP_POS_FRAMES`` seek. This
    sidesteps the seek-from-keyframe re-decode that a 4K/large-GOP H.264/H.265
    stream incurs on every random access, turning a sequential pass from
    quadratic into linear while returning the exact same BGR frames.

    Access from several threads is safe — the capture handle and its position
    counter live behind a mutex, so the concurrent SAM3 mask exporter (which
    pulls frames from a shared reader across a thread pool) is serialized only at
    the decode step. A lone reader never contends, so single-threaded scans keep
    the full speed-up.
    """

    def __init__(self, video_path: str, start: int = None, end: int = None):
        import cv2

        self.video_path = os.path.abspath(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.start = max(0, min(start or 0, total))
        self.end = max(self.start, min(end or total, total))
        self.n_frames = self.end - self.start
        # Held-open decoder + the global index its next read() will yield.
        # Created on first use; ``_mutex`` guards both fields for thread safety.
        self._capture = None
        self._next_global = 0
        self._mutex = threading.Lock()
        logger.info(
            "VideoFrameReader: '%s' frames [%d:%d] (%d of %d frames, %dx%d, %.1f fps)",
            os.path.basename(video_path), self.start, self.end,
            self.n_frames, total, self.width, self.height, self.fps,
        )

    def __len__(self) -> int:
        return self.n_frames

    def _open(self):
        """(Re)create the held-open capture. Call while holding ``_mutex``."""
        import cv2

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self._capture = cap
        self._next_global = 0

    def read_bgr(self, local_idx: int) -> np.ndarray:
        """Read frame as BGR numpy array (cv2 convention).

        Walks the held-open decoder forward to the requested frame; only an
        earlier-than-current request triggers a seek. Safe under concurrency.
        """
        import cv2

        if local_idx < 0 or local_idx >= self.n_frames:
            raise IndexError(f"Frame index {local_idx} out of range [0, {self.n_frames})")
        target = self.start + local_idx

        with self._mutex:
            if self._capture is None:
                self._open()

            # Behind the cursor -> jump back with a seek; otherwise drop the
            # frames in between with grab() (no array copy) until we line up.
            if target < self._next_global:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                self._next_global = target
            for _ in range(target - self._next_global):
                if not self._capture.grab():
                    raise RuntimeError(
                        f"Failed to skip to frame {target} in '{self.video_path}'")
                self._next_global += 1

            ok, frame = self._capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {target} from '{self.video_path}'")
            self._next_global += 1
            return frame

    def close(self):
        """Release the held-open capture if present."""
        with self._mutex:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
                self._next_global = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_rgb(self, local_idx: int) -> np.ndarray:
        """Read frame as RGB numpy array."""
        import cv2
        return cv2.cvtColor(self.read_bgr(local_idx), cv2.COLOR_BGR2RGB)

    def read_pil(self, local_idx: int):
        """Read frame as PIL Image (RGB)."""
        from PIL import Image
        return Image.fromarray(self.read_rgb(local_idx))


def cam_data_from_video(video_path: str, start: int = None, end: int = None) -> Dict[str, Any]:
    """Build a cam_data dict from an MP4 video -- no disk I/O.

    Args:
        video_path: Path to MP4 video file.
        start: First frame index (0-based, inclusive). None = 0.
        end: Last frame index (0-based, exclusive). None = all frames.

    Returns:
        Dict compatible with ``process_multi_video_auto`` / ``FrameSource`` factory.
    """
    cam_name = os.path.splitext(os.path.basename(video_path))[0]
    reader = VideoFrameReader(video_path, start=start, end=end)

    return {
        'cam_name': np.array(cam_name),
        'video_path': np.array(os.path.abspath(video_path)),
        'frame_reader': reader,
    }
