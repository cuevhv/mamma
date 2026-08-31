"""Factory for building SAM2 or SAM3 video predictors.

Both backends are wrapped to expose a unified API matching SAM2's interface:
    predictor.init_state(video_path=...) -> inference_state
    predictor.add_new_points_or_box(inference_state, frame_idx, obj_id, box=...)
    predictor.propagate_in_video(inference_state, reverse=False)
        -> yields (frame_idx, obj_ids, mask_logits)
    predictor.reset_state(inference_state)
"""

import os
import logging
import warnings

# Suppress pkg_resources deprecation warning from SAM3
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning)

logger = logging.getLogger(__name__)

# NOTE: this is process-wide module state read by the
# patched SAM loaders. They assume a single active pipeline per process (the runner's
# model — set once per run, before loading). Two pipelines with different settings in
# one process would race; revisit if the GUI ever loads pipelines concurrently.
#
# Frame filter: when set, the patched SAM frame loaders only load filenames
# in this set. Set to None to load all frames (default behavior).
# Used by _apply_frame_range to avoid copying/symlinking frames.
_frame_name_filter = None


def set_frame_filter(frame_names):
    """Set the frame filter. Only these filenames will be loaded by SAM.

    Args:
        frame_names: Set/list of filenames (basenames, e.g. {"0000000200.png", ...}),
                     or None to clear the filter.
    """
    global _frame_name_filter
    _frame_name_filter = set(frame_names) if frame_names is not None else None


def clear_frame_filter():
    """Clear the frame filter so SAM loads all frames."""
    global _frame_name_filter
    _frame_name_filter = None


# Lazy frame loading: load frames on demand (a small window in RAM) instead of
# materializing all N decoded frames at once, so host RAM is O(window) not
# O(num_frames) — the way to bound host RAM on long clips (offload bounds VRAM,
# not host). SAM accesses frames in monotonic sweeps (one per propagation step;
# memory-attention uses cached encoded features, not raw images), so a small
# window never thrashes. Two backends, both keyed off ``sam.lazy_frame_loading``:
#   - image / JPG-folder input -> _LazyEvictingFrameLoader (LRU window).
#     Byte-identical; cost is re-reading each frame from disk once per pass.
#   - native mp4 input -> _VideoStreamFrameLoader (cv2 chunk streaming). No disk
#     extraction; near-identical to the stock decord decode (edge-pixel rounding,
#     ~0.99 IoU) and ~+24% wall from re-decoding on revisit. Opt-in.
_lazy_frame_loading = False
_lazy_window = 8
_lazy_video_chunk = 64        # frames decoded per cv2 seek on the native-mp4 path


def set_lazy_frame_loading(enabled: bool, window: int = 8, video_chunk: int = 64):
    """Enable/disable lazy (LRU-windowed) decoded-frame loading.

    ``video_chunk`` sizes the mp4 path's decode buffer (host RAM is about
    2 x chunk x one decoded frame with the staged chunk included); lower it
    on low-RAM machines at the cost of more frequent decodes.
    """
    global _lazy_frame_loading, _lazy_window, _lazy_video_chunk
    _lazy_frame_loading = bool(enabled)
    _lazy_window = max(1, int(window))
    _lazy_video_chunk = max(1, int(video_chunk))


class _LazyEvictingFrameLoader:
    """List-like view over decoded frames that keeps only an LRU window in RAM.

    Matches the eager path's per-frame processing exactly — cast to ``store_dtype``,
    move to ``compute_device`` when not offloading, then ``(img - mean) / std`` —
    so ``self[i]`` equals the eager ``images[i]`` bit-for-bit. Exposes the same
    interface SAM uses: ``len()``, ``[idx]`` -> (3,H,W) tensor, ``video_height``/
    ``video_width``.
    """

    def __init__(self, img_paths, load_fn, image_size, img_mean, img_std,
                 compute_device, offload_video_to_cpu, window):
        import threading
        from collections import OrderedDict
        # The mask-export stage reads frames from a ThreadPoolExecutor (see
        # save_images_from_video); OrderedDict move_to_end/popitem are not
        # thread-safe, so serialize like _VideoStreamFrameLoader does.
        self._lock = threading.Lock()
        self._paths = img_paths
        self._load = load_fn
        self._size = image_size
        self._offload = offload_video_to_cpu
        self._device = compute_device
        # mean/std live where the arithmetic happens (device unless offloading),
        # mirroring the eager loader.
        self._mean = img_mean if offload_video_to_cpu else img_mean.to(compute_device)
        self._std = img_std if offload_video_to_cpu else img_std.to(compute_device)
        self._window = max(1, int(window))
        self._cache = OrderedDict()
        self.video_height = None
        self.video_width = None
        # Decode-ahead prefetch: SAM walks frames in monotonic sweeps, so
        # while the GPU processes frame i, background workers decode i+d..
        # i+d*ahead (d inferred from the request direction). The GPU then
        # never waits on disk/NFS/decode; identical tensors, earlier.
        self._prefetch_ahead = min(6, self._window - 1)
        self._pending = set()
        self._exec = None
        self._last_idx = None
        self._dir = 1
        self[0]  # prime dims (and cache frame 0)

    def __len__(self):
        return len(self._paths)

    def _load_one(self, idx):
        img, self.video_height, self.video_width = self._load(self._paths[idx], self._size)
        img = img.to(self._mean.dtype)
        if not self._offload:
            img = img.to(self._device)
        img = (img - self._mean) / self._std
        with self._lock:
            self._pending.discard(idx)
            if idx not in self._cache:
                self._cache[idx] = img
                while len(self._cache) > self._window:
                    self._cache.popitem(last=False)  # evict least-recently-used
        return img

    def _schedule_prefetch(self, idx):
        if self._prefetch_ahead < 1:
            return
        if self._exec is None:
            from concurrent.futures import ThreadPoolExecutor
            # Sized for slow sources: a cold-NFS PNG costs ~175 ms to acquire
            # while a model step is ~40-120 ms, so 2 workers cannot keep up.
            self._exec = ThreadPoolExecutor(max_workers=4)
        with self._lock:
            if self._last_idx is not None and idx != self._last_idx:
                self._dir = 1 if idx > self._last_idx else -1
            self._last_idx = idx
            targets = []
            for k in range(1, self._prefetch_ahead + 1):
                j = idx + self._dir * k
                if 0 <= j < len(self._paths) and j not in self._cache \
                        and j not in self._pending:
                    self._pending.add(j)
                    targets.append(j)
        for j in targets:
            self._exec.submit(self._load_one, j)

    def __getitem__(self, idx):
        self._schedule_prefetch(idx)
        with self._lock:
            cached = self._cache.get(idx)
            if cached is not None:
                self._cache.move_to_end(idx)
                return cached
        return self._load_one(idx)


class _VideoStreamFrameLoader:
    """Bounded-RAM frame view that streams directly from the mp4 (no disk extraction).

    SAM2's stock ``load_video_frames_from_video_file`` decodes *every* frame into one
    tensor (O(num_frames) host RAM — ~47 GB on a 3.7k-frame 1024px clip). This keeps
    only a ``chunk``-sized buffer, decoded on demand with ``cv2.VideoCapture``:
    sequential reads are cheap and chunk-start seeks are frame-accurate (verified).
    SAM walks frames in monotonic sweeps (forward anchor->end, then reverse
    anchor->0), so each chunk is decoded once per sweep and host RAM stays O(chunk).
    Per-frame processing matches the eager path (resize, /255, ``(img-mean)/std``);
    pixels are near-identical to native decode (cv2 vs decord differ by at most
    edge-pixel rounding). cv2 is used rather than decord because decord's reader
    corrupts the heap after a few dozen random reads.
    """

    def __init__(self, video_path, image_size, img_mean, img_std,
                 compute_device, offload_video_to_cpu, chunk):
        import cv2
        import threading
        # The mask/overlay export reads frames from a ThreadPoolExecutor, so __getitem__
        # is called concurrently. cv2.VideoCapture is a single shared decoder and is NOT
        # thread-safe (concurrent reads segfault), so serialize access with a lock. The
        # export walks frames roughly in order, so the chunk buffer still hits and the
        # serialization cost is negligible.
        self._lock = threading.Lock()
        self._path = video_path
        self._size = image_size
        self._cap = cv2.VideoCapture(video_path)
        self.video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._offload = offload_video_to_cpu
        self._device = compute_device
        self._mean = img_mean if offload_video_to_cpu else img_mean.to(compute_device)
        self._std = img_std if offload_video_to_cpu else img_std.to(compute_device)
        self._chunk = max(1, int(chunk))
        self._buf = {}
        self._buf_chunk = -1
        self._pos = 0  # next frame index the capture will read
        # Chunk-ahead prefetch state (own capture: cv2 readers are not
        # shareable across threads). One staged chunk at a time.
        self._stage_lock = threading.Lock()
        self._staged = None
        self._stage_busy = False
        self._stage_exec = None
        self._pf_cap = None
        self._pf_pos = 0
        self._last_chunk = None
        self._dir = 1

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        with self._lock:
            c = idx // self._chunk
            if c != self._buf_chunk:
                staged = None
                with self._stage_lock:
                    if self._staged is not None and self._staged[0] == c:
                        staged = self._staged
                        self._staged = None
                if staged is not None:
                    self._buf, self._buf_chunk = staged[1], c
                else:
                    self._buf = self._decode_chunk(self._cap, c, main=True)
                    self._buf_chunk = c
                if self._last_chunk is not None and c != self._last_chunk:
                    self._dir = 1 if c > self._last_chunk else -1
                self._last_chunk = c
                self._stage_ahead(c + self._dir)
            return self._buf[idx]

    def _decode_chunk(self, cap, c, main=False):
        import cv2
        import torch
        start = c * self._chunk
        end = min(start + self._chunk, self._n)
        pos_attr = "_pos" if main else "_pf_pos"
        if getattr(self, pos_attr) != start:  # avoid a seek when streaming in order
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            setattr(self, pos_attr, start)
        frames = []
        for _ in range(start, end):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(
                    f"cv2 failed to read frame {getattr(self, pos_attr)} of '{self._path}'.")
            setattr(self, pos_attr, getattr(self, pos_attr) + 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self._size, self._size))
            frames.append(torch.from_numpy(rgb))
        batch = torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0
        if not self._offload:
            batch = batch.to(self._device)
        batch = (batch - self._mean) / self._std
        return {start + i: batch[i] for i in range(end - start)}

    def _stage_ahead(self, c):
        """Decode chunk ``c`` on a background thread (own cv2 capture) so the
        next chunk switch swaps in without the GPU waiting on the decoder."""
        if not (0 <= c * self._chunk < self._n):
            return
        with self._stage_lock:
            if self._stage_busy or (self._staged is not None and self._staged[0] == c):
                return
            self._stage_busy = True
        if self._stage_exec is None:
            from concurrent.futures import ThreadPoolExecutor
            self._stage_exec = ThreadPoolExecutor(max_workers=1)

        def _work():
            import cv2
            try:
                if self._pf_cap is None:
                    self._pf_cap = cv2.VideoCapture(self._path)
                    self._pf_pos = 0
                buf = self._decode_chunk(self._pf_cap, c, main=False)
                with self._stage_lock:
                    self._staged = (c, buf)
            except Exception:
                pass  # prefetch is best-effort; the hot path decodes on miss
            finally:
                with self._stage_lock:
                    self._stage_busy = False

        self._stage_exec.submit(_work)


def _install_video_lazy_patch(misc_module):
    """Swap a SAM misc module's ``load_video_frames_from_video_file`` for the
    bounded-RAM cv2 streaming loader when ``_lazy_frame_loading`` is on.

    Stock loaders (identical in sam2.utils.misc and sam3.model.utils.sam2_utils)
    materialize *all* decoded frames at once, so full-clip mp4 input blows up host
    RAM (~47 GB on a 3.7k-frame clip). ``_VideoStreamFrameLoader`` streams a chunk
    at a time — same O(window) host RAM as the JPG path, no disk extraction. The
    caller (``load_video_frames``) passes the model's own img_mean/img_std, so the
    same patch works for both sam2 and sam3 despite their different normalizations.
    """
    import torch
    if getattr(misc_module, "_mamma_video_lazy_patched", False):
        return
    _orig = misc_module.load_video_frames_from_video_file

    def patched_load_video_frames_from_video_file(
        video_path, image_size, offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225),
        compute_device=torch.device("cuda"),
    ):
        if _lazy_frame_loading and isinstance(video_path, str):
            mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
            loader = _VideoStreamFrameLoader(
                video_path, image_size, mean, std, compute_device,
                offload_video_to_cpu, _lazy_video_chunk,
            )
            return loader, loader.video_height, loader.video_width
        return _orig(video_path, image_size, offload_video_to_cpu, img_mean, img_std, compute_device)

    misc_module.load_video_frames_from_video_file = patched_load_video_frames_from_video_file
    misc_module._mamma_video_lazy_patched = True


def _patch_sam2_video_lazy():
    """Install the bounded-RAM video loader on SAM2's frame loader."""
    from sam2.utils import misc as sam2_misc
    _install_video_lazy_patch(sam2_misc)


def _patch_sam3_video_lazy():
    """Install the bounded-RAM video loader on the SAM3 tracker's frame loader."""
    try:
        from sam3.model.utils import sam2_utils as sam3_misc
    except Exception:
        return
    if hasattr(sam3_misc, "load_video_frames_from_video_file"):
        _install_video_lazy_patch(sam3_misc)


def _patch_sam2_png_support():
    """
    Monkey-patch sam2.utils.misc to support PNG images and robust frame sorting.
    Replaces the build-time .def patch with a runtime equivalent.
    """
    try:
        import sam2.utils.misc as sam2_misc
    except ImportError:
        logger.warning("sam2.utils.misc not found; skipping PNG patch.")
        return

    _original_load_video_frames_from_jpg_images = getattr(
        sam2_misc, "load_video_frames_from_jpg_images", None
    )
    if _original_load_video_frames_from_jpg_images is None:
        return

    # Check if already patched
    if getattr(sam2_misc, "_png_patch_applied", False):
        return

    import os
    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    def _load_img_as_tensor(img_path, image_size):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size
        return img, video_height, video_width

    def patched_load_video_frames_from_jpg_images(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only image frames in a folder are supported. "
                "Use ffmpeg to extract frames from video files."
            )

        # PNG + JPEG support
        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        # Robust sort: try integer-based, fall back to lexicographic
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()

        # Apply frame filter if set (for --start/--end frame range)
        if _frame_name_filter is not None:
            frame_names = [f for f in frame_names if f in _frame_name_filter]

        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
        store_dtype = torch.float32
        img_mean = torch.tensor(img_mean, dtype=store_dtype)[:, None, None]
        img_std = torch.tensor(img_std, dtype=store_dtype)[:, None, None]

        if _lazy_frame_loading:
            lf = _LazyEvictingFrameLoader(
                img_paths, _load_img_as_tensor, image_size,
                img_mean, img_std, compute_device, offload_video_to_cpu, _lazy_window,
            )
            return lf, lf.video_height, lf.video_width

        if async_loading_frames:
            lazy_images = sam2_misc.AsyncVideoFrameLoader(
                img_paths,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=store_dtype)
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG/PNG)")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)
        images -= img_mean
        images /= img_std
        return images, video_height, video_width

    sam2_misc.load_video_frames_from_jpg_images = patched_load_video_frames_from_jpg_images
    sam2_misc._png_patch_applied = True
    logger.info("Applied SAM2 PNG/sorting monkey-patch to sam2.utils.misc.")


def _patch_sam3_png_support():
    """
    Monkey-patch sam3.model.utils.sam2_utils to support PNG images and robust
    frame sorting — same logic as _patch_sam2_png_support but targeting SAM3's
    copy of the loader.
    """
    try:
        import sam3.model.utils.sam2_utils as sam3_misc
    except ImportError:
        logger.warning("sam3.model.utils.sam2_utils not found; skipping SAM3 PNG patch.")
        return

    if getattr(sam3_misc, "_png_patch_applied", False):
        return

    if not hasattr(sam3_misc, "load_video_frames_from_jpg_images"):
        return

    import os
    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    def _load_img_as_tensor(img_path, image_size):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size
        return img, video_height, video_width

    def patched_load_video_frames(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only image frames in a folder are supported. "
                "Use ffmpeg to extract frames from video files."
            )

        frame_names = [
            p for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()

        # Apply frame filter if set (for --start/--end frame range)
        if _frame_name_filter is not None:
            frame_names = [f for f in frame_names if f in _frame_name_filter]

        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
        store_dtype = torch.float32
        img_mean = torch.tensor(img_mean, dtype=store_dtype)[:, None, None]
        img_std = torch.tensor(img_std, dtype=store_dtype)[:, None, None]

        if _lazy_frame_loading:
            lf = _LazyEvictingFrameLoader(
                img_paths, _load_img_as_tensor, image_size,
                img_mean, img_std, compute_device, offload_video_to_cpu, _lazy_window,
            )
            return lf, lf.video_height, lf.video_width

        if async_loading_frames:
            lazy_images = sam3_misc.AsyncVideoFrameLoader(
                img_paths, image_size, offload_video_to_cpu,
                img_mean, img_std, compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=store_dtype)
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG/PNG)")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)
        images -= img_mean
        images /= img_std
        return images, video_height, video_width

    sam3_misc.load_video_frames_from_jpg_images = patched_load_video_frames
    sam3_misc._png_patch_applied = True
    logger.info("Applied SAM3 PNG/sorting monkey-patch to sam3.model.utils.sam2_utils.")

    # Also patch sam3.model.io_utils (used by Sam3VideoPredictor / sam3_prompt mode)
    try:
        import sam3.model.io_utils as sam3_io
    except ImportError:
        return

    if getattr(sam3_io, "_frame_filter_patch_applied", False):
        return

    _original_load_from_folder = getattr(sam3_io, "load_video_frames_from_image_folder", None)
    if _original_load_from_folder is None:
        return

    def patched_load_from_image_folder(
        image_folder, image_size, offload_video_to_cpu, img_mean, img_std, async_loading_frames,
    ):
        """Patched loader that respects _frame_name_filter."""
        import os
        frame_names = [
            p for p in os.listdir(image_folder)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()

        # Apply frame filter if set
        if _frame_name_filter is not None:
            frame_names = [f for f in frame_names if f in _frame_name_filter]

        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {image_folder}")
        img_paths = [os.path.join(image_folder, fn) for fn in frame_names]
        img_mean_t = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        img_std_t = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

        if async_loading_frames:
            lazy_images = sam3_io.AsyncImageFrameLoader(
                img_paths, image_size, offload_video_to_cpu, img_mean_t, img_std_t
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
        video_height, video_width = None, None
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (image folder, patched)")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            images = images.cuda()
            img_mean_t = img_mean_t.cuda()
            img_std_t = img_std_t.cuda()
        images -= img_mean_t
        images /= img_std_t
        return images, video_height, video_width

    sam3_io.load_video_frames_from_image_folder = patched_load_from_image_folder
    sam3_io._frame_filter_patch_applied = True
    logger.info("Applied frame-filter monkey-patch to sam3.model.io_utils.")


def _to_relative_box(box, width, height):
    """Convert absolute pixel box [x1, y1, x2, y2] to relative [0-1] coords."""
    import numpy as np
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 1:
        box = box.reshape(1, 4)
    rel = box.copy()
    rel[:, [0, 2]] /= width
    rel[:, [1, 3]] /= height
    return rel


def _to_relative_points(points, width, height):
    """Convert absolute pixel points [[x, y], ...] to relative [0-1] coords."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    rel = points.copy()
    rel[..., 0] /= width
    rel[..., 1] /= height
    return rel


class _Sam3TextDetectMixin:
    """Open-vocabulary person detection via SAM3's own detector.

    Both SAM3 video adapters already hold a ``Sam3ImageOnVideoMultiGPU`` detector
    (the tracker shares its backbone; the session predictor exposes it at
    ``model.detector``). Pointing ``self._detector`` at it lets us text-detect
    "person" on a frame with **no extra model load** — a drop-in replacement for
    YOLO that returns the same ``(bbox_xyxy, score)`` the pipeline expects, so
    anchors / bootstrap / cross-camera matching are reused unchanged.
    """

    _detector = None
    _text_proc = None

    def text_detect(self, image_rgb, text="person", conf_thresh=0.5):
        import numpy as np
        from PIL import Image
        if self._detector is None:
            raise RuntimeError("text_detect requires self._detector (a SAM3 detector) to be set")
        if self._text_proc is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            # Build on the detector's OWN device, not Sam3Processor's hardcoded
            # "cuda" default — so text detection works on CPU/MPS hosts too.
            try:
                dev = next(self._detector.parameters()).device
            except StopIteration:
                dev = "cuda"
            self._text_proc = Sam3Processor(self._detector, device=dev)
        # Honor the caller's threshold per call. Sam3Processor hard-filters boxes
        # at self.confidence_threshold *inside* grounding (default 0.5), and we
        # cache the processor — so without this, the pipeline's lower-threshold
        # fallback (0.4/0.25/0.15 on hard frames) would be a silent no-op for SAM3.
        # Set it before set_text_prompt (which runs grounding).
        self._text_proc.confidence_threshold = float(conf_thresh)
        # Pass a PIL image: Sam3Processor.set_image reads dims as image.shape[-2:]
        # for arrays, which mis-reads (H,W,3) numpy as width=3 → collapsed x-scale.
        # PIL.size is read correctly, so boxes come back in true pixel coords.
        pil = image_rgb if isinstance(image_rgb, Image.Image) else Image.fromarray(np.ascontiguousarray(image_rgb))
        st = self._text_proc.set_image(pil)
        st = self._text_proc.set_text_prompt(text, st)
        boxes, scores = st.get("boxes"), st.get("scores")
        out = []
        if boxes is None or scores is None:
            return out
        for b, s in zip(boxes, scores):
            sc = float(s)
            if sc < conf_thresh:
                continue
            bb = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
            out.append((bb.astype(np.float32).reshape(-1)[:4], sc))
        return out


class _Sam3VideoAdapter(_Sam3TextDetectMixin):
    """
    Adapter wrapping SAM3's tracker to match SAM2's video predictor API.

    Key differences handled:
    - Build: use build_sam3_video_model().tracker with shared detector backbone
    - Coordinates: SAM3 expects relative [0-1], our pipeline passes absolute pixels
    - Reset: SAM3 uses clear_all_points_in_video() instead of reset_state()
    - Propagate: SAM3 yields 5-tuple, SAM2 yields 3-tuple
    """

    def __init__(self, sam3_model, keep_detector=False):
        # Per official sam3_for_sam2_video_task_example.ipynb:
        # use the tracker directly, share backbone from detector
        self._tracker = sam3_model.tracker
        self._tracker.backbone = sam3_model.detector.backbone
        self._video_width = None
        self._video_height = None
        # sam3_prompt_light: keep the detector for text_detect (mixin) — the lean
        # alternative to the 16x multiplex session predictor.
        self._detector = sam3_model.detector if keep_detector else None
        self._text_proc = None

    def init_state(self, video_path, **kwargs):
        state = self._tracker.init_state(video_path=video_path, **kwargs)
        # Cache video dims for coordinate conversion
        self._video_height = state["video_height"]
        self._video_width = state["video_width"]
        return state

    def reset_state(self, inference_state):
        self._tracker.clear_all_points_in_video(inference_state)

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        box=None,
        **kwargs,
    ):
        # Convert absolute pixel coords -> relative [0-1] for SAM3
        if self._video_width is None or self._video_height is None:
            raise RuntimeError("Must call init_state() before add_new_points_or_box()")
        if box is not None:
            box = _to_relative_box(box, self._video_width, self._video_height)
        if points is not None:
            points = _to_relative_points(points, self._video_width, self._video_height)

        result = self._tracker.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box,
            **kwargs,
        )
        # SAM3 tracker returns 5-tuple: (frame_idx, obj_ids, low_res, video_res, scores)
        # SAM2 returns 3-tuple: (frame_idx, obj_ids, mask_logits)
        # Return video_res as mask_logits (same resolution as SAM2's output)
        _, out_obj_ids, _low_res, video_res_masks, *_ = result
        return _, out_obj_ids, video_res_masks

    def propagate_in_video(self, inference_state, reverse=False,
                           start_frame_idx=None, max_frame_num_to_track=None,
                           **kwargs):
        """
        Adapt SAM3 tracker's 5-tuple yield to SAM2's 3-tuple.

        start_frame_idx=None lets SAM3 auto-detect the first annotation frame,
        which is critical for reverse propagation (backward from the anchor
        frame). Explicit sub-range args are FORWARDED (collapse recovery
        re-propagates only a run's frames; silently swallowing these kwargs
        used to turn every recovery into a full-clip pass on this backend).
        propagate_preflight=True is only effective on the first call
        (consolidates temp outputs); subsequent calls are a no-op.
        """
        for result in self._tracker.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            propagate_preflight=True,
        ):
            frame_idx, obj_ids, _low_res, video_res_masks, _scores = result
            yield frame_idx, obj_ids, video_res_masks


class _HostImageBatchView:
    """List-like view over a host-resident decoded-frame tensor that returns
    DEVICE slices on indexing.

    SAM3's video inference packs the whole video into ``BatchedDatapoint.
    img_batch`` and copies it to the GPU wholesale, but every consumer only
    ever reads per-frame slices (``img_batch[img_ids]``), and the image model
    already branches on non-tensor batches. Keeping the tensor on host and
    moving each requested slice bounds VRAM to O(slice) instead of O(video).
    """

    # Pin only small batches: pin_memory() COPIES the tensor, so pinning a
    # multi-GB video transiently doubles host RAM for a ~1 ms/frame transfer
    # saving that the model step dwarfs anyway.
    _PIN_MAX_BYTES = 2 << 30

    def __init__(self, images, device):
        import torch  # noqa: F401 — module avoids a top-level torch import
        if images.nbytes <= self._PIN_MAX_BYTES:
            try:
                images = images.pin_memory()
            except RuntimeError:
                pass  # pinning is a throughput optimization, never a requirement
        self._images = images
        self.device = device

    def __len__(self):
        return len(self._images)

    @property
    def shape(self):
        return self._images.shape

    @property
    def dtype(self):
        return self._images.dtype

    def __getitem__(self, ids):
        import torch
        if torch.is_tensor(ids):
            ids = ids.to("cpu")
        return self._images[ids].to(self.device, non_blocking=True)

    def to(self, *args, **kwargs):
        return self  # absorb further copy attempts; slices move on demand


def _patch_sam3_prompt_host_batch():
    """Keep ``BatchedDatapoint.img_batch`` on the host during sam3_prompt
    session start, honoring ``offload_video_to_cpu`` end to end (the session
    request only offloads INGESTION; without this the batch construction
    re-uploads the full video tensor to the GPU)."""
    try:
        from sam3.model import sam3_video_inference as svi
    except Exception:
        return
    if getattr(svi, "_mamma_host_img_batch", False):
        return
    orig = svi.copy_data_to_device

    def copy_keep_img_batch_on_host(data, device, *args, **kwargs):
        import torch
        img = getattr(data, "img_batch", None)
        if img is not None and torch.is_tensor(img) and img.dim() == 4 \
                and img.device.type == "cpu" and str(device) != "cpu":
            import dataclasses
            rest = {f.name: getattr(data, f.name) for f in dataclasses.fields(data)
                    if f.name != "img_batch"}
            rest = {k: orig(v, device, *args, **kwargs) for k, v in rest.items()}
            return type(data)(img_batch=_HostImageBatchView(img, device), **rest)
        return orig(data, device, *args, **kwargs)

    svi.copy_data_to_device = copy_keep_img_batch_on_host
    svi._mamma_host_img_batch = True
    logger.info("sam3_prompt: video batch stays on host; frames stream to GPU per slice.")


class _Sam3PromptVideoAdapter(_Sam3TextDetectMixin):
    """
    Adapter wrapping SAM3's video predictor (session-based API with text prompts).

    This uses the new SAM3 API (build_sam3_video_predictor) which supports:
    - Text prompts: "person" auto-detects and tracks all people with instance IDs
    - Point/box prompts: for manual/matched prompts on non-init cameras
    - Bidirectional propagation: forward + backward in one call

    The session-based API is fundamentally different from the SAM2-compatible
    tracker API (_Sam3VideoAdapter), so this adapter exposes its own interface.
    """

    def __init__(self, checkpoint_path=None, tracking_overrides=None):
        from sam3.model_builder import build_sam3_video_predictor  # type: ignore
        import torch
        gpus = list(range(torch.cuda.device_count())) or [0]
        kwargs = {"gpus_to_use": gpus}
        # Only pass checkpoint_path if it's a real file path (not a HuggingFace ID).
        # When None or a HF ID like "facebook/sam3", let SAM3 handle the download.
        if checkpoint_path and os.path.isfile(checkpoint_path):
            kwargs["checkpoint_path"] = checkpoint_path
        # As used in https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb
        self._predictor = build_sam3_video_predictor(**kwargs)
        self._apply_tracking_overrides(tracking_overrides)
        self._session_id = None
        self._video_width = None
        self._video_height = None
        # Reuse the predictor's own (already-loaded) image detector for text_detect
        # (mixin) — replaces the redundant YOLO call in the multi-view bootstrap,
        # with no extra model load.
        self._detector = getattr(getattr(self._predictor, "model", None), "detector", None)
        self._text_proc = None

    def _apply_tracking_overrides(self, overrides):
        """Override SAM3 tracking config on the live model after construction.

        Fields are instance attributes on predictor.model (e.g. new_det_thresh,
        trk_assoc_iou_thresh) and are read fresh on every frame during inference,
        so overriding them here takes effect immediately.
        """
        if not overrides:
            return
        model = getattr(self._predictor, "model", None)
        if model is None:
            logger.warning("SAM3 tracking overrides: cannot find predictor.model, skipping.")
            return
        applied = []
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(model, key):
                old = getattr(model, key)
                setattr(model, key, value)
                applied.append(f"{key}: {old} -> {value}")
            else:
                logger.warning(f"SAM3 tracking override ignored: '{key}' not found on model.")
        if applied:
            logger.info("SAM3 tracking overrides applied: " + ", ".join(applied))

    def start_session(self, video_path, offload_video_to_cpu=False):
        """Start a session on a video (directory of frames or MP4).

        ``offload_video_to_cpu`` is forwarded to the session request (bounds
        INGESTION VRAM: the upstream default uploads the whole fp16 video
        tensor at decode) and additionally keeps the tracking batch's
        ``img_batch`` on host (see ``_patch_sam3_prompt_host_batch``), so
        session VRAM is bounded by per-frame slices end to end.
        """
        if offload_video_to_cpu:
            _patch_sam3_prompt_host_batch()
        response = self._predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path,
                         offload_video_to_cpu=bool(offload_video_to_cpu))
        )
        self._session_id = response["session_id"]
        # Cache video dimensions from the inference state
        session = self._predictor._get_session(self._session_id)
        if session:
            state = session["state"]
            self._video_height = state["orig_height"]
            self._video_width = state["orig_width"]
        return self._session_id

    def add_text_prompt(self, frame_idx, text="person"):
        """Add a text prompt (e.g. 'person') to detect all instances on a frame.

        Returns:
            Dict with 'obj_ids' (list of int), 'masks' (dict obj_id -> bool tensor),
            'boxes' (dict obj_id -> [x1,y1,x2,y2] absolute pixels).
        """
        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=frame_idx,
                text=text,
            )
        )
        return self._parse_outputs(response.get("outputs"))

    def add_box_prompt(self, frame_idx, obj_id, box_xyxy):
        """Add a box prompt for a specific object on a frame.

        Args:
            box_xyxy: [x1, y1, x2, y2] in absolute pixel coordinates.
        """
        import numpy as np
        box = np.asarray(box_xyxy, dtype=np.float32)
        # Convert xyxy to xywh normalized [0-1]
        w_video = self._video_width or 1
        h_video = self._video_height or 1
        cx = (box[0] + box[2]) / 2.0 / w_video
        cy = (box[1] + box[3]) / 2.0 / h_video
        w = (box[2] - box[0]) / w_video
        h = (box[3] - box[1]) / h_video
        import torch
        boxes_xywh = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
        box_labels = torch.tensor([True])

        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=frame_idx,
                bounding_boxes=boxes_xywh,
                bounding_box_labels=box_labels,
                obj_id=obj_id,
            )
        )
        return self._parse_outputs(response.get("outputs"))

    def _cached_frame_outputs(self):
        """Return the session's cached_frame_outputs dict, or None."""
        try:
            sess = self._predictor._get_session(self._session_id)
            return sess["state"].get("cached_frame_outputs") if sess else None
        except Exception:
            return None

    def propagate(self, sink=None, prune_cache_window=None, keep_frames=()):
        """Propagate all prompts bidirectionally.

        If ``sink`` is provided it is called as ``sink(frame_idx, {obj_id: mask})``
        for each frame and nothing is accumulated — bounded memory, used by the
        pipeline to stream masks straight to a disk-backed store.
        Otherwise the full ``{frame_idx: {obj_id: mask}}`` dict is returned
        (back-compatible).

        prune_cache_window (opt-in, default None = off): SAM3's session stores
        every frame's outputs in ``inference_state["cached_frame_outputs"]`` and
        never prunes them during propagation, so VRAM grows ~5x faster than the
        sam3 tracker path and OOMs on long clips. When set to an int W, entries
        whose frame index is farther than W from the frame just emitted are
        dropped (keep_frames — e.g. the prompt frame — are always retained), so
        peak cache is bounded to ~2W frames. We already stream each frame's masks
        out via ``sink`` and never re-query, so this is dead weight here — but the
        prune is gated and parity-checked because the bidirectional pass can
        re-read cached frames.
        """
        import numpy as np
        video_segments = {} if sink is None else None
        keep = {int(f) for f in (keep_frames or ())}
        cfo = self._cached_frame_outputs() if prune_cache_window is not None else None
        W = int(prune_cache_window) if prune_cache_window is not None else None
        # As used in https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb
        for response in self._predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self._session_id,
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response.get("outputs")
            if outputs is None:
                continue
            parsed = self._parse_outputs(outputs)
            masks = {
                oid: (mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask))
                for oid, mask in parsed["masks"].items()
            }
            if sink is not None:
                sink(frame_idx, masks)
            else:
                video_segments[frame_idx] = masks
            # Bound the session output cache: drop entries outside the
            # sliding window around the just-emitted frame; never drop keep_frames.
            if cfo is not None:
                lo, hi = frame_idx - W, frame_idx + W
                for fi in [k for k in cfo if k not in keep and (k < lo or k > hi)]:
                    cfo.pop(fi, None)
        return None if sink is not None else dict(sorted(video_segments.items()))

    def reset_session(self):
        """Reset the current session."""
        if self._session_id:
            self._predictor.handle_request(
                request=dict(type="reset_session", session_id=self._session_id)
            )

    def close_session(self):
        """Close the current session and free GPU resources."""
        if self._session_id:
            self._predictor.handle_request(
                request=dict(type="close_session", session_id=self._session_id)
            )
            self._session_id = None

    def shutdown(self):
        """Shutdown the predictor entirely."""
        self.close_session()
        self._predictor.shutdown()

    def _parse_outputs(self, outputs):
        """Parse SAM3 video predictor outputs into a usable format."""
        import torch
        import numpy as np
        result = {"obj_ids": [], "masks": {}, "boxes": {}}
        if outputs is None:
            return result

        obj_ids = outputs.get("out_obj_ids", outputs.get("obj_ids", torch.zeros(0, dtype=torch.int64)))
        masks = outputs.get("out_binary_masks", outputs.get("binary_masks", torch.zeros(0)))
        boxes_xywh = outputs.get("out_boxes_xywh", outputs.get("boxes_xywh", torch.zeros(0, 4)))

        # Ensure obj_ids is iterable as a list of ints
        if hasattr(obj_ids, 'tolist'):
            obj_ids_list = obj_ids.tolist()
        elif isinstance(obj_ids, (list, tuple)):
            obj_ids_list = list(obj_ids)
        else:
            obj_ids_list = [int(obj_ids)]

        w_video = self._video_width or 1
        h_video = self._video_height or 1

        n_masks = masks.shape[0] if hasattr(masks, 'shape') else 0
        n_boxes = boxes_xywh.shape[0] if hasattr(boxes_xywh, 'shape') else 0

        for i, oid in enumerate(obj_ids_list):
            result["obj_ids"].append(oid)
            if i < n_masks:
                result["masks"][oid] = masks[i]  # bool array or tensor [H, W]
            if i < n_boxes:
                # Convert normalized xywh to absolute xyxy
                bx = boxes_xywh[i]
                cx = float(bx[0])
                cy = float(bx[1])
                bw = float(bx[2])
                bh = float(bx[3])
                x1 = (cx - bw / 2) * w_video
                y1 = (cy - bh / 2) * h_video
                x2 = (cx + bw / 2) * w_video
                y2 = (cy + bh / 2) * h_video
                result["boxes"][oid] = np.array([x1, y1, x2, y2], dtype=np.float32)
        return result

    @property
    def video_width(self):
        return self._video_width

    @property
    def video_height(self):
        return self._video_height


def build_video_predictor(sam_version: str, config: str | None, checkpoint: str | None, device,
                          tracking_overrides: dict | None = None) -> object:
    """
    Build a SAM video predictor for the given backend version.

    Args:
        sam_version: "sam2" or "sam3"
        config: Model config path (required for sam2, ignored for sam3)
        checkpoint: Model checkpoint path or HuggingFace model ID
        device: torch device
        tracking_overrides: Optional dict of SAM3 tracking config overrides
            (e.g. new_det_thresh, trk_assoc_iou_thresh). Only used for sam3_prompt.

    Returns:
        A video predictor object with init_state / add_new_points_or_box /
        propagate_in_video / reset_state API (SAM2-compatible interface).
    """
    if sam_version == "sam2":
        _patch_sam2_png_support()
        _patch_sam2_video_lazy()
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        return build_sam2_video_predictor(config, checkpoint, device=device)

    elif sam_version in ("sam3", "sam3_prompt_light"):
        # Both use the lean SAM3 tracker (build_sam3_video_model). sam3 detects
        # with YOLO; sam3_prompt_light text-detects "person" via the model's own
        # detector (keep_detector=True) — no YOLO, no 16x multiplex predictor.
        _patch_sam3_png_support()
        _patch_sam3_video_lazy()
        from sam3.model_builder import build_sam3_video_model  # type: ignore
        sam3_model = build_sam3_video_model()
        return _Sam3VideoAdapter(sam3_model, keep_detector=(sam_version == "sam3_prompt_light"))

    elif sam_version == "sam3_prompt":
        _patch_sam3_png_support()
        return _Sam3PromptVideoAdapter(checkpoint_path=checkpoint, tracking_overrides=tracking_overrides)

    else:
        raise ValueError(
            f"Unknown sam_version: {sam_version!r}. Expected 'sam2', 'sam3', "
            "'sam3_prompt', or 'sam3_prompt_light'.")
