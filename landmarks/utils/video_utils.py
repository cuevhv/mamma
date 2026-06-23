import glob
import os

import imageio_ffmpeg


# Prefer imageio-ffmpeg's bundled/static binary (or its PATH/sys.prefix
# discovery) over a hard-coded /usr/bin/ffmpeg. This matches the segmentation
# stack and avoids per-machine path assumptions.
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def create_video_from_images(images_folder, output_path, img_format="img_%04d.jpg"):
    """Encode the numbered frames under ``images_folder`` into an MP4 via ffmpeg.

    Returns True on success, False when the input folder has no frames
    matching ``img_format`` (silent no-op — the preview is optional) or
    when ffmpeg itself errors out (logged with ffmpeg's stderr so the
    real failure is visible, not just the Python wrapper exit code).
    """
    import subprocess

    input_path = os.path.join(images_folder, img_format)

    # Skip the call entirely when there are no input frames on disk —
    # otherwise ffmpeg returns rc=1 and the bare except below would
    # mask a benign "preview was disabled" condition as an "error".
    sample_glob = os.path.join(
        images_folder,
        img_format.replace("%04d", "*"),
    )
    if not glob.glob(sample_glob):
        return False

    ffmpeg_command = [
        FFMPEG_EXE,
        "-framerate", "30",
        "-i", input_path,
        "-vf", "scale=iw:ih",
        "-c:v", "libx264",
        "-crf", "23",
        "-r", "30",
        "-pix_fmt", "yuv420p", "-y",
        output_path,
    ]

    try:
        # capture_output so ffmpeg's stderr surfaces in the message below
        # instead of the python-wrapper "non-zero exit status N" alone.
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Video successfully created: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip().splitlines()[-5:]
        print(
            f"Error occurred while creating the video {output_path!r}: "
            f"ffmpeg rc={e.returncode}. Last stderr lines:\n  "
            + "\n  ".join(stderr_tail)
        )
        return False
    except FileNotFoundError:
        print(f"Error occurred while creating the video {output_path!r}: ffmpeg binary not found at {FFMPEG_EXE!r}")
        return False