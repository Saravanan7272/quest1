"""
scripts/create_multimodal_sample_video.py
Combines the downloaded audio/video with burned-in subtitle text 'My mind rebels at stagnation'
to produce a local multimodal test video (tests/data/multimodal_sample.mp4).
"""

import sys
import logging
from pathlib import Path
import cv2
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import locate_dialogue_in_video
from src.acquisition import download_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("create_multimodal_video")

def create_multimodal_video():
    temp_dir = Path("temp_diagnostics/acq")
    input_video = temp_dir / "input_video.mp4"
    if not input_video.exists():
        download_video("https://youtu.be/dPTKl5H5ftg", temp_dir)

    out_dir = Path("tests/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    no_audio_video = out_dir / "temp_visual_overlay.mp4"
    final_multimodal = out_dir / "multimodal_sample.mp4"

    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(str(no_audio_video), fourcc, fps, (w, h))

    frame_idx = 0
    text = "My mind rebels at stagnation"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Burn in visual subtitle text on frames between 1.0s and 4.0s (ts = frame_idx / fps)
        ts = frame_idx / fps
        if 1.0 <= ts <= 4.0:
            overlay = frame.copy()
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 120
            cv2.rectangle(overlay, (w // 6, h - 200), (5 * w // 6, h - 80), (0, 0, 0), -1)
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            writer.write(overlay)
        else:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # Combine modified video stream with original audio stream using FFmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(no_audio_video),
        "-i", str(input_video),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(final_multimodal)
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    logger.info(f"Successfully generated local multimodal test video: {final_multimodal}")

if __name__ == "__main__":
    create_multimodal_video()
