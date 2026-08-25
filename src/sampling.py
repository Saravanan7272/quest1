"""
src/sampling.py
Ultra-fast frame sampling module using single-pass FFmpeg batch extraction.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FrameRecord:
    path: Path
    index: int
    timestamp: float

def generate_sample_timestamps(start_time: float, end_time: float, coarse_fps: float = 1.0) -> np.ndarray:
    """
    Generate target timestamp array using np.arange at approximately coarse_fps step size.
    """
    if start_time >= end_time:
        return np.array([start_time])
    step = 1.0 / max(0.1, coarse_fps)
    epsilon = 1e-5
    return np.arange(start_time, end_time + epsilon, step)

def extract_frames_at_timestamps(
    video_path: Path,
    timestamps: np.ndarray,
    output_dir: Path
) -> List[FrameRecord]:
    """
    Extract video frames at specified timestamps using fast FFmpeg input seeking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_records: List[FrameRecord] = []
    
    if len(timestamps) == 0:
        return []

    t_min = float(timestamps[0])
    t_max = float(timestamps[-1])
    count = len(timestamps)

    if count >= 2 and (t_max - t_min) >= 0.1:
        duration = max(0.1, t_max - t_min)
        expected_fps = max(0.5, count / duration)
        
        # Batch extract in a single FFmpeg process
        out_pattern = str(output_dir / f"batch_{t_min:.1f}_%03d.jpg")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{t_min:.3f}",
            "-to", f"{t_max:.3f}",
            "-i", str(video_path),
            "-vf", f"fps={expected_fps:.2f}",
            "-q:v", "2",
            out_pattern
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            extracted_files = sorted(list(output_dir.glob(f"batch_{t_min:.1f}_*.jpg")))
            for idx, file_path in enumerate(extracted_files):
                ts = t_min + (idx / max(1, len(extracted_files) - 1)) * duration
                frame_records.append(FrameRecord(path=file_path, index=idx, timestamp=round(ts, 3)))
            if frame_records:
                return frame_records
        except Exception as e:
            logger.warning(f"Batch extraction fallback: {e}")

    # Fallback per-timestamp extraction
    for idx, ts in enumerate(timestamps):
        out_filename = f"frame_{idx:04d}_{ts:.3f}s.jpg"
        out_path = output_dir / out_filename
        
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if out_path.exists() and out_path.stat().st_size > 0:
                frame_records.append(FrameRecord(path=out_path, index=idx, timestamp=float(ts)))
        except Exception as e:
            logger.warning(f"Failed to extract frame at ts={ts:.3f}s: {e}")
            
    return frame_records
