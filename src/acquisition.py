"""
src/acquisition.py
Handles video acquisition via yt-dlp and ffprobe metadata inspection.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yt_dlp

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    codec_name: str
    video_path: Path

class AcquisitionError(Exception):
    """Custom exception for acquisition phase failures."""
    pass

def check_ffprobe_available() -> bool:
    """Verify system ffprobe binary is accessible."""
    try:
        res = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False

def get_video_metadata(video_path: Path) -> VideoMetadata:
    """
    Extract video metadata using ffprobe.
    """
    if not check_ffprobe_available():
        raise AcquisitionError("FFprobe executable is not available on PATH.")
        
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
    except Exception as e:
        raise AcquisitionError(f"Failed to inspect metadata with ffprobe: {e}")
        
    streams = data.get("streams", [])
    format_info = data.get("format", {})
    
    video_stream = None
    has_audio = False
    
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio":
            has_audio = True
            
    if not video_stream:
        raise AcquisitionError(f"No video stream found in {video_path}")
        
    # Calculate duration
    duration = float(format_info.get("duration", 0.0))
    if duration == 0.0:
        duration = float(video_stream.get("duration", 0.0))
        
    # Parse FPS
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    try:
        num, den = map(float, r_frame_rate.split("/"))
        fps = num / den if den != 0 else 30.0
    except Exception:
        fps = 30.0
        
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec_name = video_stream.get("codec_name", "unknown")
    
    return VideoMetadata(
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
        codec_name=codec_name,
        video_path=video_path
    )

def download_video(url: str, output_dir: Path) -> VideoMetadata:
    """
    Download video from URL using yt-dlp into specified output directory
    and return parsed VideoMetadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "input_video.%(ext)s")
    
    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "overwrites": True
    }
    
    logger.info(f"Acquiring video from: {url}")
    local_path = Path(url)
    if local_path.exists() and local_path.is_file():
        target_file = output_dir / local_path.name
        shutil.copy2(local_path, target_file)
        metadata = get_video_metadata(target_file)
        logger.info(
            f"Local video acquired: Duration={metadata.duration:.2f}s, FPS={metadata.fps:.2f}, "
            f"Resolution={metadata.width}x{metadata.height}, HasAudio={metadata.has_audio}"
        )
        return metadata

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise AcquisitionError(f"yt-dlp download failed for URL '{url}': {e}")
        
    # Find downloaded file
    downloaded_files = list(output_dir.glob("input_video.*"))
    if not downloaded_files:
        raise AcquisitionError(f"No video file created in {output_dir} after download.")
        
    video_file = downloaded_files[0]
    logger.info(f"Successfully downloaded video to: {video_file}")
    
    metadata = get_video_metadata(video_file)
    logger.info(
        f"Metadata: Duration={metadata.duration:.2f}s, FPS={metadata.fps:.2f}, "
        f"Resolution={metadata.width}x{metadata.height}, HasAudio={metadata.has_audio}"
    )
    return metadata
