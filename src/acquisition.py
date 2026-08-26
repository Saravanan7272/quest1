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

def sanitize_url(url: str) -> str:
    """Sanitize URL by removing Markdown wrappers e.g. [http...](http...) or <http...>"""
    if not url:
        return ""
    url = url.strip()
    import re
    match = re.search(r'\[.*?\]\((https?://[^\s\)]+)\)', url)
    if match:
        return match.group(1)
    match = re.search(r'<(https?://[^\s>]+)>', url)
    if match:
        return match.group(1)
    return url

def download_video(url: str, output_dir: Path) -> VideoMetadata:
    """
    Download video from URL using yt-dlp into specified output directory
    and return parsed VideoMetadata.
    """
    url = sanitize_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "input_video.mp4")
    
    import urllib.parse
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.netloc:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    except Exception:
        pass

    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "retries": 10,
        "fragment_retries": 10,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
        "nocheckcertificate": True,
        "legacy_server_connect": True,
        "http_headers": headers
    }
    
    logger.info(f"Acquiring video from: {url}")
    local_path = Path(url)
    if local_path.exists() and local_path.is_file():
        target_file = output_dir / local_path.name
        shutil.copy2(local_path, target_file)
        metadata = get_video_metadata(target_file)
        file_size_mb = target_file.stat().st_size / (1024 * 1024)
        logger.info(f"Local video acquired ({file_size_mb:.2f} MiB): Duration={metadata.duration:.2f}s, FPS={metadata.fps:.2f}, Resolution={metadata.width}x{metadata.height}")
        return metadata

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except KeyboardInterrupt:
        raise AcquisitionError("Video download cancelled by user (KeyboardInterrupt).")
    except Exception as e:
        logger.warning(f"Primary yt-dlp format failed: {e}. Retrying with fallback format...")
        ydl_opts_fallback = {
            "outtmpl": out_template,
            "format": "b",
            "retries": 10,
            "fragment_retries": 10,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "overwrites": True,
            "nocheckcertificate": True,
            "legacy_server_connect": True,
            "http_headers": headers
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl_fb:
                ydl_fb.download([url])
        except KeyboardInterrupt:
            raise AcquisitionError("Video download cancelled by user (KeyboardInterrupt).")
        except Exception as e2:
            raise AcquisitionError(f"yt-dlp download failed for URL '{url}': {e2}")
        
    # Find downloaded file
    downloaded_files = list(output_dir.glob("input_video.*"))
    if not downloaded_files:
        raise AcquisitionError(f"No video file created in {output_dir} after download.")
        
    video_file = downloaded_files[0]
    file_size_mb = video_file.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded video ({file_size_mb:.2f} MiB)")
    
    metadata = get_video_metadata(video_file)
    if not metadata.has_audio:
        logger.warning(f"Media file has no audio stream detected.")

    logger.info(
        f"Metadata: Duration={metadata.duration:.2f}s, FPS={metadata.fps:.2f}, "
        f"Resolution={metadata.width}x{metadata.height}, HasAudio={metadata.has_audio}"
    )
    return metadata
