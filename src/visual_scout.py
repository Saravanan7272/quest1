"""
src/visual_scout.py
Dual-path visual trigger generator featuring Visual Change Scout (1 FPS)
and Periodic Text Detector (0.5 FPS) with trigger merging.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def detect_visual_changes(
    frame_paths: List[Path],
    timestamps: List[float],
    threshold: float = 30.0
) -> List[float]:
    """
    Visual Change Scout: Compute mean absolute frame difference between consecutive frames.
    Returns timestamps where visual change exceeds threshold.
    """
    triggers: List[float] = []
    if len(frame_paths) < 2:
        return triggers

    prev_gray = None

    for path, ts in zip(frame_paths, timestamps):
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff >= threshold:
                triggers.append(ts)
                
        prev_gray = gray

    return triggers

def generate_periodic_triggers(
    duration: float,
    periodic_fps: float = 0.5
) -> List[float]:
    """
    Periodic Detector (Recall Insurance): Triggers periodic sampling every 1/periodic_fps seconds.
    """
    if duration <= 0:
        return [0.0]
    step = 1.0 / max(0.1, periodic_fps)
    return list(np.arange(0.0, duration + 1e-5, step))

def merge_triggers(
    scout_triggers: List[float],
    periodic_triggers: List[float],
    merge_window: float = 2.0
) -> List[float]:
    """
    Merge and deduplicate triggers from Visual Change Scout and Periodic Detector
    within a merge_window (default 2.0 seconds).
    """
    combined = sorted(list(set(scout_triggers + periodic_triggers)))
    if not combined:
        return []

    merged: List[float] = []
    for ts in combined:
        if not merged:
            merged.append(ts)
        elif ts - merged[-1] >= merge_window:
            merged.append(ts)

    return merged
