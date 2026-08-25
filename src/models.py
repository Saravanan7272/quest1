"""
src/models.py
Central data model specifications for the production-oriented Video Dialogue Locator architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModalityScores:
    asr: Optional[float] = None      # 0.0-1.0, from faster-whisper match
    ocr: Optional[float] = None      # 0.0-1.0, from OCR text similarity
    semantic: Optional[float] = None # 0.0-1.0, from sentence-transformers

@dataclass
class TrackedBox:
    timestamp: float
    bbox: List[float]                # [x1, y1, x2, y2]
    detection_confidence: float
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None

@dataclass
class TextEvent:
    track_id: int
    start: float
    end: float
    boxes: List[TrackedBox] = field(default_factory=list)
    best_frame_timestamp: float = 0.0
    best_text: str = ""
    best_relevance: float = 0.0

@dataclass
class Candidate:
    timestamp: float
    frame_number: int
    text: str
    scores: ModalityScores
    bbox: Optional[List[float]] = None
    image_path: Optional[str] = None
    source: str = "unknown"          # "asr", "visual", "multimodal"
    fused_score: Optional[float] = None

@dataclass
class SearchStats:
    scout_frames: int = 0            # frames checked by visual change scout
    detector_frames: int = 0         # frames checked by periodic detector
    ocr_calls: int = 0               # actual OCR recognition calls
    candidates_found: int = 0        # candidates passing threshold
    tracked_events: int = 0          # distinct text events tracked
    runtime_seconds: float = 0.0
