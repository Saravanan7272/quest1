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
class ASRQuerySpan:
    segment_text: str = ""
    segment_start: float = 0.0
    segment_end: float = 0.0
    query_start: float = 0.0
    query_end: float = 0.0
    score: float = 0.0

@dataclass
class VisualTrackSpan:
    track_id: int = -1
    start: float = 0.0
    end: float = 0.0
    best_frame_timestamp: float = 0.0
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    query_similarity: float = 0.0

@dataclass
class EvidenceMetadata:
    timestamp_seconds: float = 0.0
    frame_number: int = 0
    image_path: str = ""
    source: List[str] = field(default_factory=list)

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
    speech_match: bool = False
    visual_text_match: bool = False
    sources: List[str] = field(default_factory=list)
    asr_span: Optional[ASRQuerySpan] = None
    visual_span: Optional[VisualTrackSpan] = None
    evidence: Optional[EvidenceMetadata] = None

@dataclass
class SearchStats:
    scout_frames: int = 0            # frames checked by visual change scout
    detector_frames: int = 0         # frames checked by periodic detector
    ocr_calls: int = 0               # actual OCR recognition calls
    candidates_found: int = 0        # candidates passing threshold
    tracked_events: int = 0          # distinct text events tracked
    runtime_seconds: float = 0.0
    visual_search_truncated: bool = False
