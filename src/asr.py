"""
src/asr.py
Automatic Speech Recognition (ASR) temporal candidate search using faster-whisper.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel
from src.matching import compute_asr_score

logger = logging.getLogger(__name__)

@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    score: float

@dataclass
class ASRSearchResult:
    candidate_found: bool
    best_score: float
    start_time: float
    end_time: float
    best_segment_text: str
    all_segments: List[ASRSegment]

_whisper_model_instance = None
_cached_model_key = None

def get_whisper_model(
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"
) -> WhisperModel:
    """
    Singleton factory for loading faster-whisper WhisperModel once and reusing it.
    """
    global _whisper_model_instance, _cached_model_key
    key = (model_name, device, compute_type)
    
    if _whisper_model_instance is None or _cached_model_key != key:
        logger.info(f"Loading faster-whisper model '{model_name}' (device={device}, compute_type={compute_type})...")
        _whisper_model_instance = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        _cached_model_key = key
        logger.info("Whisper model loaded successfully.")
        
    return _whisper_model_instance

def search_audio_for_target(
    audio_video_path: Path,
    target_text: str,
    video_duration: float,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    window_padding: float = 5.0,
    candidate_threshold: float = 0.60,
    partial_weight: float = 0.70,
    coverage_weight: float = 0.30
) -> ASRSearchResult:
    """
    Transcribe audio using faster-whisper, evaluate segments against target text,
    and derive candidate search window with temporal padding.
    """
    model = get_whisper_model(model_name=model_name, device=device, compute_type=compute_type)
    
    logger.info(f"Transcribing '{audio_video_path}' with faster-whisper (word_timestamps=True)...")
    try:
        segments_gen, info = model.transcribe(
            str(audio_video_path),
            language=language,
            word_timestamps=True,
            beam_size=5
        )
        segments_list = list(segments_gen)
    except Exception as e:
        logger.warning(f"ASR transcription failed or produced an error: {e}")
        segments_list = []

    parsed_segments: List[ASRSegment] = []
    best_seg: Optional[ASRSegment] = None
    best_score = 0.0

    for seg in segments_list:
        score = compute_asr_score(
            segment_text=seg.text,
            target_text=target_text,
            partial_weight=partial_weight,
            coverage_weight=coverage_weight
        )
        asr_seg = ASRSegment(start=seg.start, end=seg.end, text=seg.text.strip(), score=score)
        parsed_segments.append(asr_seg)
        
        if score > best_score:
            best_score = score
            best_seg = asr_seg

    if best_seg and best_score >= candidate_threshold:
        candidate_start = max(0.0, best_seg.start - window_padding)
        candidate_end = min(video_duration, best_seg.end + window_padding)
        logger.info(
            f"ASR Candidate FOUND! Score={best_score:.4f}, Segment='{best_seg.text}', "
            f"Candidate Window=[{candidate_start:.2f}s, {candidate_end:.2f}s]"
        )
        return ASRSearchResult(
            candidate_found=True,
            best_score=best_score,
            start_time=candidate_start,
            end_time=candidate_end,
            best_segment_text=best_seg.text,
            all_segments=parsed_segments
        )
    else:
        # Fallback search window: search entire video if ASR candidate is weak or missing
        logger.info(
            f"No ASR segment reached threshold {candidate_threshold} (best={best_score:.4f}). "
            f"Falling back to full video window [0.0s, {video_duration:.2f}s]."
        )
        return ASRSearchResult(
            candidate_found=False,
            best_score=best_score,
            start_time=0.0,
            end_time=video_duration,
            best_segment_text=best_seg.text if best_seg else "",
            all_segments=parsed_segments
        )
