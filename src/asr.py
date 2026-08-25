"""
src/asr.py
Automatic Speech Recognition (ASR) temporal candidate search using faster-whisper.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from faster_whisper import WhisperModel
from src.matching import compute_asr_score

logger = logging.getLogger(__name__)

import re

def _clean_token(t: str) -> str:
    return re.sub(r'[^\w]', '', t.lower())

def find_query_span(words: List[Any], target_text: str) -> Tuple[float, float]:
    """
    Extract estimated ASR query word span (query_start, query_end) from word timestamps.
    Falls back to segment bounds if words list is empty or unmatched.
    """
    if not words:
        return (0.0, 0.0)

    target_tokens = [_clean_token(t) for t in target_text.split() if _clean_token(t)]
    if not target_tokens:
        first_s = float(getattr(words[0], 'start', 0.0))
        last_e = float(getattr(words[-1], 'end', 0.0))
        return (first_s, last_e)

    word_records = []
    for w in words:
        token = _clean_token(getattr(w, 'word', ''))
        start = float(getattr(w, 'start', 0.0))
        end = float(getattr(w, 'end', 0.0))
        if token:
            word_records.append((token, start, end))

    if not word_records:
        first_s = float(getattr(words[0], 'start', 0.0))
        last_e = float(getattr(words[-1], 'end', 0.0))
        return (first_s, last_e)

    n_target = len(target_tokens)
    n_words = len(word_records)

    best_span = (word_records[0][1], word_records[-1][2])
    best_matches = -1

    for window_len in range(max(1, n_target - 2), min(n_words + 1, n_target + 3)):
        for i in range(n_words - window_len + 1):
            sub_words = word_records[i : i + window_len]
            sub_tokens = [w[0] for w in sub_words]
            matched = sum(1 for t in target_tokens if t in sub_tokens)
            if matched > best_matches:
                best_matches = matched
                best_span = (sub_words[0][1], sub_words[-1][2])

    return best_span

@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    score: float
    words: List[Any] = field(default_factory=list)

@dataclass
class ASRSearchResult:
    candidate_found: bool
    best_score: float
    start_time: float
    end_time: float
    asr_segment_start: float
    asr_segment_end: float
    asr_query_start: float
    asr_query_end: float
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
    best_seg_obj: Optional[Any] = None
    best_parsed_seg: Optional[ASRSegment] = None
    best_score = 0.0

    for seg in segments_list:
        score = compute_asr_score(
            segment_text=seg.text,
            target_text=target_text,
            partial_weight=partial_weight,
            coverage_weight=coverage_weight
        )
        words_list = list(getattr(seg, 'words', [])) if getattr(seg, 'words', None) else []
        asr_seg = ASRSegment(start=seg.start, end=seg.end, text=seg.text.strip(), score=score, words=words_list)
        parsed_segments.append(asr_seg)
        
        if score > best_score:
            best_score = score
            best_parsed_seg = asr_seg
            best_seg_obj = seg

    if best_parsed_seg and best_score >= candidate_threshold:
        q_start, q_end = find_query_span(best_parsed_seg.words, target_text)
        if q_start == 0.0 and q_end == 0.0:
            q_start, q_end = best_parsed_seg.start, best_parsed_seg.end

        logger.info("Whisper Word-Level Timestamps for Matched Segment:")
        for w in best_parsed_seg.words:
            w_text = getattr(w, 'word', str(w))
            w_s = float(getattr(w, 'start', 0.0))
            w_e = float(getattr(w, 'end', 0.0))
            logger.info(f"  Word '{w_text.strip()}': {w_s:.2f}s -> {w_e:.2f}s")

        candidate_start = max(0.0, q_start - window_padding)
        candidate_end = min(video_duration, q_end + window_padding)
        logger.info(
            f"ASR Candidate FOUND! Score={best_score:.4f}, Segment='{best_parsed_seg.text}', "
            f"Segment Window=[{best_parsed_seg.start:.2f}s, {best_parsed_seg.end:.2f}s], "
            f"Estimated Query Span=[{q_start:.2f}s, {q_end:.2f}s], "
            f"Search Window=[{candidate_start:.2f}s, {candidate_end:.2f}s]"
        )
        return ASRSearchResult(
            candidate_found=True,
            best_score=best_score,
            start_time=candidate_start,
            end_time=candidate_end,
            asr_segment_start=best_parsed_seg.start,
            asr_segment_end=best_parsed_seg.end,
            asr_query_start=q_start,
            asr_query_end=q_end,
            best_segment_text=best_parsed_seg.text,
            all_segments=parsed_segments
        )
    else:
        logger.info(
            f"No ASR segment reached threshold {candidate_threshold} (best={best_score:.4f}). "
            f"Falling back to full video window [0.0s, {video_duration:.2f}s]."
        )
        return ASRSearchResult(
            candidate_found=False,
            best_score=best_score,
            start_time=0.0,
            end_time=video_duration,
            asr_segment_start=0.0,
            asr_segment_end=video_duration,
            asr_query_start=0.0,
            asr_query_end=video_duration,
            best_segment_text=best_parsed_seg.text if best_parsed_seg else "",
            all_segments=parsed_segments
        )
