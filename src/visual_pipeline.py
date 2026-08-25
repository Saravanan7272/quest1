"""
src/visual_pipeline.py
Visual-only discovery pipeline orchestrating Visual Scout, Periodic Detector,
Dense Sampling, Text Detection, IoU Tracking, and OCR Recognition.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.models import Candidate, ModalityScores, SearchStats
from src.visual_scout import detect_visual_changes, generate_periodic_triggers, merge_triggers
from src.sampling import generate_sample_timestamps, extract_frames_at_timestamps
from src.text_detector import TextDetector
from src.text_tracker import TextTracker
from src.ocr import OCREngineAdapter
from src.matching import compute_ocr_score

logger = logging.getLogger(__name__)

def run_visual_discovery(
    video_path: Path,
    target_text: str,
    duration: float,
    fps: float,
    config: Dict[str, Any],
    temp_dir: Path
) -> Tuple[List[Candidate], SearchStats]:
    """
    Run Visual-Only Discovery Path:
    Level 1: Scout (1 FPS) + Periodic (0.5 FPS) -> Merge Triggers (2s)
    Level 2: Dense Sampling (10 FPS) -> Text Detection -> IoU Tracking
    Level 3: Full OCR Recognition on Best Track Frames
    """
    stats = SearchStats()
    visual_cfg = config.get("visual_scout", {})
    sampling_cfg = config.get("sampling", {})
    matching_cfg = config.get("matching", {})
    track_cfg = config.get("tracking", {})
    ocr_cfg = config.get("ocr", {})

    sample_fps = visual_cfg.get("sample_fps", 1.0)
    periodic_fps = visual_cfg.get("periodic_detection_fps", 0.5)
    merge_window = visual_cfg.get("trigger_merge_window", 2.0)
    dense_fps = sampling_cfg.get("dense_fps", 10.0)

    # 1. Coarse Sampling for Change Scout
    coarse_timestamps = generate_sample_timestamps(0.0, duration, coarse_fps=sample_fps)
    coarse_frames_dir = temp_dir / "coarse_frames"
    coarse_records = extract_frames_at_timestamps(video_path, coarse_timestamps, coarse_frames_dir)
    stats.scout_frames = len(coarse_records)

    scout_frame_paths = [r.path for r in coarse_records]
    scout_ts_list = [r.timestamp for r in coarse_records]

    # Change Scout & Periodic Triggers
    scout_triggers = detect_visual_changes(scout_frame_paths, scout_ts_list, threshold=visual_cfg.get("threshold", 30.0))
    periodic_triggers = generate_periodic_triggers(duration, periodic_fps=periodic_fps)
    stats.detector_frames = len(periodic_triggers)

    merged_triggers = merge_triggers(scout_triggers, periodic_triggers, merge_window=merge_window)
    logger.info(f"Visual Discovery: Merged {len(merged_triggers)} trigger windows.")

    if not merged_triggers:
        return [], stats

    # 2. Dense Sampling around Triggers
    dense_records = []
    dense_frames_dir = temp_dir / "dense_frames"
    for trig in merged_triggers:
        t_start = max(0.0, trig - 0.5)
        t_end = min(duration, trig + 1.5)
        d_ts = generate_sample_timestamps(t_start, t_end, coarse_fps=dense_fps)
        recs = extract_frames_at_timestamps(video_path, d_ts, dense_frames_dir)
        dense_records.extend(recs)

    # 3. Text Detection & IoU Tracking
    detector = TextDetector(min_confidence=config.get("text_detector", {}).get("min_confidence", 0.30))
    tracker = TextTracker(
        iou_threshold=track_cfg.get("iou_threshold", 0.5),
        max_gap_seconds=track_cfg.get("max_gap_seconds", 0.5)
    )

    for rec in dense_records:
        detected_boxes = detector.detect(rec.path, rec.timestamp)
        tracker.update(rec.timestamp, detected_boxes)

    # Finalize tracks
    tracked_events = tracker.finalize()
    stats.tracked_events = len(tracked_events)
    logger.info(f"Visual Discovery: Tracked {len(tracked_events)} distinct text events.")

    # 4. Level 3 OCR Recognition on Tracked Best Frames
    ocr_engine = OCREngineAdapter(lang=ocr_cfg.get("lang", "en"), min_confidence=ocr_cfg.get("min_confidence", 0.30))
    candidates: List[Candidate] = []

    char_w = matching_cfg.get("character_weight", 0.60)
    tok_w = matching_cfg.get("token_weight", 0.40)

    for track in tracked_events:
        # Find frame closest to track.best_frame_timestamp
        best_rec = min(dense_records, key=lambda r: abs(r.timestamp - track.best_frame_timestamp), default=None)
        if not best_rec:
            continue

        stats.ocr_calls += 1
        ocr_boxes = ocr_engine.run_ocr(best_rec.path)
        
        for box in ocr_boxes:
            score = compute_ocr_score(box.text, target_text, character_weight=char_w, token_weight=tok_w)
            query_relevance = float(box.confidence) * score
            tracker.update_best_frame(track.track_id, best_rec.timestamp, box.text, query_relevance)

            if score >= matching_cfg.get("similarity_threshold", 0.75):
                frame_num = int(round(best_rec.timestamp * fps))
                cand = Candidate(
                    timestamp=round(best_rec.timestamp, 3),
                    frame_number=frame_num,
                    text=box.text,
                    scores=ModalityScores(ocr=score),
                    bbox=box.bounding_rect,
                    image_path=str(best_rec.path),
                    source="visual"
                )
                candidates.append(cand)

    stats.candidates_found = len(candidates)
    return candidates, stats
