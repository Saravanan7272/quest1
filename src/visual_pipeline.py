"""
src/visual_pipeline.py
Visual discovery pipeline orchestrating Visual Scout, Periodic Detector,
Dense Sampling, Text Detection, IoU Tracking, and OCR Recognition.
Supports ASR-guided targeted search windowing and budget limits.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.models import Candidate, ModalityScores, SearchStats, VisualTrackSpan
from src.visual_scout import detect_visual_changes, generate_periodic_triggers, merge_triggers
from src.sampling import generate_sample_timestamps, extract_frames_at_timestamps
from src.text_detector import TextDetector
from src.text_tracker import TextTracker
from src.ocr import OCREngineAdapter
from src.matching import compute_ocr_score

logger = logging.getLogger(__name__)

def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0

def run_visual_discovery(
    video_path: Path,
    target_text: str,
    duration: float,
    fps: float,
    config: Dict[str, Any],
    temp_dir: Path,
    target_search_window: Optional[Tuple[float, float]] = None
) -> Tuple[List[Candidate], SearchStats]:
    """
    Run Visual Discovery Path (ASR-guided targeted search or visual-only fallback).
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
    dense_fps = sampling_cfg.get("dense_fps", 3.0)

    # Budget safeguards
    max_dense_frames = sampling_cfg.get("max_dense_frames", 300)
    max_ocr_tracks = visual_cfg.get("max_ocr_tracks", 20)

    t_start_all = time.time()

    # Determine visual search bounds
    if target_search_window:
        search_start = max(0.0, target_search_window[0])
        search_end = min(duration, target_search_window[1])
        logger.info(f"Visual Discovery: TARGETED ASR-guided search window [{search_start:.2f}s, {search_end:.2f}s]")
    else:
        search_start = 0.0
        search_end = duration
        logger.info(f"Visual Discovery: GLOBAL search window [0.0s, {search_end:.2f}s]")

    search_duration = search_end - search_start

    # 1. Coarse Sampling & Change Scout
    coarse_timestamps = generate_sample_timestamps(search_start, search_end, coarse_fps=sample_fps)
    coarse_frames_dir = temp_dir / "coarse_frames"
    coarse_records = extract_frames_at_timestamps(video_path, coarse_timestamps, coarse_frames_dir)
    stats.scout_frames = len(coarse_records)

    scout_frame_paths = [r.path for r in coarse_records]
    scout_ts_list = [r.timestamp for r in coarse_records]

    t_scout_start = time.time()
    scout_triggers = detect_visual_changes(scout_frame_paths, scout_ts_list, threshold=visual_cfg.get("threshold", 30.0))
    t_scout_dur = time.time() - t_scout_start

    t_per_start = time.time()
    periodic_triggers = generate_periodic_triggers(search_duration, periodic_fps=periodic_fps)
    periodic_triggers = [search_start + p for p in periodic_triggers if search_start + p <= search_end]
    stats.detector_frames = len(periodic_triggers)
    t_per_dur = time.time() - t_per_start

    t_merge_start = time.time()
    merged_triggers = merge_triggers(scout_triggers, periodic_triggers, merge_window=merge_window)
    t_merge_dur = time.time() - t_merge_start

    logger.info(f"--- VISUAL DISCOVERY STAGE 1: TRIGGERS ---")
    logger.info(f"raw_change_triggers count: {len(scout_triggers)}")
    logger.info(f"raw_periodic_triggers count: {len(periodic_triggers)}")
    logger.info(f"merged_trigger_windows count: {len(merged_triggers)}")

    if not merged_triggers:
        return [], stats

    # 2. Coarse Text Detection Pre-filtering to isolate text-bearing trigger windows
    detector = TextDetector(min_confidence=config.get("text_detector", {}).get("min_confidence", 0.30))
    text_active_triggers = []
    
    for trig in merged_triggers:
        closest_coarse = min(coarse_records, key=lambda r: abs(r.timestamp - trig), default=None)
        if closest_coarse:
            boxes = detector.detect(closest_coarse.path, closest_coarse.timestamp)
            if boxes:
                text_active_triggers.append(trig)

    active_targets = text_active_triggers if text_active_triggers else merged_triggers
    logger.info(f"Coarse text pre-filter: {len(text_active_triggers)} text-bearing windows identified out of {len(merged_triggers)} merged windows.")

    # 3. Dense Sampling around Text-Active Triggers (with timestamp deduplication & budget enforcement)
    t_dense_start = time.time()
    dense_frames_dir = temp_dir / "dense_frames"
    requested_dense_count = 0
    unique_dense_ts_set = set()
    dense_records = []

    for trig in active_targets:
        if len(dense_records) >= max_dense_frames:
            logger.warning(f"Visual Discovery: Dense frame limit reached ({max_dense_frames}). Truncating sampling.")
            stats.visual_search_truncated = True
            break
            
        t_start = max(search_start, trig - 0.5)
        t_end = min(search_end, trig + 1.5)
        d_ts = generate_sample_timestamps(t_start, t_end, coarse_fps=dense_fps)
        requested_dense_count += len(d_ts)
        
        new_ts = [ts for ts in d_ts if round(ts, 3) not in unique_dense_ts_set]
        for ts in new_ts:
            unique_dense_ts_set.add(round(ts, 3))

        if new_ts:
            recs = extract_frames_at_timestamps(video_path, new_ts, dense_frames_dir)
            dense_records.extend(recs)

    t_dense_dur = time.time() - t_dense_start
    dup_skipped = requested_dense_count - len(dense_records)

    logger.info(f"--- VISUAL DISCOVERY STAGE 2: DENSE SAMPLING ---")
    logger.info(f"dense_fps: {dense_fps}")
    logger.info(f"requested_dense_frames: {requested_dense_count}")
    logger.info(f"actual_dense_frames: {len(dense_records)}")

    # 4. Text Detection & IoU Tracking
    t_detect_start = time.time()
    tracker = TextTracker(
        iou_threshold=track_cfg.get("iou_threshold", 0.5),
        max_gap_seconds=track_cfg.get("max_gap_seconds", 0.5)
    )

    logger.info(f"--- VISUAL DISCOVERY STAGE 3: TEXT DETECTION & TRACKING ---")
    last_box_count = -1
    for rec in dense_records:
        detected_boxes = detector.detect(rec.path, rec.timestamp)
        curr_box_count = len(detected_boxes)
        if curr_box_count != last_box_count:
            last_box_count = curr_box_count
            if detected_boxes:
                conf_str = ", ".join([f"{_safe_float(getattr(b, 'detection_confidence', getattr(b, 'ocr_confidence', 0.0))):.2f}" for b in detected_boxes])
                logger.info(f"  Frame {rec.timestamp:.3f}s: boxes={curr_box_count}, confidences=[{conf_str}]")
            else:
                logger.info(f"  Frame {rec.timestamp:.3f}s: boxes=0 (text cleared)")
        tracker.update(rec.timestamp, detected_boxes)

    t_detect_dur = time.time() - t_detect_start

    t_track_start = time.time()
    tracked_events = tracker.finalize()
    if len(tracked_events) > max_ocr_tracks:
        logger.warning(f"Visual Discovery: Track count ({len(tracked_events)}) exceeded limit ({max_ocr_tracks}). Truncating tracks.")
        tracked_events = tracked_events[:max_ocr_tracks]
        stats.visual_search_truncated = True

    stats.tracked_events = len(tracked_events)
    t_track_dur = time.time() - t_track_start

    logger.info(f"--- VISUAL DISCOVERY STAGE 4: TRACK CREATION ---")
    logger.info(f"Tracked events count: {len(tracked_events)}")
    for track in tracked_events:
        duration_sec = track.end - track.start
        logger.info(
            f"  TRACK {track.track_id}: start={track.start:.2f}s, end={track.end:.2f}s, "
            f"duration={duration_sec:.2f}s, boxes={len(track.boxes)}, best_frame={track.best_frame_timestamp:.2f}s"
        )

    # 5. Full OCR Recognition on Tracked Best Frames
    t_ocr_start = time.time()
    ocr_engine = OCREngineAdapter(lang=ocr_cfg.get("lang", "en"), min_confidence=ocr_cfg.get("min_confidence", 0.30))
    candidates: List[Candidate] = []

    char_w = matching_cfg.get("character_weight", 0.60)
    tok_w = matching_cfg.get("token_weight", 0.40)
    sim_thresh = matching_cfg.get("ocr_min_threshold", 0.45)

    debug_export_dir = Path("outputs/debug_case_03")
    debug_export_dir.mkdir(parents=True, exist_ok=True)
    debug_tracks_info = []

    logger.info(f"--- VISUAL DISCOVERY STAGE 5: OCR RECONGNITION & EVALUATION ---")

    for track in tracked_events:
        track_recs = [r for r in dense_records if track.start - 0.2 <= r.timestamp <= track.end + 0.2]
        if not track_recs:
            continue

        sample_step = max(1, int(round(1.0 * dense_fps)))
        cand_recs = track_recs[::sample_step]
        if track_recs[-1] not in cand_recs:
            cand_recs.append(track_recs[-1])

        best_rec_for_track = cand_recs[0]
        best_track_text = ""
        best_track_conf = 0.0
        best_track_score = -1.0
        best_track_boxes = []

        for rec in cand_recs:
            stats.ocr_calls += 1
            ocr_boxes = ocr_engine.run_ocr(rec.path)
            for box in ocr_boxes:
                score = compute_ocr_score(box.text, target_text, character_weight=char_w, token_weight=tok_w)
                if score > best_track_score:
                    best_track_score = score
                    best_track_text = box.text
                    best_track_conf = float(box.confidence)
                    best_rec_for_track = rec
                    best_track_boxes = [box]

        v_span = VisualTrackSpan(
            track_id=track.track_id,
            start=track.start,
            end=track.end,
            best_frame_timestamp=best_rec_for_track.timestamp,
            ocr_text=best_track_text,
            ocr_confidence=best_track_conf,
            query_similarity=max(0.0, best_track_score)
        )

        if best_track_score >= sim_thresh:
            frame_num = int(round(best_rec_for_track.timestamp * fps))
            cand = Candidate(
                timestamp=round(best_rec_for_track.timestamp, 3),
                frame_number=frame_num,
                text=best_track_text,
                scores=ModalityScores(ocr=best_track_score),
                bbox=best_track_boxes[0].bounding_rect if best_track_boxes else None,
                image_path=str(best_rec_for_track.path),
                source="visual",
                visual_text_match=True,
                sources=["ocr"],
                visual_span=v_span
            )
            candidates.append(cand)

        logger.info(
            f"  TRACK {track.track_id:02d} | best_timestamp={best_rec_for_track.timestamp:.2f}s | "
            f"text='{best_track_text}' | ocr_conf={best_track_conf:.4f} | query_sim={best_track_score:.4f}"
        )

        debug_tracks_info.append({
            "track_id": track.track_id,
            "start": round(track.start, 2),
            "end": round(track.end, 2),
            "best_frame_timestamp": round(best_rec_for_track.timestamp, 2),
            "ocr_text": best_track_text,
            "ocr_confidence": round(best_track_conf, 4),
            "query_similarity": round(max(0.0, best_track_score), 4),
            "frame_image": str(best_rec_for_track.path).replace("\\", "/")
        })

    t_ocr_dur = time.time() - t_ocr_start

    with open(debug_export_dir / "tracks.json", "w", encoding="utf-8") as f_tr:
        json.dump(debug_tracks_info, f_tr, indent=2)

    stats.candidates_found = len(candidates)
    t_tot_dur = time.time() - t_start_all

    logger.info(f"--- VISUAL DISCOVERY PROFILING TIMINGS ---")
    logger.info(f"scout_time:               {t_scout_dur:.3f}s")
    logger.info(f"periodic_detection_time:  {t_per_dur:.3f}s")
    logger.info(f"trigger_merge_time:       {t_merge_dur:.3f}s")
    logger.info(f"dense_sampling_time:      {t_dense_dur:.3f}s")
    logger.info(f"text_detection_time:      {t_detect_dur:.3f}s")
    logger.info(f"tracking_time:            {t_track_dur:.3f}s")
    logger.info(f"ocr_time:                 {t_ocr_dur:.3f}s")
    logger.info(f"total_visual_discovery:   {t_tot_dur:.3f}s")

    return candidates, stats


