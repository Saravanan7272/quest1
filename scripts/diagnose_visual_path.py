"""
scripts/diagnose_visual_path.py
Step-by-step visual path diagnostic tool inspecting frame extraction, text detection,
OCR recognition, and text similarity on https://youtu.be/dPTKl5H5ftg.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.acquisition import download_video
from src.sampling import generate_sample_timestamps, extract_frames_at_timestamps
from src.text_detector import TextDetector
from src.ocr import OCREngineAdapter
from src.matching import compute_ocr_score
from src.visual_scout import detect_visual_changes, generate_periodic_triggers, merge_triggers
from src.text_tracker import TextTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("diagnose_visual")

def run_diagnostics():
    url = "https://youtu.be/dPTKl5H5ftg"
    target_text = "My mind rebels at stagnation"
    temp_dir = Path("temp_diagnostics")
    temp_dir.mkdir(exist_ok=True)

    logger.info("=== 1. ACQUISITION ===")
    meta = download_video(url, temp_dir / "acq")
    logger.info(f"Video Metadata: Duration={meta.duration}s, FPS={meta.fps}, Resolution={meta.width}x{meta.height}")

    logger.info("=== 2. COARSE SAMPLING (1 FPS) ===")
    coarse_ts = generate_sample_timestamps(0.0, meta.duration, coarse_fps=1.0)
    coarse_frames = extract_frames_at_timestamps(meta.video_path, coarse_ts, temp_dir / "coarse")
    logger.info(f"Extracted {len(coarse_frames)} coarse frames.")

    logger.info("=== 3. VISUAL CHANGE SCOUT & PERIODIC DETECTOR ===")
    paths = [f.path for f in coarse_frames]
    ts_list = [f.timestamp for f in coarse_frames]
    
    scout_trigs = detect_visual_changes(paths, ts_list, threshold=30.0)
    periodic_trigs = generate_periodic_triggers(meta.duration, periodic_fps=0.5)
    merged_trigs = merge_triggers(scout_trigs, periodic_trigs, merge_window=2.0)
    
    logger.info(f"Scout Triggers ({len(scout_trigs)}): {scout_trigs}")
    logger.info(f"Periodic Triggers ({len(periodic_trigs)}): {periodic_trigs}")
    logger.info(f"Merged Triggers ({len(merged_trigs)}): {merged_trigs}")

    logger.info("=== 4. TEXT DETECTION & RECOGNITION ACROSS COARSE FRAMES ===")
    detector = TextDetector(min_confidence=0.30)
    ocr_engine = OCREngineAdapter(lang="en", min_confidence=0.30)
    tracker = TextTracker(iou_threshold=0.5, max_gap_seconds=0.5)

    matches_found = []

    for f_rec in coarse_frames:
        det_boxes = detector.detect(f_rec.path, f_rec.timestamp)
        tracker.update(f_rec.timestamp, det_boxes)
        
        ocr_boxes = ocr_engine.run_ocr(f_rec.path)
        if ocr_boxes:
            logger.info(f"ts={f_rec.timestamp:.2f}s: Found {len(ocr_boxes)} OCR text boxes.")
            for box in ocr_boxes:
                score = compute_ocr_score(box.text, target_text)
                logger.info(f"   - Box Text: '{box.text}' (conf={box.confidence:.2f}, sim_score={score:.4f})")
                if score >= 0.60:
                    matches_found.append((f_rec.timestamp, box.text, score, box.bounding_rect))

    tracked_events = tracker.finalize()
    logger.info(f"=== 5. TRACKER SUMMARY: {len(tracked_events)} tracked events. ===")
    for evt in tracked_events:
        logger.info(f"Track {evt.track_id}: start={evt.start:.2f}s, end={evt.end:.2f}s, boxes={len(evt.boxes)}")

    logger.info(f"=== 6. TARGET MATCHES FOUND: {len(matches_found)} ===")
    for m in matches_found:
        logger.info(f"   MATCH at ts={m[0]:.2f}s: '{m[1]}' (score={m[2]:.4f})")

if __name__ == "__main__":
    run_diagnostics()
