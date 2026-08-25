"""
scripts/create_visual_test_fixture.py
Creates a local test fixture image and synthetic video with visual text overlay
'My mind rebels at stagnation' to validate TextDetector, TextTracker, and OCREngineAdapter.
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_detector import TextDetector
from src.ocr import OCREngineAdapter
from src.text_tracker import TextTracker
from src.matching import compute_ocr_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_visual_fixture")

def create_and_test_fixture():
    test_dir = Path("tests/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    img_path = test_dir / "visual_text_frame.jpg"

    # Read base image or create canvas
    base_img = cv2.imread("temp_diagnostics/coarse/batch_0.0_001.jpg")
    if base_img is None:
        base_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Draw clear visual text overlay: "My mind rebels at stagnation"
    overlay = base_img.copy()
    h, w, _ = overlay.shape
    text = "My mind rebels at stagnation"
    
    # Draw semi-transparent background banner for subtitle readability
    cv2.rectangle(overlay, (w // 6, h - 200), (5 * w // 6, h - 80), (0, 0, 0), -1)
    
    # Draw text in white
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 120
    
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(str(img_path), overlay)
    logger.info(f"Saved visual text test fixture to: {img_path}")

    # 1. Test TextDetector (Detection API)
    detector = TextDetector(min_confidence=0.30)
    det_boxes = detector.detect(img_path, timestamp=2.5)
    logger.info(f"1. TextDetector output: {len(det_boxes)} boxes detected.")
    for b in det_boxes:
        logger.info(f"   Box bbox={b.bbox}, conf={b.detection_confidence:.2f}")

    # 2. Test TextTracker
    tracker = TextTracker(iou_threshold=0.5, max_gap_seconds=0.5)
    tracker.update(2.5, det_boxes)

    # 3. Test OCREngineAdapter (Recognition API)
    ocr_adapter = OCREngineAdapter(lang="en", min_confidence=0.30)
    ocr_boxes = ocr_adapter.run_ocr(img_path)
    logger.info(f"3. OCREngineAdapter output: {len(ocr_boxes)} OCR text recognitions.")
    
    match_found = False
    for box in ocr_boxes:
        score = compute_ocr_score(box.text, "My mind rebels at stagnation")
        logger.info(f"   OCR Text: '{box.text}' (conf={box.confidence:.2f}, similarity={score:.4f})")
        if score >= 0.75:
            match_found = True
            tracker.update_best_frame(1, 2.5, box.text, float(box.confidence) * score)

    tracked_events = tracker.finalize()
    logger.info(f"4. Tracker finalized events: {len(tracked_events)}")
    for evt in tracked_events:
        logger.info(f"   Track {evt.track_id}: start={evt.start}s, end={evt.end}s, best_text='{evt.best_text}'")

    assert len(det_boxes) > 0, "TextDetector failed to detect text box!"
    assert len(ocr_boxes) > 0, "OCREngineAdapter failed to recognize text!"
    assert match_found, "OCR text score failed to reach similarity threshold!"
    assert len(tracked_events) == 1, "TextTracker failed to produce 1 tracked event!"
    logger.info("✅ VISUAL TEXT FIXTURE DIAGNOSTIC PASSED!")

if __name__ == "__main__":
    create_and_test_fixture()
