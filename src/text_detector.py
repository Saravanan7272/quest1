"""
src/text_detector.py
Fast text detection wrapper returning text bounding box regions.
"""

import logging
from pathlib import Path
from typing import List

from src.models import TrackedBox

logger = logging.getLogger(__name__)

import cv2

def _load_scaled_image(image_path: Path, max_dim: int = 1024):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > max_dim:
        scale = float(max_dim) / float(max_side)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return img, 1.0

class TextDetector:
    """
    Lightweight text detector isolating text bounding boxes before recognition.
    """
    def __init__(self, min_confidence: float = 0.30):
        self.min_confidence = min_confidence
        self._engine = None
        self._init_detector()

    def _init_detector(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._engine_type = "rapidocr"
        except Exception as e:
            logger.warning(f"RapidOCR detector fallback: {e}")
            self._engine_type = "dummy"

    def detect(self, image_path: Path, timestamp: float) -> List[TrackedBox]:
        """
        Detect text region bounding boxes in an image.
        Returns list of TrackedBox objects [x1, y1, x2, y2].
        """
        if not image_path.exists():
            return []

        boxes: List[TrackedBox] = []

        try:
            if self._engine_type == "rapidocr":
                img_obj, scale = _load_scaled_image(image_path, max_dim=1024)
                if img_obj is None:
                    return []
                res, _ = self._engine(img_obj)
                if res:
                    for item in res:
                        bbox_raw, text, conf = item
                        conf_val = float(conf)
                        if conf_val >= self.min_confidence:
                            xs = [float(pt[0]) / scale for pt in bbox_raw]
                            ys = [float(pt[1]) / scale for pt in bbox_raw]
                            rect = [min(xs), min(ys), max(xs), max(ys)]
                            boxes.append(
                                TrackedBox(
                                    timestamp=timestamp,
                                    bbox=rect,
                                    detection_confidence=conf_val
                                )
                            )
        except Exception as e:
            logger.warning(f"Text detection failed on {image_path}: {e}")

        return boxes
