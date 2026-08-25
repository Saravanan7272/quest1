"""
src/text_detector.py
Fast text detection wrapper returning text bounding box regions.
"""

import logging
from pathlib import Path
from typing import List

from src.models import TrackedBox

logger = logging.getLogger(__name__)

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
                res, _ = self._engine(str(image_path))
                if res:
                    for item in res:
                        bbox_raw, text, conf = item
                        conf_val = float(conf)
                        if conf_val >= self.min_confidence:
                            xs = [float(pt[0]) for pt in bbox_raw]
                            ys = [float(pt[1]) for pt in bbox_raw]
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
