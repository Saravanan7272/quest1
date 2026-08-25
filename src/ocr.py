"""
src/ocr.py
OCR Engine Adapter providing normalized OCRBox dataclass representation and spatial box sorting.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: List[List[float]]  # 4 corners: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    @property
    def bounding_rect(self) -> Tuple[float, float, float, float]:
        """Returns (x_min, y_min, x_max, y_max)."""
        xs = [pt[0] for pt in self.bbox]
        ys = [pt[1] for pt in self.bbox]
        return min(xs), min(ys), max(xs), max(ys)

class OCREngineAdapter:
    """
    Adapter pattern wrapping OCR engines (RapidOCR / PaddleOCR) to expose
    a unified OCRBox interface.
    """
    def __init__(self, lang: str = "en", min_confidence: float = 0.30):
        self.lang = lang
        self.min_confidence = min_confidence
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        """
        Initialize RapidOCR (ONNX backend) with PaddleOCR fallback.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("Initializing RapidOCR engine (ONNX CPU backend)...")
            self._engine = RapidOCR()
            self._engine_type = "rapidocr"
            logger.info("RapidOCR engine initialized successfully.")
        except Exception as e1:
            logger.info(f"RapidOCR unavailable or failed ({e1}). Falling back to PaddleOCR...")
            try:
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu"
                )
                self._engine_type = "paddleocr"
                logger.info("PaddleOCR engine initialized successfully.")
            except Exception as e2:
                logger.error(f"Failed to initialize any OCR engine: {e2}")
                raise RuntimeError(f"Could not load OCR engine: RapidOCR ({e1}), PaddleOCR ({e2})")

    def run_ocr(self, image_path: Path) -> List[OCRBox]:
        """
        Extract OCR text boxes from image, filter by confidence, and sort spatially.
        """
        if not image_path.exists():
            logger.warning(f"Image path does not exist for OCR: {image_path}")
            return []

        raw_boxes: List[OCRBox] = []

        try:
            if self._engine_type == "rapidocr":
                res, elapse = self._engine(str(image_path))
                if res:
                    for item in res:
                        bbox_raw, text, conf = item
                        conf_val = float(conf)
                        if conf_val >= self.min_confidence and text.strip():
                            polygon = [[float(pt[0]), float(pt[1])] for pt in bbox_raw]
                            raw_boxes.append(OCRBox(text=text.strip(), confidence=conf_val, bbox=polygon))
            else:
                # PaddleOCR adapter path
                if hasattr(self._engine, "predict"):
                    res = self._engine.predict(str(image_path))
                else:
                    res = self._engine.ocr(str(image_path))
                    
                if res:
                    for item in res:
                        rec_texts = getattr(item, "rec_texts", [])
                        rec_scores = getattr(item, "rec_scores", [])
                        rec_boxes = getattr(item, "rec_boxes", [])
                        
                        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                            conf_val = float(score)
                            if conf_val >= self.min_confidence and text.strip():
                                polygon = [[float(pt[0]), float(pt[1])] for pt in box]
                                raw_boxes.append(OCRBox(text=text.strip(), confidence=conf_val, bbox=polygon))
        except Exception as e:
            logger.warning(f"OCR inference failed on image {image_path}: {e}")
            return []

        # Sort OCR boxes in reading order: top-to-bottom, left-to-right
        sorted_boxes = sort_ocr_boxes_spatial(raw_boxes)
        return sorted_boxes

def sort_ocr_boxes_spatial(boxes: List[OCRBox]) -> List[OCRBox]:
    """
    Sort OCR boxes in natural reading order (top-to-bottom, left-to-right).
    Uses primary key y_min (binned in 10-pixel line bands) and secondary key x_min.
    """
    if not boxes:
        return []

    def get_sort_key(box: OCRBox) -> Tuple[int, float]:
        x_min, y_min, _, _ = box.bounding_rect
        line_band = int(y_min / 15.0)  # 15px line grouping band
        return (line_band, x_min)

    return sorted(boxes, key=get_sort_key)
