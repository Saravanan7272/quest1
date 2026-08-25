"""
scripts/inspect_frame_ocr.py
Inspect exact image files and raw OCR engine output on extracted frames.
"""

from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
import cv2

def inspect():
    coarse_dir = Path("temp_diagnostics/coarse")
    frames = sorted(list(coarse_dir.glob("*.jpg")))
    print(f"Total coarse frame files in {coarse_dir}: {len(frames)}")

    engine = RapidOCR()

    for idx, f in enumerate(frames[:10]):
        img = cv2.imread(str(f))
        if img is None:
            print(f"Frame {f.name}: Failed to read image!")
            continue
        h, w, c = img.shape
        res, _ = engine(str(f))
        print(f"Frame {f.name} ({w}x{h}): Raw RapidOCR result = {res}")

if __name__ == "__main__":
    inspect()
