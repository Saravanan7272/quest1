"""
scripts/check_environment.py
Environment diagnostic script for Phase 0 Hardware + Environment Gate.
"""

import sys
import os
import shutil
import platform
import subprocess
from pathlib import Path

def get_disk_space_gb(path="."):
    total, used, free = shutil.disk_usage(path)
    return total / (1024**3), free / (1024**3)

def run_command(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, res.stdout.strip()
    except Exception as e:
        return False, str(e)

def main():
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("==========================================")
    log("ENVIRONMENT VALIDATION REPORT")
    log("==========================================")
    
    # 1. System Info
    log(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    log(f"Architecture: {platform.machine()} / {platform.architecture()[0]}")
    log(f"Python Executable: {sys.executable}")
    log(f"Python Version: {sys.version.split()[0]}")
    
    in_venv = sys.prefix != sys.base_prefix
    log(f"Inside Virtual Environment: {in_venv}")
    if not in_venv:
        log("WARN: Not running inside a virtual environment (.venv)!")
    
    total_disk, free_disk = get_disk_space_gb(".")
    log(f"Disk Space: Total {total_disk:.2f} GB, Free {free_disk:.2f} GB")
    
    # 2. Check System Binaries (ffmpeg, ffprobe)
    log("\n--- Checking System Binaries ---")
    ffmpeg_ok, ffmpeg_out = run_command(["ffmpeg", "-version"])
    ffprobe_ok, ffprobe_out = run_command(["ffprobe", "-version"])
    
    if ffmpeg_ok:
        log(f"FFmpeg: PASS ({ffmpeg_out.splitlines()[0]})")
    else:
        log(f"FFmpeg: FAIL ({ffmpeg_out})")
        
    if ffprobe_ok:
        log(f"FFprobe: PASS ({ffprobe_out.splitlines()[0]})")
    else:
        log(f"FFprobe: FAIL ({ffprobe_out})")

    # 3. Check Python Packages
    log("\n--- Checking Python Package Imports ---")
    packages = ["numpy", "rapidfuzz", "yaml", "yt_dlp", "faster_whisper", "rapidocr_onnxruntime", "cv2"]
    pkg_status = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            log(f"Package '{pkg}': PASS (v{ver})")
            pkg_status[pkg] = True
        except ImportError as e:
            log(f"Package '{pkg}': FAIL ({e})")
            pkg_status[pkg] = False

    # 4. Smoke Tests
    log("\n--- Running Functional Smoke Tests ---")
    
    # NumPy & RapidFuzz
    try:
        import numpy as np
        from rapidfuzz import fuzz
        arr = np.array([1, 2, 3])
        sim = fuzz.ratio("hello world", "hello world!")
        log(f"NumPy & RapidFuzz Basic Test: PASS (sim={sim:.1f})")
    except Exception as e:
        log(f"NumPy & RapidFuzz Basic Test: FAIL ({e})")
        
    # faster-whisper Smoke Test
    whisper_ok = False
    if pkg_status.get("faster_whisper"):
        try:
            log("Initializing faster-whisper ('tiny', device='cpu', compute_type='int8')...")
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            log("faster-whisper initialization: PASS")
            whisper_ok = True
        except Exception as e:
            log(f"faster-whisper initialization: FAIL ({e})")
    else:
        log("faster-whisper initialization: SKIPPED (package missing)")

    # RapidOCR / PaddleOCR Smoke Test
    ocr_ok = False
    if pkg_status.get("rapidocr_onnxruntime") and pkg_status.get("cv2"):
        try:
            log("Initializing RapidOCR (PP-OCRv4 ONNX CPU engine)...")
            import cv2
            import numpy as np
            from rapidocr_onnxruntime import RapidOCR
            
            dummy_img_path = Path(".tmp/smoke_test_ocr.png")
            dummy_img_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = np.ones((100, 400, 3), dtype=np.uint8) * 255
            cv2.putText(img, "TEST OCR", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            cv2.imwrite(str(dummy_img_path), img)
            
            engine = RapidOCR()
            res, elapse = engine(str(dummy_img_path))
            log(f"RapidOCR Inference: PASS (Boxes detected: {len(res) if res else 0}, elapse: {elapse})")
            if res:
                log(f"Extracted Sample Box 0: text='{res[0][1]}', score={float(res[0][2]):.4f}")
            ocr_ok = True
        except Exception as e:
            log(f"RapidOCR Test: FAIL ({e})")
    else:
        log("RapidOCR Test: SKIPPED (package missing)")

    # Overall Status
    log("\n==========================================")
    all_passed = (
        in_venv and ffmpeg_ok and ffprobe_ok and
        all(pkg_status.values()) and whisper_ok and ocr_ok
    )
    status_str = "PASS" if all_passed else "BLOCKED / FAIL"
    log(f"OVERALL ENVIRONMENT GATE STATUS: {status_str}")
    log("==========================================")
    
    # Save Report
    with open("environment_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\nSaved report to environment_report.txt")

if __name__ == "__main__":
    main()
