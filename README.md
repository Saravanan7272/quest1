# Dual-Path Video Dialogue Locator

A production-oriented, multi-modal video search system that locates dialogue occurrences in video content from either **spoken audio** or **burned-in visual text** (on-screen subtitles, title cards, dynamic captions).

---

## 🌟 Architectural Overview

The system uses a **Dual-Path Retrieval Architecture** to ensure robust discovery across diverse video content:

```text
                                  TARGET QUERY
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
             AUDIO EVIDENCE                     VISUAL EVIDENCE
          (faster-whisper ASR)           (Scout 1FPS + Periodic 0.5FPS)
                      │                                 │
                      └────────────────┬────────────────┘
                                       ▼
                         TEMPORAL CANDIDATE ASSOCIATION
                                  (Window: 2.0s)
                                       │
                                       ▼
                       MISSING-MODALITY SCORE FUSION
                     (Normalized ASR + OCR + Semantic)
                                       │
                                       ▼
                          TOP-K DEDUPLICATED CONTRACT
```

### Key Subsystems
1. **Audio Evidence Path**: Uses `faster-whisper` (word-level timestamps) to perform phrase-ratio (70%) and token-coverage (30%) fuzzy matching across transcribed speech segments.
2. **Visual Discovery Path**:
   - **Visual Change Scout (1 FPS)**: Computes frame differences to detect dynamic scene cuts and title appearances.
   - **Periodic Text Detector (0.5 FPS)**: Guarantees recall insurance for silent text appearing on static backgrounds.
   - **Trigger Merging (2s Window)**: Combines change scout and periodic triggers into continuous sampling windows.
   - **Dense Sampling & Text Detection**: Samples triggered clips at higher FPS and extracts text bounding boxes.
   - **Multi-Box IoU Tracking**: Tracks individual text regions across consecutive frames using bounding-box IoU, one-to-one assignment, and maximum gap tolerance (`0.5s`).
   - **Query Relevance Frame Selection**: Identifies best evidence frame per track based on `ocr_confidence * text_similarity`.
3. **Missing-Modality Score Fusion**: Fuses scores across available modalities using normalized weights ($\sum w_m = 1.0$), ensuring spoken dialogue without visual text (and vice-versa) is correctly identified.

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
- **Operating System**: Windows / Linux / macOS
- **Python**: 3.10+
- **System Dependencies**: FFmpeg and FFprobe installed and available on PATH.

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Environment Gate
Run the Phase 0 environment check diagnostic script:
```bash
python scripts/check_environment.py
```

### 3. Run Dialogue Locator CLI
```bash
python run.py --url https://youtu.be/dPTKl5H5ftg --target "My mind rebels at stagnation" --config config.yaml
```

---

## 📊 Sample Output Contract

```json
{
  "status": "FOUND",
  "results": [
    {
      "timestamp": "00:00:00.000",
      "timestamp_seconds": 0.0,
      "frame_number": 0,
      "extracted_text": "My mind rebels at stagnation",
      "match_strength": 0.805,
      "match_level": "MEDIUM",
      "image_path": "outputs/evidence_0.000s.jpg",
      "sources": [
        "asr"
      ],
      "evidence": {
        "speech_match": true,
        "visual_text_match": false
      },
      "scores": {
        "asr": 0.805,
        "ocr": null,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1,
  "search_summary": {
    "scout_frames": 37,
    "detector_frames": 19,
    "ocr_calls": 18,
    "candidates_found": 1,
    "tracked_events": 0,
    "runtime_seconds": 98.33
  }
}
```

---

## 🧪 Comprehensive Unit Testing

Run the automated test suite covering models, matching, sampling, visual scout, text detector, IoU tracker, scoring, candidate association, and end-to-end pipeline:

```bash
pytest tests/
```

*32 passing unit tests verified.*

---

## 🤖 AI Disclosure & Transparency

In compliance with technical assignment guidelines, all prompts, iterations, and model feedback used during development are fully documented in [`prompts.txt`](file:///e:/quest1/prompts.txt).

---

## ⚠️ Architectural Scope & Limitations

> [!NOTE]
> This architecture covers the major retrieval scenarios while retaining explicit sampling limitations for extremely short-lived visual text (< 0.5s duration).
