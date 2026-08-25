# Dual-Path Video Dialogue Locator (v1.0.0 Baseline)

A production-oriented, multi-modal video search system that locates dialogue occurrences in video content from either **spoken audio** or **burned-in visual text** (on-screen subtitles, title cards, dynamic captions).

---

## 🌟 Architectural Overview

The system uses a **Dual-Path Retrieval Architecture** with ASR-guided targeted visual discovery:

```text
                                  TARGET QUERY
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
             AUDIO EVIDENCE (ASR)              VISUAL DISCOVERY PATH
           (faster-whisper word span)            (Scout + Dense Detector)
                      │                                 │
                      ▼                                 ▼
              ASR Query Span                      Visual Tracks
           [asr_query_start, end]             [track_id, ocr_text, sim]
                      │                                 │
                      └────────────────┬────────────────┘
                                       ▼
                         MULTI-METRIC CANDIDATE ASSOCIATION
                          (Temporal Window + OCR Score)
                                       │
                                       ▼
                        MISSING-MODALITY SCORE FUSION
                      (Normalized ASR + OCR + Semantic)
                                       │
                                       ▼
                           EXACT EVIDENCE SELECTION
                     (Visual OCR Frame for Multimodal/OCR)
```

### Key Subsystems
1. **Audio Evidence Path**: Uses `faster-whisper` (word-level timestamps) to extract exact query word spans (`asr_query_start`, `asr_query_end`).
2. **Targeted Visual Discovery Path**:
   - **ASR-Guided Search Window**: When ASR candidate is present, restricts visual search to `[asr_query_start - pre_roll, asr_query_end + post_roll]`.
   - **Global Visual Scout Fallback**: When ASR candidate is absent/weak (e.g. Case 3 silent visual text), falls back to global scout (1 FPS) + text-bearing trigger filtering.
   - **Text-Bearing Trigger Filter**: Filters out non-text scenes before running dense text detection.
   - **Dense Text Detection & IoU Tracking**: Detects text boxes at 3.0 FPS and tracks bounding boxes across consecutive frames using IoU tracking.
   - **Representative Track Sampling**: Samples representative frames across each track's duration and evaluates OCR text similarity to select the highest-scoring frame (`best_frame_timestamp`).
3. **Multi-Metric Candidate Association & Fusion**:
   - Compares visual track bounds against ASR estimated query spans using temporal proximity tolerance (`5.0s`) AND multi-factor OCR similarity.
   - Preserves explicit `speech_match` (bool) and `visual_text_match` (bool) flags, with `sources = ["asr"]`, `["ocr"]`, or `["asr", "ocr"]`.
4. **Modality-Correct Evidence Frame Selection**:
   - **Multimodal (ASR + OCR)**: Selects the visual OCR frame (e.g. 15.18s text card).
   - **OCR-only**: Selects the visual OCR frame (e.g. 111.50s title card).
   - **ASR-only**: Selects representative frame from ASR query span.

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
- **Operating System**: Windows / Linux / macOS
- **Python**: 3.10+
- **System Dependencies**: FFmpeg installed and available on PATH.

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Dialogue Locator CLI
```bash
python run.py --url https://www.youtube.com/shorts/WZORRHNP9_w --target "At least tell me your name" --config config.yaml
```

---

## 🧪 Comprehensive Unit Testing

Run the automated test suite covering models, word-level ASR span extraction, sampling, visual scout, text detector, IoU tracker, scoring, multi-metric candidate association, signature consistency, and evidence selection:

```bash
pytest tests/
```

*36 passing unit tests verified.*

---

## 🏆 Golden Evaluation Test Suite

The system is evaluated against the Golden Benchmark Matrix:

| Case | Video URL | Query | Ground Truth | Speech Match | Visual Text Match | Sources | Expected Result |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **Case 1** | `YVvD7SZ7kc0` | `"Why Do We Fall?"` | Spoken Only | True | False | `["asr"]` | `FOUND` |
| **Case 3** | `YVvD7SZ7kc0` | `"Thank you for watching"` | Silent Visual Text | False | True | `["ocr"]` | `FOUND` |
| **Case 5** | `WZORRHNP9_w` | `"At least tell me your name"` | Spoken + Visual | True | True | `["asr", "ocr"]` | `FOUND` |
| **Case 6** | `WZORRHNP9_w` | `"Batman"` | Neither | False | False | `[]` | `NOT_FOUND` |

---

## 🤖 AI Disclosure & Transparency

In compliance with technical assignment guidelines, all prompts, iterations, and model feedback used during development are fully documented in [`prompts.txt`](file:///e:/quest1/prompts.txt).## ⚠️ Architectural Scope & Limitations

> [!NOTE]
> This architecture covers the major retrieval scenarios while retaining explicit sampling limitations for extremely short-lived visual text (< 0.5s duration).
