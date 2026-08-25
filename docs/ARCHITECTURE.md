# System Architecture — Dual-Path Video Dialogue Locator (v1.0.0)

The **Video Dialogue Locator** employs a **Dual-Path Multimodal Retrieval Architecture** designed to locate dialogue occurrences in video content from either spoken audio or burned-in visual text.

---

## 🏗️ High-Level Architectural Flow

```text
                                  TARGET QUERY
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
             AUDIO EVIDENCE (ASR)              VISUAL DISCOVERY PATH
           (faster-whisper word span)         (Scout + Staged Filter)
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

---

## 1. Video Acquisition Stage (`src/acquisition.py`)
- Downloads video stream using `yt-dlp` into `input_video.mp4` with options `"merge_output_format": "mp4"`, `"retries": 10`, `"fragment_retries": 10`.
- Inspects container metadata using OpenCV: `duration` (seconds), `fps` (frames/sec), `resolution` (width x height), and `has_audio` (boolean).

---

## 2. Audio Evidence Path (`src/asr.py`)
- Transcribes audio using `faster-whisper` (`word_timestamps=True`, `beam_size=5`).
- Computes ASR segment score using `compute_asr_score` in `src/matching.py`:
  $$\text{Score} = 0.70 \times \text{PartialPhraseRatio} + 0.30 \times \text{TokenCoverage}$$
- If `best_score >= candidate_threshold` (0.50), calls `find_query_span(words, target_text)` to extract precise word-level timestamps (`asr_query_start`, `asr_query_end`).
- Establishes targeted visual search window:
  $$\text{TargetedWindow} = [\max(0, \text{asr\_query\_start} - 1.5\text{s}), \min(\text{duration}, \text{asr\_query\_end} + 3.5\text{s})]$$

---

## 3. Staged Visual Discovery Path (`src/visual_pipeline.py`)

The visual pipeline uses aggressive staged reduction to avoid executing expensive OCR on every frame:

```text
121.31-second video (~3,630 frames)
        │
        ▼
61 coarse / periodic trigger windows (1 FPS scout + 0.5 FPS periodic)
        │
        ▼
Coarse Text-Bearing Window Filter (TextDetector pre-filter)
        │
        ▼
6 text-bearing windows
        │
        ▼
Dense Sampling at 3.0 FPS (36 actual dense frames)
        │
        ▼
Text Detection & Multi-Box IoU Tracking (iou_threshold=0.5, max_gap=0.5s)
        │
        ▼
11 tracked text events
        │
        ▼
Representative Frame Sampling & Selective OCR Recognition
        │
        ▼
Exact Match: "THANK YOU FOR WATCHING" (query_similarity = 1.0000)
        │
        ▼
Exact Evidence Frame: 111.500s (frame #3345)
```

### Key Stages:
1. **Change Scout & Periodic Triggers**: Computes pixel differences between 1 FPS scout frames and merges with 0.5 FPS periodic safety triggers.
2. **Text-Bearing Pre-filter**: Evaluates coarse scout frames with `TextDetector` to eliminate regions unlikely to contain text.
3. **Dense Sampling**: Samples remaining text-active windows at `sampling.dense_fps` (3.0 FPS in `config.yaml`).
4. **IoU Tracking**: Groups detected bounding boxes across consecutive frames into `TextEvent` tracks using Intersection over Union (`iou_threshold: 0.5`).
5. **Representative Track Sampling**: Samples candidate frames across the full duration of each track, runs OCR, and dynamically sets `best_frame_timestamp` to the frame yielding the highest query similarity.

---

## 4. Multi-Metric Candidate Association (`src/candidate_association.py`)
- Evaluates temporal compatibility:
  $$\text{is\_compat} = (\text{v\_start} \le \text{q\_end} + \text{tolerance}) \land (\text{v\_end} \ge \text{q\_start} - \text{tolerance})$$
  (Default `association_window = 5.0s`).
- Checks OCR acceptance policy (`ocr_min_threshold = 0.45`).
- Produces unified candidates with explicit flags:
  - Spoken + Visual: `speech_match = True`, `visual_text_match = True`, `sources = ["asr", "ocr"]`
  - Visual-Only: `speech_match = False`, `visual_text_match = True`, `sources = ["ocr"]`
  - Spoken-Only: `speech_match = True`, `visual_text_match = False`, `sources = ["asr"]`

---

## 5. Modality-Correct Evidence Frame Selection (`src/pipeline.py`)
- **Multimodal (ASR + OCR)**: Evidence timestamp is selected from the visual OCR frame (`visual_span.best_frame_timestamp`).
- **OCR-Only**: Evidence timestamp is selected from the visual OCR frame (`visual_span.best_frame_timestamp`).
- **ASR-Only**: Evidence timestamp is selected from the ASR query span (`asr_span.query_start`).
- Copies exact evidence image to `outputs/evidence_<timestamp>s.jpg`.
