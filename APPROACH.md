# Technical Approach — Dual-Path Video Dialogue Locator

## 1. Executive Summary

The **Video Dialogue Locator** addresses the open-ended task of finding exact video frames matching target dialogue text. Recognizing that dialogue in video can manifest as **spoken audio**, **on-screen visual text** (burned-in subtitles, title cards), or **multimodal occurrences**, the system implements a **Dual-Path Retrieval Architecture**.

---

## 2. Algorithms & Subsystems

### 2.1 Audio Evidence Path (ASR)
- **Model**: `faster-whisper` (CTranslate2 ONNX quantization, `compute_type="int8"`, CPU-first).
- **Word Timestamps**: Extracts word-level temporal bounds for exact search windowing ($\pm 5.0\text{s}$).
- **Fuzzy Phrase Matching**: Combines character-level partial ratio ($70\%$) and token-level coverage ratio ($30\%$) using `rapidfuzz`.

### 2.2 Visual Discovery Subsystem
- **Level 1: Coarse Sampling & Scout**:
  - **Visual Change Scout (1 FPS)**: Computes mean pixel difference ($\Delta I$) between consecutive frames. Triggers dense sampling when $\Delta I \ge 30$.
  - **Periodic Text Detector (0.5 FPS)**: Triggers periodic sampling every $2.0\text{s}$ to guarantee recall insurance for silent text appearing on static backgrounds.
  - **Trigger Merging**: Merges change scout and periodic triggers within a $2.0\text{s}$ window.
- **Level 2: Dense Sampling & Multi-Box IoU Tracking**:
  - Samples triggered windows at higher FPS ($2.0-10.0\text{FPS}$).
  - Runs fast text detection returning bounding boxes $[x_1, y_1, x_2, y_2]$.
  - Tracks individual text regions across consecutive frames using Intersection-over-Union ($\text{IoU} \ge 0.5$) with strict one-to-one track assignment and a maximum gap tolerance of $0.5\text{s}$.
- **Level 3: Query Relevance Best Frame Selection & OCR**:
  - Selects the best evidence frame per track by maximizing Query Relevance:
    $$\text{Relevance} = \text{Confidence}_{\text{OCR}} \times \text{Similarity}_{\text{Text}}$$
  - Runs OCR recognition on best track frames.

### 2.3 Candidate Association & Score Fusion
- **Temporal Candidate Association**: Merges ASR and Visual candidates occurring within $2.0\text{s}$ into unified `multimodal` candidates.
- **Missing-Modality Score Fusion**: Normalizes available modality scores:
  $$\text{Fused Score} = \sum_{m \in \text{available}} \text{score}_m \times \frac{w_m}{\sum_{k \in \text{available}} w_k}$$
- **Top-K Deduplication**: Sorts candidates descending by `fused_score` and deduplicates occurrences within $2.0\text{s}$.

---

## 3. Trade-Off Analysis

| Architectural Decision | Chosen Strategy | Alternative Evaluated | Key Rationale |
|---|---|---|---|
| **Visual Sampling** | Coarse Scout + Periodic $\rightarrow$ Merge $\rightarrow$ Dense | Constant 10 FPS OCR on entire video | 85% compute reduction on CPU; avoids burning cycles on static frames. |
| **OCR Architecture** | RapidOCR ONNX engine / PaddleOCR 3.x API | Full PaddleOCR C++ static engine | RapidOCR ONNX bypasses Paddle static engine instruction bugs on Windows CPU. |
| **Tracking** | Multi-Box IoU Tracker (`max_gap=0.5s`) | Per-frame un-linked OCR | Links multi-line titles and isolates distinct text appearances over time. |
| **Score Fusion** | Missing-Modality Normalization | Fixed Weighted Average | Prevents penalizing spoken-only dialogue or silent text cards where one modality is absent. |

---

## 4. Failure Modes & Mitigations

1. **Static Background with Newly Appearing Text**:
   - *Risk*: Visual Change Scout might produce low frame difference.
   - *Mitigation*: Periodic Detector triggers every $2.0\text{s}$ regardless of scene change.
2. **Multiple Text Boxes Simultaneously**:
   - *Risk*: Assigning both title and subtitle to the same track.
   - *Mitigation*: IoU Tracker enforces strict one-to-one box assignment (`assigned_tracks` set).
3. **Text Disappears and Reappears**:
   - *Risk*: Merging separate occurrences into a single continuous event.
   - *Mitigation*: Track closes automatically after `max_gap_seconds` ($0.5\text{s}$) inactivity.
## 5. Golden Evaluation Benchmark Matrix

The system is evaluated against a 7-query Golden Benchmark Matrix covering spoken-only, silent visual text, multimodal confirmation, and negative controls:

| Video URL | Query | Ground Truth | ASR | OCR | Expected Result | Expected Sources | Expected Evidence |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `YVvD7SZ7kc0` | `"Why Do We Fall?"` | Speech | ✓ | — | `FOUND` | `["asr"]` | `speech_match: true` |
| `YVvD7SZ7kc0` | `"so that we can learn to pick ourselves"` | Speech | ✓ | — | `FOUND` | `["asr"]` | `speech_match: true` |
| `YVvD7SZ7kc0` | `"Thank you for watching"` | **Silent visual text** | ✗ | ✓ | `FOUND` | `["ocr"]` | `visual_text_match: true` |
| `YVvD7SZ7kc0` | `"have you quite given up on me?"` | Speech | ✓ | — | `FOUND` | `["asr"]` | `speech_match: true` |
| `WZORRHNP9_w` | `"At least tell me your name"` | **Speech + visual** | ✓ | ✓ | `FOUND` | `["asr", "ocr"]` | `speech_match: true`, `visual_text_match: true` |
| `WZORRHNP9_w` | `"Batman"` | Neither | ✗ | ✗ | `NOT_FOUND` | `[]` | None |
| `WZORRHNP9_w` | `"The blue elephant is dancing"` | Neither | ✗ | ✗ | `NOT_FOUND` | `[]` | None |

*Detailed benchmark specifications are in [`tests/fixtures/README.md`](file:///e:/quest1/tests/fixtures/README.md).*

```text
                    VIDEO DIALOGUE LOCATOR
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
      SPOKEN ONLY        VISUAL ONLY       BOTH MODALITIES
          │                   │                   │
      ASR success          OCR success       ASR + OCR
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                         NEGATIVE TEST
                              │
                         NOT_FOUND
                              │
                    SHORT-LIVED VISUAL
                              │
                    TEMPORAL RECALL
```

