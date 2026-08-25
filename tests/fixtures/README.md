# Golden Evaluation Test Suite

============================================================
GOLDEN EVALUATION TEST SUITE
============================================================

These videos and queries are mandatory regression-test fixtures for the
Video Dialogue Locator.

The purpose is NOT merely to prove that OCR and ASR work.

The purpose is to prove that the system correctly determines
WHICH MODALITY contains the requested text and selects the
appropriate retrieval path.

The test suite covers:
1. Spoken-only text
2. Visual-only / silent on-screen text
3. Spoken + visual text simultaneously
4. Neither modality / negative retrieval
5. Multiple queries within the same video
6. Temporal localization
7. Correct modality evidence
8. Missing-modality score handling

---

## Golden Benchmark Evaluation Matrix

| Video URL | Query | Ground Truth | ASR | OCR | Expected Result | Expected Sources | Expected Evidence |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `YVvD7SZ7kc0` | `"Why Do We Fall?"` | Speech | ✓ | — | `FOUND` | `["asr"]` | `speech_match: true` |
| `YVvD7SZ7kc0` | `"so that we can learn to pick ourselves"` | Speech | ✓ | — | `FOUND` | `["asr"]` | `speech_match: true` |
| `YVvD7SZ7kc0` | `"Thank you for watching"` | **Silent visual text** | ✗ | ✓ | `FOUND` | `["ocr"]` | `visual_text_match: true` |
| `YVvD7SZ7kc0` | `"have you quite given up on me?"` | Speech | ✓ | — | `FOUND` | `["asr"]` | `speech_match: true` |
| `WZORRHNP9_w` | `"At least tell me your name"` | **Speech + visual** | ✓ | ✓ | `FOUND` | `["asr", "ocr"]` | `speech_match: true`, `visual_text_match: true` |
| `WZORRHNP9_w` | `"Batman"` | Neither | ✗ | ✗ | `NOT_FOUND` | `[]` | None |
| `WZORRHNP9_w` | `"The blue elephant is dancing"` | Neither | ✗ | ✗ | `NOT_FOUND` | `[]` | None |

*Note: `—` indicates the modality is not required by ground truth.*

---

## Detailed Test Case Specifications

### TEST VIDEO 1 — DUAL-PATH GOLDEN VIDEO
- **URL**: `https://www.youtube.com/watch?v=YVvD7SZ7kc0`
- **Critical Case**: `"Thank you for watching"` (Silent visual text).
  - *Expected Behavior*: ASR candidate is absent or irrelevant. Visual discovery path runs independently via Visual Change Scout (1 FPS) + Periodic Detector (0.5 FPS), merges triggers, detects on-screen text, tracks regions via IoU tracker, recognizes text, and produces `sources: ["ocr"]` with `visual_text_match: true`.

---

### TEST VIDEO 2 — MULTIMODAL + NEGATIVE GOLDEN VIDEO
- **URL**: `https://www.youtube.com/shorts/WZORRHNP9_w`
- **CASE A — SPOKEN + VISUAL**: `"At least tell me your name"`
  - *Expected Behavior*: Both ASR and Visual Discovery paths find candidates. Candidate Association merges them into a multimodal candidate with `sources: ["asr", "ocr"]` and `speech_match: true, visual_text_match: true`. Fused score utilizes both available modalities.
- **CASE B & C — NEGATIVE CONTROL**: `"Batman"`, `"The blue elephant is dancing"`
  - *Expected Behavior*: Neither speech nor on-screen text reaches similarity thresholds. Returns `status: "NOT_FOUND"` without hallucinating matches.

---

## Temporal Recall & Scope Limitation

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

> [!NOTE]
> **Temporal Recall Limitation**: Text appearing for extremely short durations (< 0.5s) may be missed if neither the 1 FPS Visual Change Scout nor the 0.5 FPS Periodic Detector extracts a frame during its appearance window. This represents a temporal sampling limitation rather than an OCR recognition failure.
