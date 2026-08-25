# Testing & Evaluation Guide — Video Dialogue Locator (v1.0.0)

This document provides complete instructions for executing the automated unit test suite and the golden evaluation benchmark.

---

## 🧪 Automated Unit Test Suite

The test suite contains **36 unit tests** across 11 test modules:

```powershell
.\.venv\Scripts\pytest.exe tests/
```

### Module Breakdown:

| Test Module | Tests | Functionality Covered |
|---|:---:|---|
| `tests/test_asr_word_span.py` | 2 | Word-level timestamp query span extraction (`find_query_span`). |
| `tests/test_candidate_association.py` | 3 | Multi-metric candidate association, evidence timestamp selection, signature consistency. |
| `tests/test_matching.py` | 8 | Text normalization, ASR phrase/token scoring, OCR box similarity. |
| `tests/test_models.py` | 4 | Dataclass model initialization and schema validation. |
| `tests/test_ocr_adapter.py` | 3 | RapidOCR adapter integration, bounding box parsing, confidence filtering. |
| `tests/test_pipeline.py` | 4 | Orchestrator execution flow, timestamp formatting, deduplication. |
| `tests/test_sampling.py` | 2 | Timestamp list generation, frame extraction pipeline. |
| `tests/test_scoring.py` | 3 | Score fusion, missing modality handling, match level categorization. |
| `tests/test_text_detector.py` | 1 | Text detector model initialization and bounding box output. |
| `tests/test_tracker.py` | 4 | IoU calculation, track creation, track age-out, best frame assignment. |
| `tests/test_visual_scout.py` | 2 | Visual change scout frame differences, trigger merging. |

---

## 🏆 Golden Benchmark Evaluation Matrix

The system is benchmarked against a 7-query Golden Benchmark Matrix defined in `tests/fixtures/golden_tests.yaml`:

| Case | Video URL | Target Query | Ground Truth | Speech Match | Visual Text Match | Sources | Expected Status | Evidence Timestamp |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Case 1** | `YVvD7SZ7kc0` | `"Why Do We Fall?"` | Spoken Only | True | False | `["asr"]` | `FOUND` | `1.50s` (ASR Query Span) |
| **Case 2** | `YVvD7SZ7kc0` | `"so that we can learn to pick ourselves"` | Spoken Only | True | False | `["asr"]` | `FOUND` | `9.90s` (ASR Query Span) |
| **Case 3** | `YVvD7SZ7kc0` | `"Thank you for watching"` | **Silent Visual Text** | False | True | `["ocr"]` | `FOUND` | `111.50s` (Visual OCR Frame) |
| **Case 4** | `YVvD7SZ7kc0` | `"have you quite given up on me?"` | Spoken Only | True | False | `["asr"]` | `FOUND` | `18.00s` (ASR Query Span) |
| **Case 5** | `WZORRHNP9_w` | `"At least tell me your name"` | **Spoken + Visual** | True | True | `["asr", "ocr"]` | `FOUND` | `15.18s` / `15.35s` (Visual OCR Frame) |
| **Case 6** | `WZORRHNP9_w` | `"Batman"` | Neither | False | False | `[]` | `NOT_FOUND` | None |
| **Case 7** | `WZORRHNP9_w` | `"The blue elephant is dancing"` | Neither | False | False | `[]` | `NOT_FOUND` | None |

---

## 🚀 Running Golden Cases

To run an individual golden test case:
```powershell
.\.venv\Scripts\python.exe scripts/execute_single_case.py 5
```

To run the complete golden evaluation suite:
```powershell
.\.venv\Scripts\python.exe scripts/execute_golden_suite.py
```
