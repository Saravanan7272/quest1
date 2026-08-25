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

| Case | Video Link | Target Query | Ground Truth | Speech Match | Visual Text Match | Sources | Expected Status | Evidence Timestamp |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Case 1** | [YVvD7SZ7kc0](https://www.youtube.com/watch?v=YVvD7SZ7kc0) | `"Why Do We Fall?"` | Spoken Only | True | False | `["asr"]` | `FOUND` | `87.660s` (ASR Query Span) |
| **Case 2** | [YVvD7SZ7kc0](https://www.youtube.com/watch?v=YVvD7SZ7kc0) | `"so that we can learn to pick ourselves"` | Spoken Only | True | False | `["asr"]` | `FOUND` | `92.740s` (ASR Query Span) |
| **Case 3** | [YVvD7SZ7kc0](https://www.youtube.com/watch?v=YVvD7SZ7kc0) | `"Thank you for watching"` | **Silent Visual Text** | False | True | `["ocr"]` | `FOUND` | `111.500s` (Visual OCR Frame) |
| **Case 4** | [YVvD7SZ7kc0](https://www.youtube.com/watch?v=YVvD7SZ7kc0) | `"have you quite given up on me?"` | Spoken Only | True | False | `["asr"]` | `FOUND` | `18.000s` (ASR Query Span) |
| **Case 5** | [WZORRHNP9_w](https://www.youtube.com/shorts/WZORRHNP9_w) | `"At least tell me your name"` | **Spoken + Visual** | True | True | `["asr", "ocr"]` | `FOUND` | `15.347s` (Visual OCR Frame) |
| **Case 6** | [WZORRHNP9_w](https://www.youtube.com/shorts/WZORRHNP9_w) | `"Batman"` | Neither | False | False | `[]` | `NOT_FOUND` | None |
| **Case 7** | [WZORRHNP9_w](https://www.youtube.com/shorts/WZORRHNP9_w) | `"The blue elephant is dancing"` | Neither | False | False | `[]` | `NOT_FOUND` | None |

---

## 🚀 Running Golden Cases

### Run Complete Suite:
```powershell
.\.venv\Scripts\python.exe scripts/execute_golden_suite.py
```

### Run Individual Golden Cases:

- **Case 1**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 1
  ```
- **Case 2**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 2
  ```
- **Case 3**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 3
  ```
- **Case 4**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 4
  ```
- **Case 5**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 5
  ```
- **Case 6**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 6
  ```
- **Case 7**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/execute_single_case.py 7
  ```

