# Golden Evaluation Report

## Test Environment

- **Execution Timestamp**: 2026-08-25 09:37:21 UTC
- **OS**: Windows 11 (10.0.26200)
- **Python Version**: 3.13.1
- **Git Commit SHA**: `c198c3c`
- **Configuration File**: `config.dev.yaml`
- **PaddleOCR Version**: 3.7.0
- **faster-whisper Version**: 1.2.1
- **OpenCV Version**: 4.10.0

## Test Matrix

| Case ID | Query | Ground Truth | Expected | Actual | Result |
|---|---|---|---|---|:---:|
| `case_01` | `Why Do We Fall?` | spoken_only | FOUND (asr) | NOT_FOUND (none) | ❌ FAIL |
| `case_02` | `so that we can learn to pick ourselves` | spoken_only | FOUND (asr) | FOUND (asr) | ✅ PASS |
| `case_03` | `Thank you for watching` | visual_only | FOUND (ocr) | NOT_FOUND (none) | ❌ FAIL |
| `case_04` | `have you quite given up on me?` | spoken_only | FOUND (asr) | NOT_FOUND (none) | ❌ FAIL |
| `case_05` | `At least tell me your name` | spoken_and_visual | FOUND (asr, ocr) | FOUND (asr) | ❌ FAIL |
| `case_06` | `Batman` | neither | NOT_FOUND (none) | ERROR (none) | ❌ FAIL |
| `case_07` | `The blue elephant is dancing` | neither | NOT_FOUND (none) | NOT_FOUND (none) | ✅ PASS |

## Case-by-Case Results

### CASE_01 — Why Do We Fall?
- **Video URL**: `https://www.youtube.com/watch?v=YVvD7SZ7kc0`
- **Query**: `Why Do We Fall?`
- **Ground Truth**: `spoken_only`
- **Expected Status**: `FOUND`
- **Actual Status**: `NOT_FOUND`

**Actual JSON Output**:
```json
{
  "status": "NOT_FOUND",
  "results": [],
  "total_candidates": 0,
  "search_summary": {
    "scout_frames": 122,
    "detector_frames": 25,
    "ocr_calls": 39,
    "candidates_found": 0,
    "tracked_events": 22,
    "runtime_seconds": 184.06
  }
}
```

### CASE_02 — so that we can learn to pick ourselves
- **Video URL**: `https://www.youtube.com/watch?v=YVvD7SZ7kc0`
- **Query**: `so that we can learn to pick ourselves`
- **Ground Truth**: `spoken_only`
- **Expected Status**: `FOUND`
- **Actual Status**: `FOUND`
- **Evidence Image**: `[tests/evaluation_results/evidence/case_02/so_that_we_can_learn_to_pick_o_001.jpg](tests/evaluation_results/evidence/case_02/so_that_we_can_learn_to_pick_o_001.jpg)`

**Actual JSON Output**:
```json
{
  "status": "FOUND",
  "results": [
    {
      "timestamp": "00:01:32.740",
      "timestamp_seconds": 92.74,
      "frame_number": 2782,
      "extracted_text": "so that we can learn to pick ourselves",
      "match_strength": 1.0,
      "match_level": "HIGH",
      "image_path": "outputs/evidence_92.740s.jpg",
      "sources": [
        "asr"
      ],
      "evidence": {
        "speech_match": true,
        "visual_text_match": false
      },
      "scores": {
        "asr": 1.0,
        "ocr": null,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1,
  "search_summary": {
    "scout_frames": 122,
    "detector_frames": 25,
    "ocr_calls": 35,
    "candidates_found": 1,
    "tracked_events": 22,
    "runtime_seconds": 207.67
  }
}
```

### CASE_03 — Thank you for watching
- **Video URL**: `https://www.youtube.com/watch?v=YVvD7SZ7kc0`
- **Query**: `Thank you for watching`
- **Ground Truth**: `visual_only`
- **Expected Status**: `FOUND`
- **Actual Status**: `NOT_FOUND`

**Actual JSON Output**:
```json
{
  "status": "NOT_FOUND",
  "results": [],
  "total_candidates": 0,
  "search_summary": {
    "scout_frames": 122,
    "detector_frames": 25,
    "ocr_calls": 29,
    "candidates_found": 0,
    "tracked_events": 22,
    "runtime_seconds": 321.96
  }
}
```

### CASE_04 — have you quite given up on me?
- **Video URL**: `https://www.youtube.com/watch?v=YVvD7SZ7kc0`
- **Query**: `have you quite given up on me?`
- **Ground Truth**: `spoken_only`
- **Expected Status**: `FOUND`
- **Actual Status**: `NOT_FOUND`

**Actual JSON Output**:
```json
{
  "status": "NOT_FOUND",
  "results": [],
  "total_candidates": 0,
  "search_summary": {
    "scout_frames": 122,
    "detector_frames": 25,
    "ocr_calls": 35,
    "candidates_found": 0,
    "tracked_events": 22,
    "runtime_seconds": 441.87
  }
}
```

### CASE_05 — At least tell me your name
- **Video URL**: `https://www.youtube.com/shorts/WZORRHNP9_w`
- **Query**: `At least tell me your name`
- **Ground Truth**: `spoken_and_visual`
- **Expected Status**: `FOUND`
- **Actual Status**: `FOUND`
- **Evidence Image**: `[tests/evaluation_results/evidence/case_05/at_least_tell_me_your_name_001.jpg](tests/evaluation_results/evidence/case_05/at_least_tell_me_your_name_001.jpg)`

**Actual JSON Output**:
```json
{
  "status": "FOUND",
  "results": [
    {
      "timestamp": "00:00:09.900",
      "timestamp_seconds": 9.9,
      "frame_number": 297,
      "extracted_text": "At least tell me your name",
      "match_strength": 0.95,
      "match_level": "HIGH",
      "image_path": "outputs/evidence_9.900s.jpg",
      "sources": [
        "asr"
      ],
      "evidence": {
        "speech_match": true,
        "visual_text_match": false
      },
      "scores": {
        "asr": 0.95,
        "ocr": null,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1,
  "search_summary": {
    "scout_frames": 25,
    "detector_frames": 5,
    "ocr_calls": 23,
    "candidates_found": 1,
    "tracked_events": 11,
    "runtime_seconds": 107.47
  }
}
```

### CASE_06 — Batman
- **Video URL**: `https://www.youtube.com/shorts/WZORRHNP9_w`
- **Query**: `Batman`
- **Ground Truth**: `neither`
- **Expected Status**: `NOT_FOUND`
- **Actual Status**: `ERROR`

**Actual JSON Output**:
```json
{
  "status": "ERROR",
  "error": "yt-dlp download failed for URL 'https://www.youtube.com/shorts/WZORRHNP9_w': ERROR: Postprocessing: Error opening input files: Invalid data found when processing input"
}
```

### CASE_07 — The blue elephant is dancing
- **Video URL**: `https://www.youtube.com/shorts/WZORRHNP9_w`
- **Query**: `The blue elephant is dancing`
- **Ground Truth**: `neither`
- **Expected Status**: `NOT_FOUND`
- **Actual Status**: `NOT_FOUND`

**Actual JSON Output**:
```json
{
  "status": "NOT_FOUND",
  "results": [],
  "total_candidates": 0,
  "search_summary": {
    "scout_frames": 25,
    "detector_frames": 5,
    "ocr_calls": 11,
    "candidates_found": 0,
    "tracked_events": 11,
    "runtime_seconds": 60.87
  }
}
```
