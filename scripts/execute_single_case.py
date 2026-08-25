"""
scripts/execute_single_case.py
Executes a SINGLE Golden Test Case specified by 1-based index (e.g. python scripts/execute_single_case.py 1),
prints the complete pipeline logs, actual returned JSON, saves the JSON result file, and copies evidence image.
"""

import sys
import json
import yaml
import shutil
import time
import platform
import subprocess
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import locate_dialogue_in_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("single_case_executor")

def execute_case(case_num: int):
    manifest_path = Path("tests/fixtures/golden_tests.yaml")
    if not manifest_path.exists():
        logger.error(f"Test manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        golden_cases = yaml.safe_load(f)

    if case_num < 1 or case_num > len(golden_cases):
        logger.error(f"Invalid case number: {case_num}. Must be between 1 and {len(golden_cases)}.")
        sys.exit(1)

    case = golden_cases[case_num - 1]
    case_id = case["id"]
    case_name = case["name"]
    url = case["video_url"]
    query = case["query"]
    ground_truth = case["ground_truth"]
    expected_status = case["expected_status"]

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_results_dir = Path("tests/evaluation_results")
    evidence_dir = Path("outputs/evaluation")
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n============================================================")
    print(f"GOLDEN TEST CASE {case_num}/{len(golden_cases)} — {case_name}")
    print(f"============================================================")
    print(f"Video URL:    {url}")
    print(f"Query:        '{query}'")
    print(f"Ground Truth: {ground_truth}")
    print(f"Expected:     {expected_status}")
    print("Running pipeline...\n")

    res = locate_dialogue_in_video(
        url=url,
        target_text=query,
        config=config,
        output_dir=Path("outputs/runtime")
    )

    actual_json_str = json.dumps(res, indent=2)
    print("\nFINAL JSON:")
    print(actual_json_str)
    print("============================================================\n")

    actual_status = res.get("status", "ERROR")
    passed = (actual_status == expected_status)
    reason = f"Status matched expected '{expected_status}'" if passed else f"Status mismatch: got '{actual_status}', expected '{expected_status}'"

    if passed and actual_status == "FOUND" and res.get("results"):
        top_res = res["results"][0]
        exp_sources = set(case.get("expected_sources", []))
        act_sources = set(top_res.get("sources", []))
        if exp_sources and not exp_sources.issubset(act_sources):
            passed = False
            reason = f"Sources mismatch: got {top_res.get('sources')}, expected {case.get('expected_sources')}"

    # Save evidence image if FOUND into outputs/evaluation/<case_id>/
    case_evidence_dir = evidence_dir / case_id
    case_evidence_dir.mkdir(parents=True, exist_ok=True)
    
    saved_img_path = ""
    if actual_status == "FOUND" and res.get("results"):
        orig_img = res["results"][0].get("image_path")
        if orig_img and Path(orig_img).exists():
            safe_q_name = "".join([c if c.isalnum() else "_" for c in query.lower()])[:30]
            img_name = f"{safe_q_name}_001.jpg"
            dest_img = case_evidence_dir / img_name
            shutil.copy2(orig_img, dest_img)
            saved_img_path = str(dest_img).replace("\\", "/")

    # Save case JSON file
    case_result_data = {
        "case_id": case_id,
        "case_name": case_name,
        "video_url": url,
        "query": query,
        "ground_truth": ground_truth,
        "expected_status": expected_status,
        "expected_sources": case.get("expected_sources", []),
        "actual_result": res,
        "evidence_image_path": saved_img_path,
        "evaluation": {
            "passed": passed,
            "reason": reason
        }
    }

    json_out_file = eval_results_dir / f"{case_name}.json"
    with open(json_out_file, "w", encoding="utf-8") as f_out:
        json.dump(case_result_data, f_out, indent=2)
    
    logger.info(f"Saved case result JSON: {json_out_file}")
    if saved_img_path:
        logger.info(f"Saved evidence frame image: {saved_img_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        case_idx = 1
    else:
        case_idx = int(sys.argv[1])
    execute_case(case_idx)
