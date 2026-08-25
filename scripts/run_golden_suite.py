"""
scripts/run_golden_suite.py
Canonical Golden Benchmark Runner for Video Dialogue Locator (v1.0.0).
Executes all 7 Golden Test Cases, saves evaluation JSONs & evidence images,
and outputs a clean evaluation summary table.
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_golden_suite")

def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "v1.0.0"

def run_golden_suite():
    manifest_path = Path("tests/fixtures/golden_tests.yaml")
    if not manifest_path.exists():
        print(f"Error: Test manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        golden_cases = yaml.safe_load(f)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_results_dir = Path("tests/evaluation_results")
    evidence_dir = Path("outputs/evaluation")
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    passed_count = 0
    total_count = len(golden_cases)

    print("\nGolden Evaluation")
    print("--------------------------------------------")

    for idx, case in enumerate(golden_cases, 1):
        case_id = case["id"]
        case_name = case["name"]
        url = case["video_url"]
        query = case["query"]
        ground_truth = case["ground_truth"]
        expected_status = case["expected_status"]

        res = locate_dialogue_in_video(
            url=url,
            target_text=query,
            config=config,
            output_dir=Path("outputs/runtime")
        )

        actual_status = res.get("status", "ERROR")
        passed = (actual_status == expected_status)
        reason = f"Status matched '{expected_status}'" if passed else f"Status mismatch: got '{actual_status}'"

        if passed and actual_status == "FOUND" and res.get("results"):
            top_res = res["results"][0]
            exp_sources = set(case.get("expected_sources", []))
            act_sources = set(top_res.get("sources", []))
            if exp_sources and not exp_sources.issubset(act_sources):
                passed = False
                reason = f"Sources mismatch: got {top_res.get('sources')}, expected {case.get('expected_sources')}"

        if passed:
            passed_count += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"

        case_label = f"Case {idx}"
        print(f"{case_label:<9} {actual_status:<11} {status_str}")

        # Save evidence image into outputs/evaluation/<case_id>/
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

        # Save individual result JSON
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

    print("--------------------------------------------")
    print(f"{passed_count} / {total_count} PASS\n")

    return passed_count == total_count

if __name__ == "__main__":
    success = run_golden_suite()
    sys.exit(0 if success else 1)
