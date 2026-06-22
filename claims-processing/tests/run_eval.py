#!/usr/bin/env python3
"""
tests/run_eval.py
-----------------
Runs the full eval suite against a live server.

Usage:
    # Server must be running:
    python tests/run_eval.py [--url http://localhost:8000] [--cases test_cases.json]

Output:
    tests/eval_report.json  — machine-readable results
    Console                 — summary table

Each test case in test_cases.json must have:
    id              str     test case identifier (e.g. "TC001")
    description     str     human-readable description
    member_id       str
    policy_id       str
    claim_category  str     e.g. "CONSULTATION"
    treatment_date  str     ISO date
    claimed_amount  float
    files           list    list of file paths relative to repo root
    expected_decision str   APPROVED / PARTIAL / REJECTED / MANUAL_REVIEW
    expected_approved_amount  float | null   (null = not checked)

Optional per-case:
    hospital_name   str
    ytd_claims_amount float
    expected_confidence_min float   minimum acceptable confidence score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://localhost:8000"
DEFAULT_CASES = str(Path(__file__).parent.parent / "test_cases.json")
DEFAULT_REPORT = str(Path(__file__).parent / "eval_report.json")

TIMEOUT = 120  # seconds per request (OCR can be slow)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(client: httpx.Client, base_url: str, case: dict[str, Any]) -> dict[str, Any]:
    """
    Submit one test case and return a result dict.
    """
    tc_id = case.get("id", "UNKNOWN")
    start = time.monotonic()

    # Build multipart form data
    data: dict[str, Any] = {
        "member_id":       case["member_id"],
        "policy_id":       case["policy_id"],
        "claim_category":  case["claim_category"],
        "treatment_date":  case["treatment_date"],
        "claimed_amount":  str(case["claimed_amount"]),
        "ytd_claims_amount": str(case.get("ytd_claims_amount", 0.0)),
    }
    if "hospital_name" in case and case["hospital_name"]:
        data["hospital_name"] = case["hospital_name"]

    # Open files
    file_handles: list = []
    files_param: list = []
    missing_files: list[str] = []

    for rel_path in case.get("files", []):
        p = Path(rel_path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / rel_path
        if not p.exists():
            missing_files.append(str(rel_path))
            continue
        fh = open(p, "rb")
        file_handles.append(fh)
        files_param.append(("files", (p.name, fh, _mime(p))))

    if missing_files:
        return {
            "id": tc_id,
            "description": case.get("description", ""),
            "passed": False,
            "error": f"Missing files: {missing_files}",
            "duration_s": round(time.monotonic() - start, 2),
        }

    if not files_param:
        return {
            "id": tc_id,
            "description": case.get("description", ""),
            "passed": False,
            "error": "No files specified for this test case.",
            "duration_s": round(time.monotonic() - start, 2),
        }

    try:
        resp = client.post(
            f"{base_url}/claims/submit",
            data=data,
            files=files_param,
            timeout=TIMEOUT,
        )
        duration = round(time.monotonic() - start, 2)
    except Exception as exc:
        return {
            "id": tc_id,
            "description": case.get("description", ""),
            "passed": False,
            "error": f"HTTP error: {exc}",
            "duration_s": round(time.monotonic() - start, 2),
        }
    finally:
        for fh in file_handles:
            fh.close()

    # Parse response
    if resp.status_code not in (200, 201):
        return {
            "id": tc_id,
            "description": case.get("description", ""),
            "passed": False,
            "http_status": resp.status_code,
            "error": resp.text[:500],
            "duration_s": duration,
        }

    try:
        body = resp.json()
    except Exception:
        return {
            "id": tc_id,
            "description": case.get("description", ""),
            "passed": False,
            "error": "Response was not JSON",
            "duration_s": duration,
        }

    actual_decision = body.get("decision", "")
    actual_approved = body.get("approved_amount", 0.0)
    actual_confidence = body.get("confidence_score", 0.0)
    claim_id = body.get("claim_id", "")

    expected_decision = case.get("expected_decision", "")
    expected_approved = case.get("expected_approved_amount")
    expected_conf_min = case.get("expected_confidence_min", 0.0)

    failures: list[str] = []

    if actual_decision != expected_decision:
        failures.append(
            f"decision: expected={expected_decision} actual={actual_decision}"
        )

    if expected_approved is not None:
        # Allow ±5 tolerance for float rounding
        if abs(actual_approved - expected_approved) > 5.0:
            failures.append(
                f"approved_amount: expected={expected_approved} actual={actual_approved}"
            )

    if actual_confidence < expected_conf_min:
        failures.append(
            f"confidence: expected≥{expected_conf_min} actual={actual_confidence}"
        )

    passed = len(failures) == 0

    return {
        "id": tc_id,
        "description": case.get("description", ""),
        "claim_id": claim_id,
        "passed": passed,
        "failures": failures,
        "actual_decision": actual_decision,
        "expected_decision": expected_decision,
        "actual_approved_amount": actual_approved,
        "expected_approved_amount": expected_approved,
        "actual_confidence": actual_confidence,
        "duration_s": duration,
        "pipeline_complete": body.get("pipeline_complete", True),
        "rejection_reasons": body.get("rejection_reasons", []),
        "fraud_flags": (body.get("fraud_result") or {}).get("flags", []),
    }


def _mime(p: Path) -> str:
    ext = p.suffix.lower()
    return {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run claims eval suite")
    parser.add_argument("--url",   default=DEFAULT_URL,   help="Base URL of the running server")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Path to test_cases.json")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Output path for eval_report.json")
    parser.add_argument("--tc",    default=None, help="Run only this test case ID (e.g. TC001)")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"[ERROR] test_cases.json not found at {cases_path}", file=sys.stderr)
        sys.exit(1)

    with open(cases_path, encoding="utf-8") as f:
        all_cases: list[dict] = json.load(f)

    if args.tc:
        all_cases = [c for c in all_cases if c.get("id") == args.tc]
        if not all_cases:
            print(f"[ERROR] Test case '{args.tc}' not found.", file=sys.stderr)
            sys.exit(1)

    # Check server is up
    with httpx.Client() as probe:
        try:
            r = probe.get(f"{args.url}/health", timeout=10)
            r.raise_for_status()
        except Exception as exc:
            print(f"[ERROR] Server not reachable at {args.url}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"\nRunning {len(all_cases)} test case(s) against {args.url}\n")
    print(f"{'ID':<8} {'Decision':^20} {'Approved':>10} {'Conf':>6} {'Pass':>5} {'s':>6}  Description")
    print("-" * 100)

    results: list[dict] = []
    passed_count = 0

    with httpx.Client() as client:
        for case in all_cases:
            r = run_case(client, args.url, case)
            results.append(r)

            if r["passed"]:
                passed_count += 1
                status_sym = "✓"
            else:
                status_sym = "✗"

            decision_str = f"{r.get('actual_decision','?')} / {r.get('expected_decision','?')}"
            approved_str = f"{r.get('actual_approved_amount', 0.0):>8.0f}"
            conf_str = f"{r.get('actual_confidence', 0.0):>5.2f}"
            dur_str = f"{r.get('duration_s', 0.0):>6.1f}"
            desc = r.get("description", "")[:45]

            print(
                f"{r['id']:<8} {decision_str:^20} {approved_str} {conf_str} "
                f"{status_sym:>5} {dur_str}  {desc}"
            )

            if not r["passed"] and r.get("failures"):
                for fail in r["failures"]:
                    print(f"         ↳ FAIL: {fail}")
            if r.get("error"):
                print(f"         ↳ ERROR: {r['error']}")

    total = len(results)
    print("-" * 100)
    print(f"\nResults: {passed_count}/{total} passed")

    # Write report
    report = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server_url": args.url,
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(passed_count / total, 4) if total else 0.0,
        "results": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport written to {report_path}")
    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
