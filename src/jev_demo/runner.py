"""Shared runner for the demos: load cases, decide, print, time.

Each demo is a thin script that calls run_demo() with its own data
file. The runner prints every case's answers, then a timing summary:

- ms per case and ms per question, averaged over the run;
- a note that the FIRST case includes model loading (Ollama loads
  the model into memory on first use), so the honest steady-state
  number is the average excluding the first case.

Nothing here hides a slow first call inside an average — both numbers
are printed, and the first case is labelled.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .client import DecisionError, SystemOneClient
from .reporting import format_result


def run_demo(data_path: str, title: str) -> int:
    """Run every case in a data file. Returns 0 on success, 1 on error."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--model", default="tev1:0.8b",
                        help="decision model to use (default: tev1:0.8b)")
    parser.add_argument("--base-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    args = parser.parse_args()

    with open(data_path, encoding="utf-8") as handle:
        data = json.load(handle)
    cases: List[Dict[str, Any]] = data["cases"]
    client = SystemOneClient(base_url=args.base_url, model=args.model)

    print(f"{title} — model: {args.model}, {len(cases)} cases\n")
    case_times: List[float] = []
    question_count = 0
    try:
        for index, case in enumerate(cases):
            result = client.decide(case["state"], case["questions"])
            case_times.append(result.elapsed_ms)
            question_count += len(case["questions"])
            label = f"Case {index + 1}: {case.get('id', '')}"
            if index == 0:
                label += " (includes model load time)"
            print(format_result(result, label))
            print()
    except DecisionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total_ms = sum(case_times)
    steady = case_times[1:] if len(case_times) > 1 else case_times
    print("Timing")
    print(f"  total: {total_ms:.0f} ms over {len(cases)} cases, "
          f"{question_count} questions")
    print(f"  average per case: {total_ms / len(case_times):.0f} ms")
    if question_count:
        print(f"  average per question: {total_ms / question_count:.0f} ms")
    if len(steady) != len(case_times):
        print(f"  steady-state average per case (excluding first, "
              f"model-load case): {sum(steady) / len(steady):.0f} ms")
    print("\nNote: these are YOUR machine's numbers. Compare models by "
          "re-running with --model, not by quoting any vendor figure.")
    return 0


def data_file_here(caller_file: str, name: str) -> str:
    """Resolve data/<name> relative to the repo root (demos/ sibling)."""
    return str(Path(caller_file).resolve().parent.parent / "data" / name)
