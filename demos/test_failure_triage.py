#!/usr/bin/env python3
"""Demo 2: test-failure triage with a decision model.

A QA-flavoured version of the same pattern: given a failure log
snippet, the model decides in ONE request whether the failure looks
flaky, environmental, or a real bug (choice), and whether re-running
is worthwhile (yes/no). That verdict is exactly the first branch of
a run > report > triage workflow — decided by a small local model
instead of a generative call that must be parsed back into a label.

Run from the repo root (no install step, stdlib only):

    python demos/test_failure_triage.py
    python demos/test_failure_triage.py --model nimble

Requires Ollama 0.35+ and a pulled decision model
(see README.md). All log snippets here are fictional.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_demo.runner import data_file_here, run_demo

if __name__ == "__main__":
    raise SystemExit(
        run_demo(data_file_here(__file__, "test_failure_triage.json"),
                 "Test-failure triage")
    )
