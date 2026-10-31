#!/usr/bin/env python3
"""Demo 1: support-ticket triage with a decision model.

Each ticket gets three typed answers in ONE request: a label
(choice), a priority (choice), and whether a human needs to look at
it (yes/no). No prose, no prompt-engineering for JSON — the answers
come back typed, with a probability for every option.

Run from the repo root (no install step, stdlib only):

    python demos/ticket_triage.py
    python demos/ticket_triage.py --model nimble

Requires Ollama 0.35+ and a pulled decision model
(see README.md). All ticket text here is fictional.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_demo.runner import data_file_here, run_demo

if __name__ == "__main__":
    raise SystemExit(
        run_demo(data_file_here(__file__, "ticket_triage.json"),
                 "Ticket triage")
    )
