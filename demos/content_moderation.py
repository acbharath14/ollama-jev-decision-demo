#!/usr/bin/env python3
"""Demo 3: content moderation with a decision model.

Given a user comment, the model decides in ONE request whether the
comment is allowed (yes/no) and which category it falls into
(choice). The probability on the yes/no answer is the moderation
signal: a comment allowed at 0.52 is a very different case from one
allowed at 0.99, and the demo prints both so you can see where a
human-review threshold would bite.

Run from the repo root (no install step, stdlib only):

    python demos/content_moderation.py
    python demos/content_moderation.py --model nimble

Requires Ollama 0.35+ and a pulled decision model
(see README.md). All comments here are fictional.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_demo.runner import data_file_here, run_demo

if __name__ == "__main__":
    raise SystemExit(
        run_demo(data_file_here(__file__, "content_moderation.json"),
                 "Content moderation")
    )
