# ollama-jev-decision-demo

A small demo repo for trying **Jev-style decision models** through
Ollama — local models that answer typed questions instead of
generating prose. No new dependencies: Python standard library only.

> Status: **live-verified.** The ticket-triage demo ran on a CPU-only
> 2017 iMac (Ollama 0.40.10): `tev1:0.8b` answers 3 typed questions per
> case in ~3.0 s steady-state with all 4 labels correct; `tev1` (4B)
> takes ~16.9 s/case and mislabeled one case the smaller model got
> right. See [Live timing on CPU](#live-timing-on-cpu-measured-2026-10-31) for the numbers.

## What a decision model is (and is not)

A generative LLM answers in paragraphs. You then have to parse the
paragraph back into a label, a yes/no, or a score — and the parse
breaks whenever the model wraps the answer in markdown, hedges, or
rephrases the label.

A decision model skips the prose. You send it a block of text (the
**state**) plus a set of **typed questions**, and it answers all of
them in one request with machine-usable numbers:

| Question type | You ask | You get back |
| --- | --- | --- |
| `choice` | Pick one option from a list | The winning option, a probability for **every** option, and a confidence value |
| `noul` | A yes/no question | A single probability: how likely the answer is yes |
| `score` | Where on an ordered scale? | A float (the expected position on the scale), a legend mapping positions to labels, probabilities, and confidence |

One request, all questions, no parsing. That is the whole idea — and
why these models fit inside an inner loop: ticket triage, test
triage, moderation, routing. They do not chat, write, or reason out
loud. If you need prose, use a generative model; if you need a
decision, use one of these.

### One naming gotcha

The yes/no type is spelled **`noul`** in the API (TypeSafe's Jev API
name), not `boolean`. This repo's `boolean_question()` builder hides
that: you write `boolean_question(...)`, it sends `"type": "noul"`.

## The API shape (verified)

Verified against Ollama's official blog post, *"Ollama now supports
Jev-style decision models"* (September 29, 2026):
<https://ollama.com/blog/ollama-now-supports-jev-style-decision-models>.
Decision support arrived in **Ollama 0.35**, at
`POST http://localhost:11434/v1/systemone`. Local requests need no
API key.

Request:

```json
{
  "model": "nimble",
  "state": {"ticket": "I was charged twice. Please refund the extra payment."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {"billing": "Payments and refunds", "technical": "Bugs and integrations"}
    },
    "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Routine", "Soon", "Urgent"]
    }
  }
}
```

Response (from the same official example):

```json
{
  "model": "nimble",
  "answers": {
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.985, "technical": 0.012},
             "confidence": 0.922},
    "refund": {"type": "noul", "noul": 0.997},
    "urgency": {"type": "score", "score": 0.815,
                "legend": {"0": "Routine", "1": "Soon", "2": "Urgent"},
                "probabilities": {"0": 0.378, "1": 0.429, "2": 0.193},
                "confidence": 0.046}
  },
  "usage": {"input_tokens": 841, "output_tokens": 4}
}
```

Read the numbers like this:

- **Choice:** `choice` is the winner; `probabilities` is the full
  distribution (they sum to ~1); `confidence` is the model's own
  certainty signal. A winner at 0.51 with confidence 0.1 is a coin
  flip, not a decision — route those to a human.
- **Yes/no:** there is no separate confidence field. The `noul`
  probability **is** the confidence: 0.997 means "almost certainly
  yes"; 0.52 means "barely yes". This demo treats >= 0.5 as yes.
- **Score:** the float is the *expected position* on your scale.
  0.815 on `[Routine, Soon, Urgent]` means "between Routine and
  Soon, leaning Soon" — more information than either label alone.
  The `legend` maps positions back to your labels.

## Prerequisites

1. **Ollama 0.35 or newer.** Check yours:

   ```bash
   ollama --version
   ```

   If `POST /v1/systemone` returns 404, your Ollama is older than
   0.35 — upgrade it. (The client in this repo detects that 404 and
   tells you exactly this, instead of showing a raw error.)

2. **Pull a decision model.** Start small — the 0.8B model is the
   CPU-first try:

   ```bash
   ollama pull tev1:0.8b   # ~811 MB, Together AI, experimental
   ollama pull tev1         # ~4.5 GB, Together AI, experimental
   ollama pull nimble       # ~9.5 GB, Bespoke Labs, open source
   ```

   Later catalog additions exist (Nimble v3, Kev, Clef, and others,
   plus `ollama run --ask` in 0.40.4+ for quick CLI checks), but this
   repo builds on the v1 core: the `/v1/systemone` HTTP API and the
   three launch models above.

## Honest expectations on CPU

Ollama's headline figure — nimble 9B averaging 91 ms per decision —
was measured on an **M5 Max** (Apple Silicon, MLX). Do not expect
that number on an Intel iMac: it is CPU-only, MLX is
Apple-Silicon-only, and the vision-encoder-free decision models still
run on plain CPU cores here.

So this repo does not promise speed. It **measures** it: every demo
prints per-case milliseconds, an average per question, and a
steady-state average that excludes the first case (the first request
includes loading the model into memory, so it is always the slowest
and the least representative). Pull `tev1:0.8b` first, measure it,
and only move to `nimble` if the numbers justify it.

## Run the demos

No install step. Python 3.9+ standard library only. From the repo
root:

```bash
python demos/ticket_triage.py
python demos/test_failure_triage.py
python demos/content_moderation.py
```

Each demo runs its fictional cases through one model (default
`tev1:0.8b`) and prints every answer with its probabilities and
confidence, then the timing summary. Try another model with:

```bash
python demos/ticket_triage.py --model nimble
python demos/ticket_triage.py --model tev1 --base-url http://localhost:11434
```

### The three demos

1. **Ticket triage** (`demos/ticket_triage.py`, cases in
   `data/ticket_triage.json`): label (choice: billing / bug /
   account / feature), priority (choice), needs a human (yes/no).
   The official blog's own example shape, extended to a full ticket.
2. **Test-failure triage** (`demos/test_failure_triage.py`): verdict
   (choice: flaky / infra / real-bug) and whether re-running is
   worthwhile (yes/no). This is the QA-flavoured one: the verdict is
   the first branch of a run > report > triage workflow, decided by
   a small local model instead of a generative call you must parse.
3. **Content moderation** (`demos/content_moderation.py`): allowed
   (yes/no) and category (choice: spam / harassment / safe). Watch
   the yes/no probability: allowed at 0.52 and allowed at 0.99 are
   different cases, and the demo prints both so you can see where a
   human-review threshold would bite.

A quick manual check without any demo, via curl:

```bash
curl http://localhost:11434/v1/systemone -d '{
  "model": "tev1:0.8b",
  "state": {"ticket": "I was charged twice. Please refund the extra payment."},
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"}
  }
}'
```

## Write your own questions

Use the builders in `src/jev_demo/questions.py` (validated up
front, so a typo fails here with a clear message, not as a server
error), or write the raw schema by hand — the demos load question
schemas straight from the `data/*.json` files, so editing a JSON
file is enough to try new questions:

```python
from jev_demo.client import SystemOneClient
from jev_demo.questions import choice_question, boolean_question, score_question

client = SystemOneClient(model="tev1:0.8b")
result = client.decide(
    {"ticket": "Our checkout has returned 500 errors since 9am."},
    {
        "label": choice_question(
            "Which label fits this ticket?",
            {"billing": "Payments and refunds", "bug": "Software errors"},
        ),
        "urgent": boolean_question("Is this ticket urgent?"),
    },
)
for name, answer in result.answers.items():
    print(name, "->", answer)
print(f"({result.elapsed_ms:.0f} ms, usage: {result.usage})")
```

`state` may be an object (as in the official example) or a plain
string — the client passes it through untouched. If a request fails,
you get a `DecisionError` with a human-readable next step: Ollama
not running, Ollama older than 0.35, or a model that was never
pulled (with the exact `ollama pull` command).

## Repo layout

```text
ollama-jev-decision-demo/
├── README.md                  # this guide
├── LICENSE                    # MIT
├── src/jev_demo/
│   ├── client.py              # SystemOneClient: POST /v1/systemone, parsing, errors
│   ├── questions.py           # choice / boolean (noul) / score question builders
│   ├── reporting.py           # human-readable rendering of answers
│   └── runner.py              # shared demo runner: load cases, decide, print, time
├── demos/
│   ├── ticket_triage.py       # demo 1 (thin wrapper around runner.py)
│   ├── test_failure_triage.py # demo 2
│   └── content_moderation.py  # demo 3
├── data/
│   ├── ticket_triage.json     # fictional tickets + question schemas
│   ├── test_failure_triage.json
│   └── content_moderation.json
└── tests/
    ├── test_client.py         # parsing + error handling, mocked endpoint
    └── test_questions.py      # builders + reporting
```

No secrets, no tokens, no committed model weights. `.gitignore`
covers virtualenvs and bytecode. Nothing here phones home: every
live call goes to your own Ollama server on localhost.

## Test documentation

### What ran in the sandbox (verified)

```bash
python -m unittest discover -s tests -v
```

- **Tests pass** against a **mocked** `/v1/systemone` endpoint
  (the sandbox has no Ollama and no decision models, so no test
  touches a live server). Covered: parsing of the exact official
  blog response (choice + noul + score, probabilities, confidence,
  usage); string and object states; answer-type mismatch; Ollama
  down (helpful error); 404 without a model name (old-Ollama
  message); 404 naming the model (pull suggestion); malformed JSON;
  missing `answers`; question-builder shapes and validation;
  answer reporting.
- Every Python file is `py_compile`-clean.
- README claims were checked against the code (endpoint, types,
  field names) and against Ollama's official post.

### Live timing on CPU (measured 2026-10-31)

Ticket-triage demo, 4 cases × 3 typed questions, 2017 iMac
(i7-7700K, CPU-only), Ollama 0.40.10. Steady-state excludes the first
case (model load):

| Model | Per case | Per question | Labels correct |
|---|---|---|---|
| tev1:0.8b | 3.0 s | 1.0 s | 4/4 |
| tev1 (4B) | 16.9 s | 5.6 s | 3/4 |

The 4B model was 5.5x slower **and** labeled the checkout-500-errors
ticket "billing" (62.6%) where the 0.8B model correctly said "bug"
(74.7%). Bigger ≠ better here — and neither is near the 91 ms M5 Max
figure, as expected on CPU. `nimble` (9B) not measured yet.

## License

MIT — see [LICENSE](LICENSE).
