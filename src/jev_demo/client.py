"""Stdlib-only client for Ollama's decision-model endpoint.

API shape verified against the official Ollama blog post "Ollama now
supports Jev-style decision models" (September 29, 2026,
https://ollama.com/blog/ollama-now-supports-jev-style-decision-models):

    POST http://localhost:11434/v1/systemone
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

Response:

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

Notes on the shape (all from that official example):

- Question types use TypeSafe's Jev API names. The yes/no type is
  spelled "noul", not "boolean" — see questions.boolean_question(),
  which builds a "noul" question for you.
- "state" is usually an object ({"ticket": "..."}) but plain strings
  are also accepted by the endpoint in published examples. This client
  passes state through untouched, so either form works.
- A "choice" answer carries the winning option, a probability for
  every option, and a confidence value. A "noul" answer carries only
  the probability that the answer is yes — there is no separate
  confidence field; the probability IS the confidence. A "score"
  answer carries a float: the expected position on your ordered scale
  (0.815 on [Routine, Soon, Urgent] means "between Routine and Soon,
  leaning Soon"), plus a legend mapping positions to labels.
- Local requests need no API key. A 404 on /v1/systemone means the
  Ollama server is older than 0.35 — decision support arrived in 0.35.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


class DecisionError(Exception):
    """Anything that stops a decision request from returning answers.

    The message is written for a human at a terminal: it says what
    happened and what to try next (start Ollama, upgrade it, pull the
    model), instead of leaking a raw traceback.
    """


@dataclass
class ChoiceAnswer:
    """A "choice" answer: one winning option plus its distribution."""

    name: str
    choice: str
    probabilities: Dict[str, float]
    confidence: Optional[float] = None


@dataclass
class BooleanAnswer:
    """A "noul" (yes/no) answer: probability that the answer is yes."""

    name: str
    probability: float

    @property
    def yes(self) -> bool:
        """True when the model leans yes (probability >= 0.5)."""
        return self.probability >= 0.5


@dataclass
class ScoreAnswer:
    """A "score" answer: expected position on an ordered scale."""

    name: str
    score: float
    legend: Dict[str, str] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None


Answer = Union[ChoiceAnswer, BooleanAnswer, ScoreAnswer]


@dataclass
class DecideResult:
    """Everything one /v1/systemone call returns, parsed and timed."""

    answers: Dict[str, Answer]
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


def _parse_answer(name: str, raw: Any, expected_type: Optional[str]) -> Answer:
    """Turn one raw answer dict into a typed answer object."""
    if not isinstance(raw, dict):
        raise DecisionError(f"Answer {name!r} is not an object: {raw!r}")
    answer_type = raw.get("type")
    if expected_type is not None and answer_type != expected_type:
        raise DecisionError(
            f"Answer {name!r} came back as type {answer_type!r}, "
            f"but the question asked for {expected_type!r}."
        )
    if answer_type == "choice":
        if "choice" not in raw or "probabilities" not in raw:
            raise DecisionError(
                f"Choice answer {name!r} is missing 'choice' or "
                f"'probabilities': {raw!r}"
            )
        return ChoiceAnswer(
            name=name,
            choice=raw["choice"],
            probabilities=dict(raw["probabilities"]),
            confidence=raw.get("confidence"),
        )
    if answer_type == "noul":
        if "noul" not in raw:
            raise DecisionError(
                f"Yes/no answer {name!r} is missing 'noul': {raw!r}"
            )
        return BooleanAnswer(name=name, probability=float(raw["noul"]))
    if answer_type == "score":
        if "score" not in raw:
            raise DecisionError(
                f"Score answer {name!r} is missing 'score': {raw!r}"
            )
        return ScoreAnswer(
            name=name,
            score=float(raw["score"]),
            legend=dict(raw.get("legend", {})),
            probabilities=dict(raw.get("probabilities", {})),
            confidence=raw.get("confidence"),
        )
    raise DecisionError(
        f"Answer {name!r} has unknown type {answer_type!r}: {raw!r}"
    )


class SystemOneClient:
    """Minimal client for POST /v1/systemone. Stdlib only (urllib)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "tev1:0.8b",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def decide(
        self,
        state: Union[str, Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> DecideResult:
        """Ask all questions about one state in a single request.

        Returns a DecideResult with typed answers, token usage (when
        the server reports it), and the wall-clock time of the call in
        milliseconds. Raises DecisionError on any failure.
        """
        url = f"{self.base_url}/v1/systemone"
        body = json.dumps(
            {
                "model": model or self.model,
                "state": state,
                "questions": questions,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            wanted = model or self.model
            # A 404 can mean two different things: an Ollama older than
            # 0.35 (no /v1/systemone at all) or a model that was never
            # pulled. The body tells them apart — a missing model is
            # named in it, a missing endpoint is not.
            if "not found" in detail.lower() and wanted in detail:
                raise DecisionError(
                    f"Ollama does not have model {wanted!r}. "
                    f"Pull it first: ollama pull {wanted} "
                    f"(server said: {detail[:200]})"
                ) from exc
            if exc.code == 404:
                raise DecisionError(
                    "Ollama returned 404 for /v1/systemone. That endpoint "
                    "arrived in Ollama 0.35 — upgrade Ollama "
                    "(check with: ollama --version) and retry."
                ) from exc
            raise DecisionError(
                f"Ollama returned HTTP {exc.code} for /v1/systemone"
                + (f": {detail[:200]}" if detail else ".")
            ) from exc
        except urllib.error.URLError as exc:
            raise DecisionError(
                f"Cannot reach Ollama at {self.base_url}. Is Ollama "
                f"running? Start it with `ollama serve` (or launch the "
                f"Ollama app) and retry. ({exc.reason})"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise DecisionError(
                f"Ollama returned a response that is not valid JSON: "
                f"{raw_body[:200]!r}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("answers"), dict
        ):
            raise DecisionError(
                f"Ollama response has no 'answers' object: {raw_body[:200]!r}"
            )
        expected = {name: q.get("type") for name, q in questions.items()}
        answers = {
            name: _parse_answer(name, raw, expected.get(name))
            for name, raw in payload["answers"].items()
        }
        return DecideResult(
            answers=answers,
            model=payload.get("model", model or self.model),
            usage=payload.get("usage", {}),
            elapsed_ms=elapsed_ms,
        )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    """Best-effort read of an HTTP error body (never raises)."""
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
