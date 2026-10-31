"""Tests for the /v1/systemone client.

Every test runs against a MOCKED endpoint — the sandbox has no
Ollama and no decision models, so no test here touches a live server.
A live run is Bharath's iMac job (see README.md).

The success payload is the exact example from Ollama's official blog
post (September 29, 2026), so the parser is tested against the real
documented shape, not an invented one.
"""

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_demo.client import (
    BooleanAnswer,
    ChoiceAnswer,
    DecisionError,
    ScoreAnswer,
    SystemOneClient,
)

# Exact response from Ollama's official September 29, 2026 blog post.
OFFICIAL_RESPONSE = {
    "model": "nimble",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.985, "technical": 0.012, "other": 0.003},
            "confidence": 0.922,
        },
        "refund": {"type": "noul", "noul": 0.997},
        "urgency": {
            "type": "score",
            "score": 0.815,
            "legend": {"0": "Routine", "1": "Soon", "2": "Urgent"},
            "probabilities": {"0": 0.378, "1": 0.429, "2": 0.193},
            "confidence": 0.046,
        },
    },
    "usage": {"input_tokens": 841, "output_tokens": 4},
}

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Payments and refunds", "technical": "Bugs and integrations"},
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Routine", "Soon", "Urgent"],
    },
}


class FakeResponse:
    """Minimal stand-in for urllib's HTTPResponse (context manager)."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(url, code, body):
    return urllib.error.HTTPError(
        url, code, "error", {}, io.BytesIO(body.encode("utf-8"))
    )


class ClientParsingTests(unittest.TestCase):
    def test_parses_official_example(self):
        client = SystemOneClient(model="nimble")
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse(OFFICIAL_RESPONSE)
        ):
            result = client.decide({"ticket": "I was charged twice."}, QUESTIONS)

        self.assertEqual(result.model, "nimble")
        self.assertEqual(result.usage, {"input_tokens": 841, "output_tokens": 4})
        self.assertGreaterEqual(result.elapsed_ms, 0.0)

        team = result.answers["team"]
        self.assertIsInstance(team, ChoiceAnswer)
        self.assertEqual(team.choice, "billing")
        self.assertAlmostEqual(team.probabilities["billing"], 0.985)
        self.assertAlmostEqual(team.confidence, 0.922)

        refund = result.answers["refund"]
        self.assertIsInstance(refund, BooleanAnswer)
        self.assertAlmostEqual(refund.probability, 0.997)
        self.assertTrue(refund.yes)

        urgency = result.answers["urgency"]
        self.assertIsInstance(urgency, ScoreAnswer)
        self.assertAlmostEqual(urgency.score, 0.815)
        self.assertEqual(urgency.legend, {"0": "Routine", "1": "Soon", "2": "Urgent"})
        self.assertAlmostEqual(urgency.confidence, 0.046)

    def test_boolean_yes_boundary(self):
        self.assertTrue(BooleanAnswer("q", 0.5).yes)
        self.assertFalse(BooleanAnswer("q", 0.49).yes)

    def test_state_string_and_dict_both_serialized(self):
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(OFFICIAL_RESPONSE)

        client = SystemOneClient(model="tev1:0.8b")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.decide("plain string state", QUESTIONS)
            client.decide({"ticket": "dict state"}, QUESTIONS)

        self.assertEqual(seen[0]["state"], "plain string state")
        self.assertEqual(seen[1]["state"], {"ticket": "dict state"})
        self.assertEqual(seen[0]["model"], "tev1:0.8b")
        self.assertEqual(seen[0]["questions"], QUESTIONS)

    def test_answer_type_mismatch_raises(self):
        wrong = {"model": "nimble",
                 "answers": {"refund": {"type": "choice", "choice": "x",
                                        "probabilities": {"x": 1.0}}}}
        client = SystemOneClient()
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse(wrong)
        ):
            with self.assertRaises(DecisionError) as ctx:
                client.decide({"ticket": "t"}, QUESTIONS)
        self.assertIn("noul", str(ctx.exception))


class ClientErrorTests(unittest.TestCase):
    def test_ollama_down_raises_helpful_error(self):
        client = SystemOneClient()
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(DecisionError) as ctx:
                client.decide("state", QUESTIONS)
        self.assertIn("ollama serve", str(ctx.exception))

    def test_404_without_model_name_means_old_ollama(self):
        client = SystemOneClient()
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=http_error(
                "http://localhost:11434/v1/systemone", 404, "404 page not found"
            ),
        ):
            with self.assertRaises(DecisionError) as ctx:
                client.decide("state", QUESTIONS)
        self.assertIn("0.35", str(ctx.exception))

    def test_404_naming_the_model_suggests_pull(self):
        client = SystemOneClient(model="nimble")
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=http_error(
                "http://localhost:11434/v1/systemone",
                404,
                "model 'nimble' not found",
            ),
        ):
            with self.assertRaises(DecisionError) as ctx:
                client.decide("state", QUESTIONS)
        self.assertIn("ollama pull nimble", str(ctx.exception))

    def test_malformed_json_raises(self):
        class BadBody(FakeResponse):
            def read(self):
                return b"this is not json {"

        client = SystemOneClient()
        with mock.patch("urllib.request.urlopen", return_value=BadBody({})):
            with self.assertRaises(DecisionError) as ctx:
                client.decide("state", QUESTIONS)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_missing_answers_raises(self):
        client = SystemOneClient()
        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeResponse({"model": "nimble"}),
        ):
            with self.assertRaises(DecisionError) as ctx:
                client.decide("state", QUESTIONS)
        self.assertIn("answers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
