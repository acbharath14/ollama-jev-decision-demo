"""Tests for the question builders and the answer reporting.

Builders are pure functions, so these tests need no server at all —
mocked or otherwise.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_demo.client import BooleanAnswer, ChoiceAnswer, ScoreAnswer
from jev_demo.questions import (
    boolean_question,
    choice_question,
    score_question,
)
from jev_demo.reporting import format_answer


class QuestionBuilderTests(unittest.TestCase):
    def test_choice_builder_shape(self):
        question = choice_question(
            "Which label fits?",
            {"billing": "Payments", "bug": None},
        )
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["instructions"], "Which label fits?")
        self.assertEqual(
            question["criteria"], {"billing": "Payments", "bug": None}
        )

    def test_boolean_builder_uses_noul_on_the_wire(self):
        # The Jev API spells the yes/no type "noul", not "boolean".
        # Callers write boolean_question(); the wire format stays correct.
        question = boolean_question("Does the customer ask for a refund?")
        self.assertEqual(question["type"], "noul")
        self.assertNotIn("criteria", question)

    def test_score_builder_criteria_is_a_list(self):
        question = score_question(
            "How urgent?", ["Routine", "Soon", "Urgent"]
        )
        self.assertEqual(question["type"], "score")
        self.assertEqual(question["criteria"], ["Routine", "Soon", "Urgent"])

    def test_empty_instructions_rejected(self):
        for builder, args in [
            (choice_question, ("  ", {"a": "A"})),
            (boolean_question, ("",)),
            (score_question, ("", ["Low", "High"])),
        ]:
            with self.assertRaises(ValueError):
                builder(*args)

    def test_empty_choice_criteria_rejected(self):
        with self.assertRaises(ValueError):
            choice_question("Pick one.", {})

    def test_score_needs_at_least_two_levels(self):
        with self.assertRaises(ValueError):
            score_question("How bad?", ["Only"])
        with self.assertRaises(ValueError):
            score_question("How bad?", [])


class ReportingTests(unittest.TestCase):
    def test_choice_line(self):
        line = format_answer(
            ChoiceAnswer("label", "bug",
                         {"billing": 0.04, "bug": 0.91}, 0.66)
        )
        self.assertIn("label: bug", line)
        self.assertIn("bug 91.0%", line)
        self.assertIn("confidence 0.660", line)

    def test_boolean_line(self):
        self.assertIn("needs_human: yes",
                      format_answer(BooleanAnswer("needs_human", 0.84)))
        self.assertIn("needs_human: no",
                      format_answer(BooleanAnswer("needs_human", 0.2)))

    def test_score_line_uses_legend(self):
        line = format_answer(
            ScoreAnswer("severity", 2.37,
                        {"0": "cosmetic", "1": "minor", "2": "major",
                         "3": "full outage"},
                        {"2": 0.569, "3": 0.403}, 0.1)
        )
        self.assertIn("severity: 2.37", line)
        self.assertIn("cosmetic < minor < major < full outage", line)


if __name__ == "__main__":
    unittest.main()
