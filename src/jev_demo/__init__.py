"""Stdlib-only client for Ollama's Jev-style decision models.

See client.py for the /v1/systemone API shape and questions.py for
the question builders. Nothing here needs any third-party package.
"""

from .client import (
    BooleanAnswer,
    ChoiceAnswer,
    DecideResult,
    DecisionError,
    ScoreAnswer,
    SystemOneClient,
)
from .questions import boolean_question, choice_question, score_question

__all__ = [
    "BooleanAnswer",
    "ChoiceAnswer",
    "DecideResult",
    "DecisionError",
    "ScoreAnswer",
    "SystemOneClient",
    "boolean_question",
    "choice_question",
    "score_question",
]
