"""Builders for /v1/systemone question schemas.

A question is a small dict: a type, human-readable instructions, and
criteria. The builders below validate the shape up front so a typo
fails here with a clear message instead of as a server error later.

One naming gotcha, from the official Ollama example: the yes/no type
is spelled "noul" (TypeSafe's Jev API name), not "boolean".
boolean_question() builds a "noul" question, so callers can think in
plain words while the wire format stays correct.
"""

from typing import Any, Dict, List, Optional


def _check_instructions(instructions: str) -> str:
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    return instructions


def choice_question(
    instructions: str, criteria: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    """A pick-one question.

    criteria maps each option name to a short description (or None).
    Example: {"billing": "Payments and refunds", "bug": "Software errors"}.
    """
    _check_instructions(instructions)
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("choice criteria must be a non-empty object")
    if any(not name for name in criteria):
        raise ValueError("choice option names must be non-empty strings")
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": dict(criteria),
    }


def boolean_question(instructions: str) -> Dict[str, Any]:
    """A yes/no question. Builds a "noul" question on the wire.

    The answer comes back as a single probability: the probability
    that the answer is yes. See client.BooleanAnswer.
    """
    _check_instructions(instructions)
    return {"type": "noul", "instructions": instructions}


def score_question(
    instructions: str, levels: List[str]
) -> Dict[str, Any]:
    """An ordered-scale question.

    levels lists the scale from lowest to highest, e.g.
    ["Routine", "Soon", "Urgent"]. The answer's "score" is the expected
    position on that scale, and its "legend" maps positions to labels.
    """
    _check_instructions(instructions)
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("score levels must be a list of at least 2 labels")
    if any(not label for label in levels):
        raise ValueError("score level labels must be non-empty strings")
    return {
        "type": "score",
        "instructions": instructions,
        "criteria": list(levels),
    }
