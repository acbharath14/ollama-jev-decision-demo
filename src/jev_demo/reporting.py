"""Human-readable rendering of decision answers.

Decision models return numbers, not prose. These helpers turn those
numbers into one honest line per answer, so a demo run reads like a
report instead of a JSON dump:

- choice: the winning option, every option's probability, confidence.
- yes/no: yes or no (probability >= 0.5 means yes) plus the probability.
- score:  the expected position on the scale, with the scale labels.
"""

from .client import (
    Answer,
    BooleanAnswer,
    ChoiceAnswer,
    DecideResult,
    ScoreAnswer,
)


def _pct(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def format_answer(answer: Answer) -> str:
    """One line for one typed answer."""
    if isinstance(answer, ChoiceAnswer):
        probs = ", ".join(
            f"{option} {_pct(p)}" for option, p in answer.probabilities.items()
        )
        line = f"{answer.name}: {answer.choice} (probabilities: {probs})"
        if answer.confidence is not None:
            line += f"; confidence {answer.confidence:.3f}"
        return line
    if isinstance(answer, BooleanAnswer):
        verdict = "yes" if answer.yes else "no"
        return f"{answer.name}: {verdict} (probability {_pct(answer.probability)})"
    if isinstance(answer, ScoreAnswer):
        labels = [answer.legend.get(str(i), f"level {i}") for i in range(len(answer.legend))] \
            if answer.legend else []
        scale = " < ".join(labels) if labels else "ordered scale"
        line = f"{answer.name}: {answer.score:.2f} on [{scale}]"
        if answer.probabilities:
            probs = ", ".join(
                f"{answer.legend.get(k, k)} {_pct(p)}"
                for k, p in answer.probabilities.items()
            )
            line += f" (probabilities: {probs})"
        if answer.confidence is not None:
            line += f"; confidence {answer.confidence:.3f}"
        return line
    raise TypeError(f"Unknown answer type: {type(answer)!r}")


def format_result(result: DecideResult, title: str = "") -> str:
    """All answers of one call, plus timing and token usage."""
    lines = []
    if title:
        lines.append(title)
    for name in result.answers:
        lines.append("  " + format_answer(result.answers[name]))
    lines.append(f"  [{result.model}, {result.elapsed_ms:.0f} ms")
    if result.usage:
        tokens = ", ".join(f"{k}={v}" for k, v in result.usage.items())
        lines[-1] += f", usage: {tokens}"
    lines[-1] += "]"
    return "\n".join(lines)
