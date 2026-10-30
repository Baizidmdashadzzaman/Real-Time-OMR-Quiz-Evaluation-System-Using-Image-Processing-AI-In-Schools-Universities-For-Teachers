"""Grade a read sheet against a teacher key."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .reader import QuestionReading, ReadResult
from .template import Template


@dataclass
class QuestionResult:
    q: int
    selected: list[int]
    key_answer: int | None
    outcome: str  # CORRECT | WRONG | BLANK | MULTI | REVIEW
    marks: float
    awarded: float
    confidence: float
    reason: str = ""


@dataclass
class GradeResult:
    questions: list[QuestionResult]
    total: float
    maximum: float
    correct: int
    wrong: int
    blank: int
    multi: int
    review: int
    review_questions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "maximum": round(self.maximum, 4),
            "percentage": round(100.0 * self.total / self.maximum, 2) if self.maximum else 0.0,
            "correct": self.correct,
            "wrong": self.wrong,
            "blank": self.blank,
            "multi": self.multi,
            "review": self.review,
            "review_questions": self.review_questions,
            "questions": [
                {
                    "q": r.q,
                    "selected": r.selected,
                    "key_answer": r.key_answer,
                    "outcome": r.outcome,
                    "marks": r.marks,
                    "awarded": round(r.awarded, 4),
                    "confidence": round(r.confidence, 4),
                    "reason": r.reason,
                }
                for r in self.questions
            ],
        }


def load_key(path: str | Path, template: Template, strict_template: bool = True) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    key_template = data.get("template_id")
    if strict_template and key_template and key_template != template.template_id:
        raise ValueError(
            f"key template_id {key_template!r} does not match template "
            f"{template.template_id!r}"
        )
    return data


def grade(read: ReadResult, key: dict, template: Template) -> GradeResult:
    qkey = key.get("questions", {})
    policy = key.get("policy", {})
    negative = float(policy.get("negative_per_wrong", 0.0))
    multi_policy = policy.get("multi", "wrong")
    expected = set(range(1, template.questions + 1))
    missing = expected - {int(q) for q in qkey}
    if missing:
        raise ValueError(f"key is missing questions: {sorted(missing)}")

    results: list[QuestionResult] = []
    for q in range(1, template.questions + 1):
        entry = qkey[str(q)]
        answer = int(entry["answer"])
        marks = float(entry.get("marks", 1))
        qr: QuestionReading = read.questions[q]

        if qr.state == "ANSWERED":
            if qr.selected == [answer]:
                outcome, awarded = "CORRECT", marks
            else:
                outcome, awarded = "WRONG", (-negative * marks if negative > 0 else 0.0)
        elif qr.state == "BLANK":
            outcome, awarded = "BLANK", 0.0
        elif qr.state == "MULTI":
            if multi_policy == "partial":
                outcome = "MULTI"
                awarded = (
                    marks / max(len(qr.selected), 1)
                    if answer in qr.selected
                    else (-negative * marks if negative > 0 else 0.0)
                )
            elif multi_policy == "review":
                outcome, awarded = "REVIEW", 0.0
            else:
                outcome, awarded = "MULTI", (-negative * marks if negative > 0 else 0.0)
        else:
            outcome, awarded = "REVIEW", 0.0

        results.append(
            QuestionResult(
                q=q,
                selected=qr.selected,
                key_answer=answer,
                outcome=outcome,
                marks=marks,
                awarded=awarded,
                confidence=qr.confidence,
                reason=qr.reason or qr.state,
            )
        )

    correct = sum(r.outcome == "CORRECT" for r in results)
    wrong = sum(r.outcome == "WRONG" for r in results)
    blank = sum(r.outcome == "BLANK" for r in results)
    multi = sum(r.outcome == "MULTI" for r in results)
    review = sum(r.outcome == "REVIEW" for r in results)
    total = sum(r.awarded for r in results)
    maximum = sum(r.marks for r in results)
    return GradeResult(
        questions=results,
        total=total,
        maximum=maximum,
        correct=correct,
        wrong=wrong,
        blank=blank,
        multi=multi,
        review=review,
        review_questions=[r.q for r in results if r.outcome == "REVIEW"],
    )
