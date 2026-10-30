"""Render a graded canonical sheet.

Colour language (matches the supplied result image):
  green  = the student's selected answer is correct
  red    = the student's selected answer is wrong
  blue   = the key answer when the student was wrong
  orange = blank / unobserved question marker
  yellow = review marker
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .grade import GradeResult
from .reader import ReadResult
from .template import Template

GREEN = (40, 180, 40)
RED = (30, 30, 230)
BLUE = (230, 120, 20)
ORANGE = (0, 140, 255)
YELLOW = (0, 220, 220)
PURPLE = (180, 40, 180)
BLACK = (30, 30, 30)


def _region_px(template: Template, region: dict) -> tuple[int, int, int, int]:
    dpi = template.canonical_dpi
    x = int(round(region["x_mm"] / 25.4 * dpi))
    y = int(round(region["y_mm"] / 25.4 * dpi))
    w = int(round(region["w_mm"] / 25.4 * dpi))
    h = int(round(region["h_mm"] / 25.4 * dpi))
    return x, y, w, h


def _draw_question_marker(img, template, q: int, color: tuple[int, int, int], text: str = "") -> None:
    by_q = template.bubbles_by_question()
    opts = by_q[q]
    cx, cy, r = template.bubble_px(opts[0])
    x = int(cx - r * 1.9)
    y = int(cy - r * 1.7)
    cv2.circle(img, (x, y), max(4, int(r * 0.32)), color, -1, cv2.LINE_AA)
    if text:
        cv2.putText(
            img,
            text,
            (x + int(r * 0.55), y + int(r * 0.25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, r / 45.0),
            color,
            1,
            cv2.LINE_AA,
        )


def render_result(
    warped_bgr: np.ndarray,
    template: Template,
    read: ReadResult,
    grade: GradeResult,
    output_path: str | Path,
    audit_path: str | Path | None = None,
) -> Path:
    img = warped_bgr.copy()
    by_q = template.bubbles_by_question()

    for r in grade.questions:
        opts = by_q[r.q]
        if r.outcome == "CORRECT":
            b = opts[r.selected[0]]
            cx, cy, rad = template.bubble_px(b)
            cv2.circle(img, (int(cx), int(cy)), int(rad * 1.16), GREEN, 3, cv2.LINE_AA)
        elif r.outcome == "WRONG":
            for opt in r.selected:
                b = opts[opt]
                cx, cy, rad = template.bubble_px(b)
                cv2.circle(img, (int(cx), int(cy)), int(rad * 1.16), RED, 3, cv2.LINE_AA)
            if r.key_answer is not None and r.key_answer not in r.selected:
                b = opts[r.key_answer]
                cx, cy, rad = template.bubble_px(b)
                cv2.circle(img, (int(cx), int(cy)), int(rad * 1.16), BLUE, 3, cv2.LINE_AA)
        elif r.outcome == "MULTI":
            for opt in r.selected:
                b = opts[opt]
                cx, cy, rad = template.bubble_px(b)
                cv2.circle(img, (int(cx), int(cy)), int(rad * 1.16), PURPLE, 3, cv2.LINE_AA)
            if r.key_answer is not None and r.key_answer not in r.selected:
                b = opts[r.key_answer]
                cx, cy, rad = template.bubble_px(b)
                cv2.circle(img, (int(cx), int(cy)), int(rad * 1.16), BLUE, 2, cv2.LINE_AA)
            _draw_question_marker(img, template, r.q, PURPLE, "M")
        elif r.outcome == "BLANK":
            _draw_question_marker(img, template, r.q, ORANGE, "?")
        else:
            _draw_question_marker(img, template, r.q, YELLOW, "!")

    # Total mark inside the printed মোট নম্বর panel.
    if "total_marks" in template.regions:
        x, y, w, h = _region_px(template, template.regions["total_marks"])
        text = str(int(round(grade.total))) if float(grade.total).is_integer() else f"{grade.total:.1f}"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(1.0, h / 85.0)
        thickness = max(2, int(round(scale * 1.8)))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, text, (tx, ty), font, scale, RED, thickness, cv2.LINE_AA)

    # Small footer with the outcome counts; keeps the printed sheet readable.
    line = f"Correct {grade.correct}   Wrong {grade.wrong}   Blank {grade.blank}   Review {grade.review}"
    cv2.putText(
        img,
        line,
        (int(0.03 * img.shape[1]), int(0.985 * img.shape[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLACK,
        1,
        cv2.LINE_AA,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(audit_path), warped_bgr)
    return output_path
