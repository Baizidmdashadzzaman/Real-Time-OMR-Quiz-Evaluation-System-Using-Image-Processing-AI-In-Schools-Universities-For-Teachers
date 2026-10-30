#!/usr/bin/env python3
"""M0 acceptance checks.

Run from the project root:
    python scripts/selftest.py

Expected:
  flat sample          29/30, one wrong (Q1), full-frame mode
  synthetic tilted     29/30, page-quad mode
  blank sheet          0/30, all blank
  mixed sheet          Q1 answered, Q2 blank, Q3 multi, Q4 review, Q5 answered
  test-new captures    29/30, frame-quad mode, all 30 answered (four angles)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omr.geometry import warp_to_template  # noqa: E402
from omr.grade import grade, load_key  # noqa: E402
from omr.reader import read_bubbles  # noqa: E402
from omr.template import Template  # noqa: E402

TEMPLATE = ROOT / "templates" / "q-paper.json"
KEY = ROOT / "keys" / "sample_key.json"


def run(image: str, detect: bool = True, force: bool = False):
    template = Template.load(TEMPLATE)
    img = cv2.imread(str(ROOT / image))
    if img is None:
        raise FileNotFoundError(image)
    warp = warp_to_template(img, template, detect_page=detect, force_detect=force)
    read = read_bubbles(warp.warped, template)
    result = grade(read, load_key(KEY, template), template)
    return warp, read, result


def main() -> int:
    failures: list[str] = []

    warp, read, result = run("code try/q-paper_page-0001.jpg")
    print(f"flat:   mode={warp.mode} total={result.total:g} correct={result.correct} wrong={result.wrong}")
    if warp.mode != "full_frame":
        failures.append("flat scan should use full_frame")
    if (result.total, result.correct, result.wrong) != (29, 29, 1):
        failures.append(f"flat expected 29/29/1, got {result.total}/{result.correct}/{result.wrong}")
    if read.questions[1].selected != [0]:
        failures.append("flat Q1 should be option 0 (ক)")

    warp, read, result = run("samples/q-paper_tilted.jpg")
    print(f"tilted: mode={warp.mode} conf={warp.page_confidence:.2f} total={result.total:g} correct={result.correct}")
    if warp.mode != "page_quad":
        failures.append("tilted sample should use page_quad")
    if result.total != 29 or result.correct != 29:
        failures.append(f"tilted expected 29/29, got {result.total}/{result.correct}")

    warp, read, result = run("samples/q-paper_blank.png")
    marked = sum(b.state == "MARKED" for b in read.bubbles)
    print(f"blank:  total={result.total:g} blank={result.blank} marked_bubbles={marked}")
    if result.total != 0 or result.blank != 30 or marked != 0:
        failures.append(f"blank expected 0/30/0, got {result.total}/{result.blank}/{marked}")

    warp, read, result = run("samples/q-paper_mixed.png")
    states = {q: read.questions[q].state for q in range(1, 6)}
    print(f"mixed:  states={states} review={result.review_questions}")
    expected = {1: "ANSWERED", 2: "BLANK", 3: "MULTI", 4: "REVIEW", 5: "ANSWERED"}
    if states != expected:
        failures.append(f"mixed expected {expected}, got {states}")

    test_new = sorted((ROOT / "samples" / "test-new").glob("Asset*"))
    for path in test_new:
        warp, read, result = run(str(path.relative_to(ROOT)))
        answered = sum(q.state == "ANSWERED" for q in read.questions.values())
        print(
            f"{path.name:18s} mode={warp.mode} total={result.total:g} "
            f"correct={result.correct} wrong={result.wrong} answered={answered}"
        )
        if warp.mode != "frame_quad":
            failures.append(f"{path.name}: expected frame_quad, got {warp.mode}")
        if (result.correct, result.wrong, answered) != (29, 1, 30):
            failures.append(
                f"{path.name}: expected 29 correct, 1 wrong, 30 answered; "
                f"got {result.correct}, {result.wrong}, {answered}"
            )

    # Exercise all four in-plane orientations on the same base capture.
    orientation_base = cv2.imread(str(ROOT / "samples" / "test-new" / "Asset 1@4x.png"))
    if orientation_base is not None:
        template = Template.load(TEMPLATE)
        key = load_key(KEY, template)
        rotations = [
            ("0deg", None, 0),
            ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE, 1),
            ("180deg", cv2.ROTATE_180, 2),
            ("90cw", cv2.ROTATE_90_CLOCKWISE, 3),
        ]
        for label, code, expected_orientation in rotations:
            img = orientation_base if code is None else cv2.rotate(orientation_base, code)
            warp = warp_to_template(img, template)
            read = read_bubbles(warp.warped, template)
            result = grade(read, key, template)
            answered = sum(q.state == "ANSWERED" for q in read.questions.values())
            got_orientation = warp.debug.get("frame_orientation")
            print(
                f"orientation {label:7s} mode={warp.mode} k={got_orientation} "
                f"total={result.total:g} answered={answered}"
            )
            if warp.mode != "frame_quad" or got_orientation != expected_orientation:
                failures.append(
                    f"orientation {label}: expected frame_quad k={expected_orientation}, "
                    f"got {warp.mode} k={got_orientation}"
                )
            if (result.correct, result.wrong, answered) != (29, 1, 30):
                failures.append(
                    f"orientation {label}: expected 29/1/30, "
                    f"got {result.correct}/{result.wrong}/{answered}"
                )

    if failures:
        print("\nFAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("\nPASS: all M0 checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
