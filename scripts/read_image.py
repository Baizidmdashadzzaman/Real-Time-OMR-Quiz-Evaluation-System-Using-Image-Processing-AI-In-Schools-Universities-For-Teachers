#!/usr/bin/env python3
"""Read one still OMR sheet, grade it, and render an annotated result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omr.geometry import warp_to_template  # noqa: E402
from omr.grade import grade, load_key  # noqa: E402
from omr.overlay import render_result  # noqa: E402
from omr.reader import read_bubbles  # noqa: E402
from omr.template import Template  # noqa: E402


def parse_corners(text: str) -> np.ndarray:
    vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 8:
        raise argparse.ArgumentTypeError("corners need 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return np.array(vals, dtype=np.float32).reshape(4, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="still image of a filled sheet")
    ap.add_argument("--template", default=str(ROOT / "templates" / "q-paper.json"))
    ap.add_argument("--key", default=str(ROOT / "keys" / "sample_key.json"))
    ap.add_argument("--out", default=None, help="output directory or result.jpg path")
    ap.add_argument("--corners", type=parse_corners, default=None,
                    help="manual page corners x1,y1,...,x4,y4")
    ap.add_argument("--no-detect", action="store_true",
                    help="treat the image as a full-page scan (no page-quad detection)")
    ap.add_argument("--force-detect", action="store_true",
                    help="run page-quad detection even if the image looks like a flat scan")
    ap.add_argument("--dpi", type=int, default=None, help="override canonical DPI")
    ap.add_argument("--debug", action="store_true", help="save the warped canonical image")
    args = ap.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not read image {image_path}", file=sys.stderr)
        return 2
    if not Path(args.template).exists():
        print(f"error: template not found: {args.template}", file=sys.stderr)
        return 2
    if not Path(args.key).exists():
        print(f"error: key not found: {args.key}", file=sys.stderr)
        return 2

    template = Template.load(args.template)
    if args.dpi:
        template.canonical_dpi = args.dpi

    warp = warp_to_template(
        image,
        template,
        corners=args.corners,
        detect_page=not args.no_detect,
        force_detect=args.force_detect,
        dpi=template.canonical_dpi,
    )
    read = read_bubbles(warp.warped, template)
    key = load_key(args.key, template)
    result = grade(read, key, template)

    out = Path(args.out) if args.out else ROOT / "results" / image_path.stem
    if out.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        result_path = out
        out_dir = out.parent
    else:
        out_dir = out
        result_path = out_dir / "result.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_result(
        warp.warped,
        template,
        read,
        result,
        result_path,
        audit_path=out_dir / "audit_canonical.png",
    )

    payload = {
        "image": str(image_path),
        "template_id": template.template_id,
        "warp": {
            "mode": warp.mode,
            "page_confidence": round(warp.page_confidence, 4),
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in warp.corners],
            "debug": warp.debug,
        },
        "calibration": read.calibration,
        "diagnostics": read.diagnostics,
        "answers": {
            str(q): {
                "selected": r.selected,
                "state": r.state,
                "confidence": round(r.confidence, 4),
                "reason": r.reason,
            }
            for q, r in read.questions.items()
        },
        "grade": result.to_dict(),
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.debug:
        cv2.imwrite(str(out_dir / "debug_warped.png"), warp.warped)

    print(f"image: {image_path}")
    print(f"warp: {warp.mode} (confidence {warp.page_confidence:.2f}) -> {warp.warped.shape[1]}x{warp.warped.shape[0]}")
    print(f"calibration: {read.calibration.get('method')} sep={read.calibration.get('separation', 0):.2f} "
          f"t={read.calibration.get('threshold', 0):.3f}")
    print(f"total: {result.total:g}/{result.maximum:g}  correct={result.correct} wrong={result.wrong} "
          f"blank={result.blank} multi={result.multi} review={result.review}")
    for q in result.review_questions:
        print(f"  review Q{q}: {read.questions[q].reason or read.questions[q].state}")
    print(f"result: {result_path}")
    print(f"json:   {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
