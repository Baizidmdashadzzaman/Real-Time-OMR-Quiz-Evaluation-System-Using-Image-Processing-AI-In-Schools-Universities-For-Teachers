#!/usr/bin/env python3
"""Compile an OMR sheet PDF into template.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omr.template import compile_template  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="vector OMR sheet PDF (e.g. 'code try/q-paper.pdf')")
    ap.add_argument("--out", default=None, help="output template JSON")
    ap.add_argument("--id", default=None, help="template id (default: PDF stem)")
    ap.add_argument("--dpi", type=int, default=200, help="canonical reading DPI")
    ap.add_argument("--no-reference", action="store_true", help="skip reference PNG render")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out = Path(args.out) if args.out else ROOT / "templates" / f"{pdf.stem}.json"
    template = compile_template(
        pdf,
        template_id=args.id,
        dpi=args.dpi,
        render_reference=not args.no_reference,
        reference_dir=out.parent,
    )
    template.save(out)

    print(f"template: {template.template_id}")
    print(f"page: {template.page_width_mm:.1f} x {template.page_height_mm:.1f} mm")
    print(f"grid: {template.columns} columns x {template.options} options x {template.bands} bands")
    print(f"questions: {template.questions}, bubbles: {len(template.bubbles)}")
    print(f"saved: {out}")
    if not args.no_reference:
        print(f"reference: {out.parent / (template.template_id + '_reference.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
