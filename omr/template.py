"""Compile the vector geometry of an OMR sheet PDF into a reusable template.

The sheet used by this project stores each bubble as a stroked circle path in
the PDF.  That is exact geometry, not an estimate from a raster: 120 circles
on a 182 x 257 mm page.  We extract the circle centres, cluster them into
rows/columns, and assign question/option numbers.

If a PDF does not expose circles as separate paths, the caller can fall back
to a raster/Hough-based calibration (not part of M0).

Template coordinates are stored in millimetres.  Pixels are derived only at
sampling time, so the same template works at 150, 200 or 300 DPI.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PT_TO_MM = 25.4 / 72.0
OPTION_LABELS = ["ক", "খ", "গ", "ঘ", "ঙ"]


@dataclass(frozen=True)
class Bubble:
    q: int
    option: int
    label: str
    x_mm: float
    y_mm: float
    r_mm: float
    x_pt: float = 0.0
    y_pt: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "x_mm": round(self.x_mm, 4),
            "y_mm": round(self.y_mm, 4),
            "r_mm": round(self.r_mm, 4),
            "x_pt": round(self.x_pt, 4),
            "y_pt": round(self.y_pt, 4),
        }


@dataclass
class Template:
    template_id: str
    page_width_mm: float
    page_height_mm: float
    page_width_pt: float
    page_height_pt: float
    columns: int
    options: int
    bands: int
    bubbles: list[Bubble]
    canonical_dpi: int = 200
    source_pdf: str = ""
    source_sha256: str = ""
    regions: dict = field(default_factory=dict)
    schema: str = "latticeomr/template/1"
    frame: dict | None = None

    # ------------------------------------------------------------------ basics
    @property
    def questions(self) -> int:
        return len({b.q for b in self.bubbles})

    @property
    def page_size_px(self) -> tuple[int, int]:
        return (
            int(round(self.page_width_mm / 25.4 * self.canonical_dpi)),
            int(round(self.page_height_mm / 25.4 * self.canonical_dpi)),
        )

    def bubble_px(self, b: Bubble, dpi: int | None = None) -> tuple[float, float, float]:
        dpi = dpi or self.canonical_dpi
        return (
            b.x_mm / 25.4 * dpi,
            b.y_mm / 25.4 * dpi,
            b.r_mm / 25.4 * dpi,
        )

    def frame_corners_mm(self) -> list[tuple[float, float]] | None:
        """Outer alignment frame corners as TL, TR, BR, BL (millimetres)."""
        if not self.frame:
            return None
        x0 = float(self.frame["x0_mm"])
        y0 = float(self.frame["y0_mm"])
        x1 = float(self.frame["x1_mm"])
        y1 = float(self.frame["y1_mm"])
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def frame_corners_px(self, dpi: int | None = None) -> list[tuple[float, float]] | None:
        """Outer alignment frame corners in canonical raster pixels."""
        corners = self.frame_corners_mm()
        if corners is None:
            return None
        dpi = dpi or self.canonical_dpi
        return [(x / 25.4 * dpi, y / 25.4 * dpi) for x, y in corners]

    def bubbles_by_question(self) -> dict[int, list[Bubble]]:
        out: dict[int, list[Bubble]] = {}
        for b in self.bubbles:
            out.setdefault(b.q, []).append(b)
        for q in out:
            out[q].sort(key=lambda b: b.option)
        return dict(sorted(out.items()))

    # ------------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        by_q = self.bubbles_by_question()
        questions = []
        for q, opts in by_q.items():
            questions.append(
                {
                    "q": q,
                    "band": (q - 1) // self.columns,
                    "column": (q - 1) % self.columns,
                    "options": [b.to_dict() for b in opts],
                }
            )
        return {
            "schema": self.schema,
            "template_id": self.template_id,
            "source_pdf": self.source_pdf,
            "source_sha256": self.source_sha256,
            "page": {
                "width_pt": round(self.page_width_pt, 4),
                "height_pt": round(self.page_height_pt, 4),
                "width_mm": round(self.page_width_mm, 4),
                "height_mm": round(self.page_height_mm, 4),
            },
            "canonical_dpi": self.canonical_dpi,
            "bubble": {
                "diameter_mm": round(2 * self.bubbles[0].r_mm, 4),
                "ring_stroke_rgb255": [170, 105, 123],
            },
            "grid": {
                "columns": self.columns,
                "options": self.options,
                "bands": self.bands,
                "questions": self.questions,
            },
            "questions": questions,
            "regions": self.regions,
            "frame": self.frame,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Template":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        page = data["page"]
        grid = data["grid"]
        bubbles: list[Bubble] = []
        for qd in data["questions"]:
            for opt_idx, bd in enumerate(qd["options"]):
                bubbles.append(
                    Bubble(
                        q=int(qd["q"]),
                        option=opt_idx,
                        label=bd.get("label", ""),
                        x_mm=float(bd["x_mm"]),
                        y_mm=float(bd["y_mm"]),
                        r_mm=float(bd["r_mm"]),
                        x_pt=float(bd.get("x_pt", 0.0)),
                        y_pt=float(bd.get("y_pt", 0.0)),
                    )
                )
        return cls(
            template_id=data["template_id"],
            page_width_mm=float(page["width_mm"]),
            page_height_mm=float(page["height_mm"]),
            page_width_pt=float(page["width_pt"]),
            page_height_pt=float(page["height_pt"]),
            columns=int(grid["columns"]),
            options=int(grid["options"]),
            bands=int(grid["bands"]),
            bubbles=bubbles,
            canonical_dpi=int(data.get("canonical_dpi", 200)),
            source_pdf=data.get("source_pdf", ""),
            source_sha256=data.get("source_sha256", ""),
            regions=data.get("regions", {}),
            schema=data.get("schema", "latticeomr/template/1"),
            frame=data.get("frame"),
        )


# --------------------------------------------------------------------- helpers
def _cluster_1d(values: Iterable[float], tol: float = 4.0) -> list[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return []
    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return text or "template"


def default_regions(width_mm: float, height_mm: float) -> dict:
    """Layout defaults for the Al-Quran Academy 30-question sheet.

    Only the total-marks box is used by M0.  These positions match the printed
    panel on the supplied PDF; a different layout can override them in JSON.
    """
    return {
        "student_id": {
            "x_mm": round(0.736 * width_mm, 3),
            "y_mm": round(0.148 * height_mm, 3),
            "w_mm": round(0.242 * width_mm, 3),
            "h_mm": round(0.078 * height_mm, 3),
        },
        "subject_code": {
            "x_mm": round(0.736 * width_mm, 3),
            "y_mm": round(0.257 * height_mm, 3),
            "w_mm": round(0.242 * width_mm, 3),
            "h_mm": round(0.070 * height_mm, 3),
        },
        "total_marks": {
            "x_mm": round(0.736 * width_mm, 3),
            "y_mm": round(0.366 * height_mm, 3),
            "w_mm": round(0.242 * width_mm, 3),
            "h_mm": round(0.117 * height_mm, 3),
        },
    }


def _extract_frame(page, width_pt: float, height_pt: float) -> dict | None:
    """Find the large stroked, white-filled alignment frame on the page.

    The supplied sheet draws a thin pink rectangle around the answer grid and
    the marks panel.  Mapping its four corners is far more reliable than the
    outer page edge, which is often white-on-white in renders and scans.
    """
    best: tuple[float, object] | None = None
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if drawing.get("color") is None:
            continue
        fill = drawing.get("fill")
        if fill is None or min(fill) < 0.95:
            continue
        if rect.width < 0.50 * width_pt or rect.height < 0.20 * height_pt:
            continue
        if rect.width >= 0.999 * width_pt or rect.height >= 0.999 * height_pt:
            continue
        area = rect.width * rect.height
        if best is None or area > best[0]:
            best = (area, rect)
    if best is None:
        return None
    rect = best[1]
    return {
        "x0_mm": round(rect.x0 * PT_TO_MM, 4),
        "y0_mm": round(rect.y0 * PT_TO_MM, 4),
        "x1_mm": round(rect.x1 * PT_TO_MM, 4),
        "y1_mm": round(rect.y1 * PT_TO_MM, 4),
    }


def compile_template(
    pdf_path: str | Path,
    template_id: str | None = None,
    dpi: int = 200,
    render_reference: bool = True,
    reference_dir: str | Path | None = None,
) -> Template:
    """Extract bubble geometry from a vector OMR sheet PDF.

    pymupdf is imported here rather than at module level so that reading a
    sheet only needs OpenCV and an existing template JSON.  Some managed
    Windows environments block pymupdf's native DLL, which must not stop the
    live scanner from running.
    """
    try:
        import pymupdf
    except Exception as exc:  # DLL policy blocks, missing package, etc.
        raise RuntimeError(
            "pymupdf is needed only to compile a template from a PDF, not to "
            "read a sheet. Use an existing templates/*.json, or install a "
            "working pymupdf to compile a new template."
        ) from exc

    pdf_path = Path(pdf_path)
    raw = pdf_path.read_bytes()
    doc = pymupdf.open(pdf_path)
    if not doc:
        raise ValueError(f"could not open PDF: {pdf_path}")
    page = doc[0]

    circles: list[tuple[float, float, float]] = []  # cx, cy, r (pt)
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if drawing.get("color") is None:
            continue
        w, h = rect.width, rect.height
        if not (16.0 < w < 18.0 and 16.0 < h < 18.0):
            continue
        if abs(w - h) > 1.0:
            continue
        circles.append(((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2, (w + h) / 4))

    if not circles:
        raise ValueError(
            "no bubble circles found in PDF. This compiler expects stroked "
            "17x17 pt circle paths; use raster calibration for other PDFs."
        )

    xs = _cluster_1d([c[0] for c in circles], tol=4.0)
    ys = _cluster_1d([c[1] for c in circles], tol=4.0)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(f"could not form a grid: {len(xs)} columns x {len(ys)} rows")

    # The largest vertical gap separates the two answer bands (e.g. 15+15).
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    split = max(range(len(gaps)), key=lambda i: gaps[i]) + 1
    band_rows = [ys[:split], ys[split:]]
    bands = [b for b in band_rows if b]
    if len(bands) != 2:
        raise ValueError(
            f"expected two answer bands, found {len(bands)}; "
            "this compiler is for the two-band sheet layout"
        )
    options_per_band = len(bands[0])
    if len(bands[1]) != options_per_band:
        raise ValueError("answer bands have different option counts")
    if options_per_band > len(OPTION_LABELS):
        raise ValueError(f"unsupported option count: {options_per_band}")

    radius_pt = sum(c[2] for c in circles) / len(circles)
    radius_mm = radius_pt * PT_TO_MM
    columns = len(xs)
    width_mm = page.rect.width * PT_TO_MM
    height_mm = page.rect.height * PT_TO_MM

    bubbles: list[Bubble] = []
    for band_idx, band_ys in enumerate(bands):
        for col_idx, x_pt in enumerate(xs):
            q = band_idx * columns + col_idx + 1
            for opt_idx, y_pt in enumerate(band_ys):
                bubbles.append(
                    Bubble(
                        q=q,
                        option=opt_idx,
                        label=OPTION_LABELS[opt_idx],
                        x_mm=x_pt * PT_TO_MM,
                        y_mm=y_pt * PT_TO_MM,
                        r_mm=radius_mm,
                        x_pt=x_pt,
                        y_pt=y_pt,
                    )
                )

    template = Template(
        template_id=template_id or _slug(pdf_path.stem),
        page_width_mm=width_mm,
        page_height_mm=height_mm,
        page_width_pt=page.rect.width,
        page_height_pt=page.rect.height,
        columns=columns,
        options=options_per_band,
        bands=2,
        bubbles=bubbles,
        canonical_dpi=dpi,
        source_pdf=pdf_path.name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        regions=default_regions(width_mm, height_mm),
        frame=_extract_frame(page, page.rect.width, page.rect.height),
    )

    if template.questions != columns * 2:
        raise ValueError(
            f"grid produced {template.questions} questions, expected {columns * 2}"
        )

    if render_reference:
        ref_dir = Path(reference_dir) if reference_dir else pdf_path.parent
        ref_dir.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        pix.save(str(ref_dir / f"{template.template_id}_reference.png"))

    return template
