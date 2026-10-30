"""Sheet localisation and canonical perspective warping.

M0 uses a page-quad detector with a full-frame fallback for clean scans.
Fiducial/QR modes and lattice refinement are later milestones; the interfaces
already return the homography and the source corners those modes will replace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .template import Template


@dataclass
class WarpResult:
    warped: np.ndarray
    homography: np.ndarray
    corners: np.ndarray  # 4x2, original image coordinates, ordered TL/TR/BR/BL
    mode: str  # "page_quad" | "manual" | "full_frame"
    page_confidence: float
    debug: dict


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]  # x - y; np.diff would give y - x
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # TL
    ordered[2] = pts[np.argmax(s)]  # BR
    ordered[1] = pts[np.argmax(d)]  # TR
    ordered[3] = pts[np.argmin(d)]  # BL
    return ordered


def _edge_support(edges: np.ndarray, quad: np.ndarray, samples: int = 40) -> float:
    """Fraction of sampled side points that sit on an edge pixel."""
    h, w = edges.shape[:2]
    hits = 0
    total = 0
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        for t in np.linspace(0.05, 0.95, samples):
            x = int(round(p0[0] + (p1[0] - p0[0]) * t))
            y = int(round(p0[1] + (p1[1] - p0[1]) * t))
            total += 1
            if 0 <= x < w and 0 <= y < h:
                y0, y1 = max(0, y - 2), min(h, y + 3)
                x0, x1 = max(0, x - 2), min(w, x + 3)
                if edges[y0:y1, x0:x1].any():
                    hits += 1
    return hits / max(total, 1)


def _pink_mask(image: np.ndarray) -> np.ndarray:
    """Printed pink ink, robust to black/blue pen marks."""
    b, g, r = cv2.split(image)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    # Magenta print has B >= G; warm indoor white has G > B, so the last term
    # keeps a warm-lit white page from being mistaken for the pink frame.
    mask = (
        (r16 - g16 > 18)
        & (r16 - b16 > 6)
        & (b16 - g16 > -2)
        & (r > 130)
    ).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_frame_quad(
    image: np.ndarray, template: Template
) -> tuple[np.ndarray | None, float]:
    """Find the printed alignment frame around the answer area.

    The frame is a thin pink rectangle; its contour is the largest pink
    quadrilateral in the image.  Unlike the page edge it is visible on white
    backgrounds, survives cropping outside the frame, and does not depend on
    the sheet being page-shaped in the frame.
    """
    corners_mm = template.frame_corners_mm()
    if corners_mm is None:
        return None, 0.0
    frame_mm = np.asarray(corners_mm, dtype=np.float32)
    fw = float(np.linalg.norm(frame_mm[1] - frame_mm[0]))
    fh = float(np.linalg.norm(frame_mm[3] - frame_mm[0]))
    if fw <= 1 or fh <= 1:
        return None, 0.0
    target_aspect = min(fw, fh) / max(fw, fh)

    h, w = image.shape[:2]
    pink = _pink_mask(image)
    support_mask = cv2.dilate(pink, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(pink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(h * w)
    best: tuple[float, np.ndarray, float, float, float] | None = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.02 * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            (rw, rh) = cv2.minAreaRect(quad)[1]
            if rw <= 1 or rh <= 1:
                continue
            aspect = min(rw, rh) / max(rw, rh)
            aspect_err = abs(aspect - target_aspect)
            if aspect_err > 0.08:
                continue
            support = _edge_support(support_mask, quad, samples=60)
            if support < 0.55:
                continue
            score = area * (1.0 - aspect_err / 0.08) * (0.5 + 0.5 * support)
            if best is None or score > best[0]:
                best = (score, quad, area, support, aspect_err)
            break

    if best is None:
        return None, 0.0
    _, quad, area, support, _ = best
    area_frac = area / image_area
    confidence = 0.60 + 0.30 * min(1.0, area_frac / 0.10) + 0.10 * support
    return order_quad(quad), float(np.clip(confidence, 0.0, 0.99))


def _panel_centroid(
    pink: np.ndarray, frame_quad: np.ndarray, image_shape: tuple[int, ...]
) -> np.ndarray | None:
    """Centroid of the large solid pink boxes in the marks panel."""
    h, w = image_shape[:2]
    frame_area = abs(float(cv2.contourArea(frame_quad)))
    min_area = max(200.0, 0.003 * frame_area)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (pink > 0).astype(np.uint8), 8
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for i in range(1, count):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        if ww > 0.30 * w or hh > 0.30 * h:
            continue
        if area / max(ww * hh, 1) < 0.10:
            continue
        candidates.append((float(area), centroids[i]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return np.mean([c for _, c in candidates[:3]], axis=0)


def _choose_frame_orientation(
    frame_quad: np.ndarray,
    frame_dst: np.ndarray,
    panel: np.ndarray,
    template: Template,
    dpi: int,
) -> tuple[int, float]:
    """Pick the cyclic corner assignment that puts the marks panel on the right."""
    centers = [
        (r["x_mm"] + r["w_mm"] / 2.0, r["y_mm"] + r["h_mm"] / 2.0)
        for r in template.regions.values()
        if all(k in r for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
    ]
    if not centers:
        return 0, 999.0
    expected = np.mean(np.asarray(centers, dtype=np.float32), axis=0)
    best: tuple[float, int] | None = None
    for k in range(4):
        src = np.roll(frame_quad, k, axis=0).astype(np.float32)
        homography = cv2.getPerspectiveTransform(src, frame_dst.astype(np.float32))
        mapped = cv2.perspectiveTransform(
            panel.reshape(1, 1, 2).astype(np.float32), homography
        )[0, 0]
        mm = mapped * 25.4 / dpi
        distance = float(np.linalg.norm((mm - expected) / np.array([20.0, 20.0])))
        if best is None or distance < best[0]:
            best = (distance, k)
    assert best is not None
    return best[1], best[0]


def find_page_quad(image: np.ndarray, template: Template) -> tuple[np.ndarray | None, float]:
    """Find the largest convex quadrilateral matching the template page aspect.

    Returns corners in the *original* image coordinate system and a 0..1
    confidence.  `None` means no trustworthy page boundary was found.
    """
    h, w = image.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    target_aspect = min(template.page_width_mm, template.page_height_mm) / max(
        template.page_width_mm, template.page_height_mm
    )
    image_area = float(small.shape[0] * small.shape[1])
    best: tuple[float, np.ndarray, float] | None = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.04 * image_area or area > 0.98 * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        (rw, rh) = cv2.minAreaRect(quad)[1]
        if rw <= 1 or rh <= 1:
            continue
        aspect = min(rw, rh) / max(rw, rh)
        aspect_err = abs(aspect - target_aspect)
        if aspect_err > 0.10:
            continue
        rect_area = rw * rh
        rectangularity = min(1.0, area / max(rect_area, 1.0))
        if rectangularity < 0.70:
            continue
        support = _edge_support(edges, quad)
        if support < 0.35:
            continue
        score = (
            0.45 * min(1.0, area / (0.70 * image_area))
            + 0.30 * rectangularity
            + 0.15 * (1.0 - aspect_err / 0.10)
            + 0.10 * support
        )
        if best is None or score > best[0]:
            best = (score, quad, support)

    if best is None:
        return None, 0.0

    score, quad, support = best
    quad = quad / scale
    return order_quad(quad), float(np.clip(score, 0.0, 1.0))


def looks_like_flat_scan(image: np.ndarray, template: Template) -> bool:
    """True when the sheet already fills a white-bordered, page-shaped frame.

    A flat scan has no contrasting page boundary to detect, and running quad
    detection on it can latch onto the answer grid.  A camera photo on a dark
    mat has a dark border and fails this test, so it goes through detection.
    """
    h, w = image.shape[:2]
    if h < 20 or w < 20:
        return False
    target = min(template.page_width_mm, template.page_height_mm) / max(
        template.page_width_mm, template.page_height_mm
    )
    aspect = min(w, h) / max(w, h)
    if abs(math.log(max(aspect, 1e-6) / target)) > 0.06:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    band = max(2, int(round(min(h, w) * 0.02)))
    border = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[-band:, :].ravel(),
            gray[:, :band].ravel(),
            gray[:, -band:].ravel(),
        ]
    )
    return float(np.median(border)) >= 235.0


def warp_to_template(
    image: np.ndarray,
    template: Template,
    corners: np.ndarray | None = None,
    detect_page: bool = True,
    force_detect: bool = False,
    dpi: int | None = None,
) -> WarpResult:
    """Map the sheet to the canonical template raster."""
    dpi = dpi or template.canonical_dpi
    h, w = image.shape[:2]
    debug: dict = {}

    out_w, out_h = template.page_size_px
    page_dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    dst = page_dst

    if corners is not None:
        src = order_quad(np.asarray(corners, dtype=np.float32))
        mode = "manual"
        confidence = 1.0
    else:
        src = None
        confidence = 0.0
        mode = "full_frame"

        # The printed frame is the preferred target: it works on white-on-white
        # renders, survives cropping outside the frame, and gives an exact
        # four-point correspondence to the template.  A page-shaped flat scan
        # is already aligned, and its anti-aliased frame can bias the quad by a
        # pixel or two, so keep the full-frame path for that case.
        flat_scan = detect_page and looks_like_flat_scan(image, template)
        if detect_page and template.frame is not None and (force_detect or not flat_scan):
            frame_src, frame_conf = find_frame_quad(image, template)
            if frame_src is not None:
                frame_dst = np.asarray(template.frame_corners_px(dpi), dtype=np.float32)
                pink = _pink_mask(image)
                panel = _panel_centroid(pink, frame_src, image.shape)
                orientation = 0
                if panel is not None:
                    orientation, panel_distance = _choose_frame_orientation(
                        frame_src, frame_dst, panel, template, dpi
                    )
                    debug["frame_orientation"] = orientation
                    debug["panel_distance"] = round(panel_distance, 3)
                    if panel_distance >= 1.5:
                        frame_conf = min(frame_conf, 0.75)
                src = np.roll(frame_src, orientation, axis=0).astype(np.float32)
                dst = frame_dst
                mode = "frame_quad"
                confidence = frame_conf

        if src is None and detect_page and (force_detect or not flat_scan):
            src, confidence = find_page_quad(image, template)
            if src is not None:
                mode = "page_quad"
                dst = page_dst
        elif src is None and detect_page:
            confidence = 0.90

    if src is None:
        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        dst = page_dst
        mode = "full_frame"
        confidence = 0.0
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        homography,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    debug["source_size"] = (w, h)
    debug["warped_size"] = (out_w, out_h)
    return WarpResult(
        warped=warped,
        homography=homography,
        corners=src,
        mode=mode,
        page_confidence=confidence,
        debug=debug,
    )
