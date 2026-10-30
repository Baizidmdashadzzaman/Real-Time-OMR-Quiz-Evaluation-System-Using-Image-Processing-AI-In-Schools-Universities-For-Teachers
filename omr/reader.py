"""Read bubble evidence from a canonical (warped) sheet.

M0 deliberately avoids a global pixel threshold.  It converts the sheet to a
paper-subtracted ink map, measures every bubble's interior and ring, then fits
a two-component model to *that sheet's* 120 scores.  The output has three
states: MARKED, EMPTY and UNCERTAIN, plus UNOBSERVED when glare covers a
bubble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from .template import Bubble, Template

MARKED = "MARKED"
EMPTY = "EMPTY"
UNCERTAIN = "UNCERTAIN"
UNOBSERVED = "UNOBSERVED"


@dataclass
class BubbleReading:
    q: int
    option: int
    label: str
    score: float
    p_marked: float
    state: str
    features: dict = field(default_factory=dict)


@dataclass
class QuestionReading:
    q: int
    selected: list[int]
    state: str  # ANSWERED | BLANK | MULTI | REVIEW | UNOBSERVED
    confidence: float
    reason: str = ""


@dataclass
class ReadResult:
    bubbles: list[BubbleReading]
    questions: dict[int, QuestionReading]
    calibration: dict
    diagnostics: dict = field(default_factory=dict)

    def selected_answers(self) -> dict[int, list[int]]:
        return {q: r.selected for q, r in self.questions.items()}


def _paper_subtracted_ink(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ink score (0..1), raw ink (0..255), and pink chroma."""
    # Pink print has a high minimum channel; black/blue ink has a low one.
    raw_ink = 255.0 - rgb.min(axis=2).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    paper = cv2.morphologyEx(raw_ink, cv2.MORPH_OPEN, kernel)
    paper = cv2.GaussianBlur(paper, (0, 0), sigmaX=9)
    ink = np.clip((raw_ink - paper) / np.maximum(255.0 - paper, 1.0), 0.0, 1.0)
    pink = rgb[..., 0] - (rgb[..., 1] + rgb[..., 2]) / 2.0
    return ink, raw_ink, pink


def _glare_mask(rgb: np.ndarray) -> np.ndarray:
    """Specular highlights are brighter than the local *paper*, not merely white.

    A small median window is dragged down by nearby black ink, so white paper
    around a filled bubble looks "locally brighter" and is wrongly flagged.
    Use a large morphological opening to estimate the paper level instead: it
    removes dark marks and is not affected by a bubble's fill.
    """
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    h, w = v.shape[:2]
    # A flatbed-style white scan has paper at 250-255; there is no specular
    # highlight to mask, and any local detector would flag the paper itself.
    if float(np.percentile(v, 90)) >= 250.0:
        return np.zeros((h, w), dtype=bool)
    # Estimate paper at 1/4 scale with a ~100 px effective median.  A single
    # 47 px filled bubble is well under half the window, so it cannot drag the
    # median down the way a 31 px local median did.
    small = cv2.resize(v, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 25)
    paper_v = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    mask = (
        (v >= 250)
        & (hsv[..., 1] < 20)
        & ((v.astype(np.int16) - paper_v) > 8)
    )
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _bubble_features(
    ink: np.ndarray,
    pink: np.ndarray,
    glare: np.ndarray,
    cx: float,
    cy: float,
    r: float,
) -> dict:
    pad = int(math.ceil(r * 1.7))
    x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1, y1 = min(ink.shape[1], int(cx) + pad + 1), min(ink.shape[0], int(cy) + pad + 1)
    patch = ink[y0:y1, x0:x1]
    ppatch = pink[y0:y1, x0:x1]
    gpatch = glare[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    inner = dist2 <= (0.52 * r) ** 2
    annulus = (dist2 >= (0.85 * r) ** 2) & (dist2 <= (1.15 * r) ** 2)
    if not inner.any() or not annulus.any():
        return {
            "f_fill": 0.0,
            "f_dark_frac": 0.0,
            "f_ring": 0.0,
            "f_contrast": 0.0,
            "f_pink": 0.0,
            "f_glare": 1.0,
            "f_uniform": 0.0,
            "observable": False,
        }

    inner_vals = patch[inner]
    ann_vals = patch[annulus]
    f_fill = float(np.median(inner_vals))
    f_dark_frac = float(np.mean(inner_vals > 0.35))
    f_ring = float(np.percentile(ann_vals, 75))
    f_pink = float(np.mean(ppatch[inner]))
    f_glare = float(np.mean(gpatch[inner]))
    f_uniform = float(
        np.clip(1.0 - np.std(inner_vals) / (np.mean(inner_vals) + 1e-6), 0.0, 1.0)
    )
    return {
        "f_fill": f_fill,
        "f_dark_frac": f_dark_frac,
        "f_ring": f_ring,
        "f_contrast": f_fill - f_ring,
        "f_pink": f_pink,
        "f_glare": f_glare,
        "f_uniform": f_uniform,
        "observable": f_glare <= 0.20,
    }


def _score(f: dict) -> float:
    return float(
        0.55 * f["f_fill"]
        + 0.25 * f["f_dark_frac"]
        + 0.20 * np.clip(f["f_contrast"] / 0.5, 0.0, 1.0)
    )


def _calibrate(scores: np.ndarray, ring_baseline: float) -> dict:
    """Fit empty/marked components to this sheet's scores."""
    x = scores.reshape(-1, 1)
    fallback_t = float(max(0.12, 0.5 * ring_baseline))
    calibration = {
        "method": "fallback",
        "threshold": fallback_t,
        "separation": 0.0,
        "mu_empty": 0.0,
        "mu_marked": 0.0,
        "sigma_empty": 0.0,
        "sigma_marked": 0.0,
        "marked_count": int(np.sum(scores >= fallback_t)),
    }

    if len(scores) >= 8 and float(np.ptp(scores)) > 1e-4:
        try:
            gm = GaussianMixture(
                n_components=2,
                covariance_type="full",
                random_state=0,
                n_init=3,
                reg_covar=1e-4,
            ).fit(x)
            means = gm.means_.ravel()
            high = int(np.argmax(means))
            low = 1 - high
            p = gm.predict_proba(x)[:, high]
            mu_high, mu_low = float(means[high]), float(means[low])
            var_high = float(gm.covariances_[high].ravel()[0])
            var_low = float(gm.covariances_[low].ravel()[0])
            sep = (mu_high - mu_low) / math.sqrt(max(0.5 * (var_high + var_low), 1e-9))
            high_count = int(np.sum(p >= 0.5))
            valid = (
                mu_high > max(0.15, mu_low + 0.08)
                and high_count >= 3
                and sep >= 2.0
            )
            if valid:
                return {
                    "method": "gmm",
                    "threshold": float(max(mu_low + 3.0 * math.sqrt(var_low), 0.5 * ring_baseline)),
                    "separation": float(sep),
                    "mu_empty": mu_low,
                    "mu_marked": mu_high,
                    "sigma_empty": math.sqrt(var_low),
                    "sigma_marked": math.sqrt(var_high),
                    "marked_count": high_count,
                    "p_marked": p.tolist(),
                    "high_component": high,
                }
            calibration.update(
                {
                    "method": "fallback",
                    "gmm_rejected": True,
                    "gmm_separation": float(sep),
                    "gmm_means": [mu_low, mu_high],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            calibration.update({"method": "fallback", "gmm_error": str(exc)})

    t = calibration["threshold"]
    p = 1.0 / (1.0 + np.exp(-(scores - t) / 0.04))
    calibration["p_marked"] = p.tolist()
    return calibration


def read_bubbles(warped_bgr: np.ndarray, template: Template) -> ReadResult:
    """Measure every bubble and decide its state."""
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ink, raw_ink, pink = _paper_subtracted_ink(rgb)
    glare = _glare_mask(rgb)

    features: list[dict] = []
    for b in template.bubbles:
        cx, cy, r = template.bubble_px(b)
        features.append(_bubble_features(ink, pink, glare, cx, cy, r))

    scores = np.array([_score(f) for f in features], dtype=np.float64)
    ring_baseline = float(np.median([f["f_ring"] for f in features]))
    calibration = _calibrate(scores, ring_baseline)
    p_marked = np.asarray(calibration.get("p_marked", np.zeros_like(scores)), dtype=np.float64)
    t_abs = float(calibration["threshold"])

    readings: list[BubbleReading] = []
    for b, f, s, p in zip(template.bubbles, features, scores, p_marked):
        if not f["observable"]:
            state = UNOBSERVED
        elif p >= 0.90 and s >= t_abs:
            state = MARKED
        elif s < 0.90 * t_abs and p <= 0.20:
            # Clearly below the sheet's own empty/marked boundary.  A score
            # above it is not allowed to become EMPTY just because the GMM's
            # variance is tiny (faint pencil is the important case).
            state = EMPTY
        else:
            state = UNCERTAIN
        readings.append(
            BubbleReading(
                q=b.q,
                option=b.option,
                label=b.label,
                score=float(s),
                p_marked=float(p),
                state=state,
                features={k: float(v) if isinstance(v, (int, float, np.floating)) else bool(v) for k, v in f.items()},
            )
        )

    by_q: dict[int, list[BubbleReading]] = {}
    for r in readings:
        by_q.setdefault(r.q, []).append(r)

    questions: dict[int, QuestionReading] = {}
    for q, rs in sorted(by_q.items()):
        rs.sort(key=lambda x: x.option)
        marked = [r for r in rs if r.state == MARKED]
        uncertain = [r for r in rs if r.state == UNCERTAIN]
        unobserved = [r for r in rs if r.state == UNOBSERVED]

        if unobserved:
            state, reason = "UNOBSERVED", "glare/occlusion on " + ",".join(r.label for r in unobserved)
            conf = 0.0
        elif len(marked) == 1 and not uncertain:
            state, reason = "ANSWERED", ""
            conf = marked[0].p_marked
        elif not marked and not uncertain:
            state, reason = "BLANK", ""
            conf = 1.0 - max((r.p_marked for r in rs), default=0.0)
        elif len(marked) > 1:
            state, reason = "MULTI", ""
            conf = min(r.p_marked for r in marked)
        else:
            state, reason = "REVIEW", ""
            conf = max((r.p_marked for r in rs), default=0.0)

        questions[q] = QuestionReading(
            q=q,
            selected=[r.option for r in marked],
            state=state,
            confidence=float(conf),
            reason=reason,
        )

    return ReadResult(
        bubbles=readings,
        questions=questions,
        calibration={k: v for k, v in calibration.items() if k != "p_marked"},
        diagnostics={
            "mean_ink": float(np.mean(raw_ink)),
            "glare_fraction": float(np.mean(glare)),
            "ring_baseline": ring_baseline,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        },
    )
