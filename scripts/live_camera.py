#!/usr/bin/env python3
"""Live OMR tester: webcam or phone stream -> preview -> SPACE -> graded result.

Examples
--------
Laptop webcam::

    python scripts/live_camera.py

Second camera / USB phone camera::

    python scripts/live_camera.py --source 1

Android phone on the same Wi-Fi (IP Webcam app)::

    python scripts/live_camera.py --source http://192.168.0.23:8080/video

Check the whole pipeline on a still image without opening a camera::

    python scripts/live_camera.py --check "samples/test-new/Asset 1@4x.png"

Keys while live
---------------
SPACE  capture the current frame, grade it, show the result
A      arm/disarm automatic capture when the frame is ready
S      save the current camera frame
Q/ESC  quit

The preview deliberately reads bubbles only on capture.  Detecting the printed
frame and checking focus/exposure every frame is cheap; running the full
120-bubble/GMM reader on every frame would make the preview lag.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omr.geometry import (  # noqa: E402
    find_frame_quad,
    find_page_quad,
    looks_like_flat_scan,
    warp_to_template,
)
from omr.grade import grade, load_key  # noqa: E402
from omr.overlay import render_result  # noqa: E402
from omr.reader import read_bubbles  # noqa: E402
from omr.template import Template  # noqa: E402

WINDOW = "LatticeLock OMR - Live"
GREEN = (40, 180, 40)
YELLOW = (0, 210, 230)
RED = (30, 30, 230)
CYAN = (220, 220, 30)
WHITE = (245, 245, 245)
DARK = (24, 24, 24)


@dataclass
class Assessment:
    ready: bool
    label: str
    color: tuple[int, int, int]
    quad: np.ndarray | None
    quad_norm: np.ndarray | None
    mode: str
    coverage: float
    blur: float
    brightness: float
    stable: bool
    detail: str
    stable_since: float | None


@dataclass
class CaptureResult:
    out_dir: Path
    warp: object
    read: object
    result: object
    image: np.ndarray


def fit_to_box(image: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 0.999:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _camera_backends() -> list[int]:
    if sys.platform.startswith("win"):
        return [
            getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
            cv2.CAP_ANY,
        ]
    if sys.platform == "darwin":
        return [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(
    source: str, width: int, height: int, fps: int
) -> tuple[cv2.VideoCapture | None, str, str | None]:
    """Open a webcam index or a URL/path. Returns (cap, label, error)."""
    cap: cv2.VideoCapture | None = None
    label = source
    if source.isdigit():
        index = int(source)
        label = f"camera {index}"
        for backend in _camera_backends():
            candidate = cv2.VideoCapture(index, backend)
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
    else:
        label = source
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        return None, label, f"could not open {source!r}"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    ok = False
    frame = None
    for _ in range(12):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)
    if not ok or frame is None:
        cap.release()
        return None, label, "camera opened but returned no frames"

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"camera: {label} -> {actual_w}x{actual_h} @ {actual_fps:.0f} fps")
    return cap, label, None


class FrameGrabber:
    """Keep only the newest camera frame so grading never works on a backlog."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.seq = 0
        self.error: str | None = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self.lock:
                    self.error = "camera read failed"
                time.sleep(0.03)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1
                self.error = None

    def latest(self, copy: bool = False) -> tuple[np.ndarray | None, int, str | None]:
        with self.lock:
            frame = self.frame
            if copy and frame is not None:
                frame = frame.copy()
            return frame, self.seq, self.error

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)


def assess_frame(
    image: np.ndarray,
    template: Template,
    previous_quad_norm: np.ndarray | None,
    stable_since: float | None,
    now: float,
    min_blur: float,
) -> Assessment:
    """Cheap per-frame guidance: frame presence, focus, exposure and steadiness."""
    h, w = image.shape[:2]
    small, scale = fit_to_box(image, 960, 960)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    frame_quad, frame_conf = find_frame_quad(small, template)
    quad = frame_quad
    mode = "frame"
    confidence = frame_conf
    min_coverage, max_coverage = 0.08, 0.88

    if quad is None:
        page_quad, page_conf = find_page_quad(small, template)
        if page_quad is not None:
            quad = page_quad
            mode = "page"
            confidence = page_conf
            min_coverage, max_coverage = 0.12, 0.95

    if quad is None:
        return Assessment(
            ready=False,
            label="NO SHEET",
            color=RED,
            quad=None,
            quad_norm=None,
            mode="none",
            coverage=0.0,
            blur=blur,
            brightness=brightness,
            stable=False,
            detail="Show the whole sheet; a dark background makes the edge visible.",
            stable_since=None,
        )

    quad = quad.astype(np.float32) / scale
    quad_norm = quad / np.array([w, h], dtype=np.float32)
    coverage = abs(float(cv2.contourArea(quad))) / max(1.0, float(w * h))
    center_offset = float(
        np.linalg.norm(quad_norm.mean(axis=0) - np.array([0.5, 0.5]))
    )

    stable = False
    if previous_quad_norm is not None:
        delta = float(np.max(np.abs(quad_norm - previous_quad_norm)))
        stable = delta < 0.012

    warnings: list[tuple[str, str]] = []
    if blur < min_blur:
        warnings.append(("TOO BLURRY", "hold still or clean the lens"))
    if brightness < 45:
        warnings.append(("TOO DARK", "add more light"))
    elif brightness > 252:
        warnings.append(("TOO BRIGHT", "reduce light or tilt away from glare"))
    if coverage < min_coverage:
        warnings.append(("MOVE CLOSER", f"sheet fills {coverage * 100:.0f}% of frame"))
    elif coverage > max_coverage:
        warnings.append(("MOVE BACK", f"sheet fills {coverage * 100:.0f}% of frame"))
    if center_offset > 0.16:
        warnings.append(("CENTER THE SHEET", "keep the whole sheet inside the frame"))
    if not stable:
        warnings.append(("HOLD STEADY", "wait for the green ready banner"))

    detail = (
        f"{mode} {confidence:.2f} | fill {coverage * 100:.0f}% | "
        f"light {brightness:.0f} | blur {blur:.0f}"
    )

    if warnings:
        label, hint = warnings[0]
        return Assessment(
            ready=False,
            label=label,
            color=YELLOW,
            quad=quad,
            quad_norm=quad_norm,
            mode=mode,
            coverage=coverage,
            blur=blur,
            brightness=brightness,
            stable=stable,
            detail=f"{hint}  ({detail})",
            stable_since=None,
        )

    new_since = stable_since if stable_since is not None else now
    held = now - new_since
    if held < 0.45:
        return Assessment(
            ready=False,
            label="HOLD STEADY",
            color=YELLOW,
            quad=quad,
            quad_norm=quad_norm,
            mode=mode,
            coverage=coverage,
            blur=blur,
            brightness=brightness,
            stable=True,
            detail=f"ready in {0.45 - held:.1f}s  ({detail})",
            stable_since=new_since,
        )

    return Assessment(
        ready=True,
        label="READY - PRESS SPACE",
        color=GREEN,
        quad=quad,
        quad_norm=quad_norm,
        mode=mode,
        coverage=coverage,
        blur=blur,
        brightness=brightness,
        stable=True,
        detail=detail,
        stable_since=new_since,
    )


def draw_preview(
    frame: np.ndarray,
    assessment: Assessment,
    fps: float,
    source_label: str,
    auto: bool,
    notice: str = "",
) -> np.ndarray:
    view, scale = fit_to_box(frame, 1180, 760)
    if assessment.quad is not None:
        quad = (assessment.quad * scale).astype(np.int32)
        cv2.polylines(view, [quad], True, assessment.color, 3, cv2.LINE_AA)
        for point in quad:
            cv2.circle(view, tuple(point), 5, assessment.color, -1, cv2.LINE_AA)

    h, w = view.shape[:2]
    bar_h = 98
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), DARK, -1)
    cv2.rectangle(overlay, (0, h - 40), (w, h), DARK, -1)
    cv2.addWeighted(overlay, 0.72, view, 0.28, 0, view)

    cv2.putText(
        view,
        assessment.label,
        (18, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        assessment.color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        assessment.detail[:120],
        (18, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    source_short = source_label if len(source_label) <= 42 else source_label[:39] + "..."
    cv2.putText(
        view,
        f"{fps:4.1f} FPS  {source_short}",
        (w - 300, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    if auto:
        cv2.putText(
            view,
            "AUTO ARMED",
            (w - 150, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CYAN,
            2,
            cv2.LINE_AA,
        )
    if notice:
        cv2.putText(
            view,
            notice[:110],
            (18, bar_h + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            YELLOW,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        view,
        "SPACE capture & grade   |   A auto   |   S snapshot   |   Q quit",
        (14, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    return view


def draw_result(result: CaptureResult) -> np.ndarray:
    view, _ = fit_to_box(result.image, 900, 980)
    bar_h = 78
    canvas = np.full((view.shape[0] + bar_h, view.shape[1], 3), 245, np.uint8)
    canvas[bar_h:] = view
    grade_result = result.result
    cv2.putText(
        canvas,
        f"GRADE {grade_result.total:g}/{grade_result.maximum:g}",
        (16, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (30, 30, 200),
        2,
        cv2.LINE_AA,
    )
    summary = (
        f"correct {grade_result.correct}   wrong {grade_result.wrong}   "
        f"blank {grade_result.blank}   review {grade_result.review}"
    )
    cv2.putText(
        canvas,
        summary,
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    hint = "SPACE next sheet  |  S save copy  |  Q quit"
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        canvas,
        hint,
        (canvas.shape[1] - tw - 16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


def process_capture(
    image: np.ndarray, template: Template, key: dict, out_root: Path, source: str
) -> CaptureResult:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(out_dir / "capture.jpg"),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )

    warp = warp_to_template(
        image,
        template,
        force_detect=True,
        dpi=template.canonical_dpi,
    )
    if warp.mode == "full_frame" and not looks_like_flat_scan(image, template):
        raise RuntimeError(
            "sheet not found - fill the frame, use a dark background, or tap to focus"
        )

    read = read_bubbles(warp.warped, template)
    result = grade(read, key, template)
    result_path = out_dir / "result.jpg"
    render_result(
        warp.warped,
        template,
        read,
        result,
        result_path,
        audit_path=out_dir / "audit_canonical.png",
    )

    payload = {
        "source": source,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "template_id": template.template_id,
        "warp": {
            "mode": warp.mode,
            "page_confidence": round(float(warp.page_confidence), 4),
            "corners": [
                [round(float(x), 2), round(float(y), 2)] for x, y in warp.corners
            ],
            "debug": warp.debug,
        },
        "calibration": read.calibration,
        "diagnostics": read.diagnostics,
        "answers": {
            str(q): {
                "selected": r.selected,
                "state": r.state,
                "confidence": round(float(r.confidence), 4),
                "reason": r.reason,
            }
            for q, r in read.questions.items()
        },
        "grade": result.to_dict(),
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result_image = cv2.imread(str(result_path))
    if result_image is None:
        raise RuntimeError("result image was written but could not be read back")
    return CaptureResult(out_dir, warp, read, result, result_image)


def print_capture(capture: CaptureResult) -> None:
    warp = capture.warp
    read = capture.read
    result = capture.result
    print(
        f"warp: {warp.mode} (confidence {warp.page_confidence:.2f}) -> "
        f"{warp.warped.shape[1]}x{warp.warped.shape[0]}"
    )
    print(
        f"calibration: {read.calibration.get('method')} "
        f"sep={read.calibration.get('separation', 0):.2f} "
        f"t={read.calibration.get('threshold', 0):.3f}"
    )
    print(
        f"total: {result.total:g}/{result.maximum:g}  correct={result.correct} "
        f"wrong={result.wrong} blank={result.blank} multi={result.multi} "
        f"review={result.review}"
    )
    for q in result.review_questions:
        print(f"  review Q{q}: {read.questions[q].reason or read.questions[q].state}")
    print(f"saved: {capture.out_dir}")


def run_check(
    path: str, template: Template, key: dict, out_root: Path, min_blur: float
) -> int:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not read image {path}", file=sys.stderr)
        return 2
    assessment = assess_frame(image, template, None, None, time.time(), min_blur)
    print(f"quality: {assessment.label}  {assessment.detail}")
    print("note: a single still has no steadiness history; HOLD STEADY is expected here.")
    try:
        capture = process_capture(image, template, key, out_root, f"check:{path}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_capture(capture)
    return 0


def list_cameras(width: int, height: int) -> int:
    print("Probing camera indices 0..5 (a camera already in use may fail):")
    found = False
    for index in range(6):
        cap, label, error = open_camera(str(index), width, height, 30)
        if cap is None:
            print(f"  {index}: {error}")
            continue
        found = True
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"  {index}: OK  {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  {index}: opened but no frame")
        cap.release()
    if not found:
        print("No camera found. Try a phone stream with --source http://PHONE_IP:8080/video")
    return 0


def _save_snapshot(frame: np.ndarray, out_root: Path) -> Path:
    snap_dir = out_root / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = snap_dir / f"snapshot_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def _save_result_copy(capture: CaptureResult, out_root: Path) -> Path:
    saved_dir = out_root / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = saved_dir / f"result_{stamp}.jpg"
    cv2.imwrite(str(path), capture.image)
    return path


def run_camera(args: argparse.Namespace, template: Template, key: dict) -> int:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cap, source_label, error = open_camera(
        args.source, args.width, args.height, args.fps
    )
    if cap is None or error:
        print(f"error: {error}", file=sys.stderr)
        print("\nPhone as a webcam:")
        print("  Android: install 'IP Webcam', start the server, then run:")
        print("    python scripts/live_camera.py --source http://PHONE_IP:8080/video")
        print("  USB/iPhone: use DroidCam/Irium/Camo, then try --source 1")
        print("  Both devices must be on the same Wi-Fi network.")
        return 2

    grabber = FrameGrabber(cap)
    assessment: Assessment | None = None
    previous_quad_norm: np.ndarray | None = None
    stable_since: float | None = None
    last_assess = 0.0
    fps = 0.0
    fps_t0 = time.time()
    fps_count = 0
    last_frame_time = time.time()
    state = "live"
    auto = bool(args.auto)
    last_capture: CaptureResult | None = None
    notice = ""
    notice_until = 0.0

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1180, 800)
    except cv2.error:
        grabber.stop()
        cap.release()
        print(
            "error: this OpenCV build has no GUI support (opencv-python-headless).\n"
            "Install the normal wheel, then retry:\n"
            "  python -m pip install --upgrade opencv-python",
            file=sys.stderr,
        )
        return 2

    try:
        while True:
            frame, _seq, read_error = grabber.latest()
            now = time.time()
            if frame is None:
                if now - last_frame_time > 5.0:
                    print(f"error: {read_error or 'no frames from camera'}", file=sys.stderr)
                    return 1
                time.sleep(0.02)
                continue

            last_frame_time = now
            fps_count += 1
            if now - fps_t0 >= 0.5:
                fps = fps_count / (now - fps_t0)
                fps_t0 = now
                fps_count = 0

            if state == "live":
                if assessment is None or now - last_assess >= args.detect_interval:
                    assessment = assess_frame(
                        frame,
                        template,
                        previous_quad_norm,
                        stable_since,
                        now,
                        args.min_blur,
                    )
                    previous_quad_norm = assessment.quad_norm
                    stable_since = assessment.stable_since
                    last_assess = now

                if auto and assessment.ready:
                    auto = False
                    state = "result"
                    print("auto capture...")
                    try:
                        capture_frame = grabber.latest(copy=True)[0]
                        if capture_frame is None:
                            capture_frame = frame
                        last_capture = process_capture(
                            capture_frame,
                            template,
                            key,
                            out_root,
                            f"live:{source_label}",
                        )
                        print_capture(last_capture)
                        notice = ""
                    except Exception as exc:
                        state = "live"
                        notice = f"Capture failed: {exc}"
                        notice_until = time.time() + 5.0
                        print(f"capture failed: {exc}", file=sys.stderr)
                    continue

                view = draw_preview(
                    frame,
                    assessment,
                    fps,
                    source_label,
                    auto,
                    notice if time.time() < notice_until else "",
                )
                cv2.imshow(WINDOW, view)
                keypress = cv2.waitKey(1) & 0xFF
                if keypress in (27, ord("q")):
                    break
                if keypress == ord("a"):
                    auto = not auto
                    notice = "auto armed" if auto else "auto disarmed"
                    notice_until = time.time() + 2.0
                elif keypress == ord("s"):
                    path = _save_snapshot(frame, out_root)
                    notice = f"saved {path}"
                    notice_until = time.time() + 3.0
                    print(f"snapshot: {path}")
                elif keypress == 32:
                    print("grading...")
                    try:
                        capture_frame = grabber.latest(copy=True)[0]
                        if capture_frame is None:
                            capture_frame = frame
                        last_capture = process_capture(
                            capture_frame,
                            template,
                            key,
                            out_root,
                            f"live:{source_label}",
                        )
                        print_capture(last_capture)
                        state = "result"
                        notice = ""
                    except Exception as exc:
                        notice = f"Capture failed: {exc}"
                        notice_until = time.time() + 5.0
                        print(f"capture failed: {exc}", file=sys.stderr)
            else:
                if last_capture is None:
                    state = "live"
                    continue
                cv2.imshow(WINDOW, draw_result(last_capture))
                keypress = cv2.waitKey(1) & 0xFF
                if keypress in (27, ord("q")):
                    break
                if keypress == 32:
                    state = "live"
                    assessment = None
                    previous_quad_norm = None
                    stable_since = None
                    last_assess = 0.0
                    notice = "ready for the next sheet"
                    notice_until = time.time() + 2.0
                elif keypress == ord("s"):
                    path = _save_result_copy(last_capture, out_root)
                    notice = f"saved {path}"
                    print(f"saved copy: {path}")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--source",
        default="0",
        help="camera index, video path, or phone MJPEG URL (default: 0)",
    )
    ap.add_argument("--template", default=str(ROOT / "templates" / "q-paper.json"))
    ap.add_argument("--key", default=str(ROOT / "keys" / "sample_key.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "live"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--min-blur",
        type=float,
        default=45.0,
        help="Laplacian-variance threshold at 960 px (default: 45)",
    )
    ap.add_argument(
        "--detect-interval",
        type=float,
        default=0.20,
        help="seconds between frame/quality checks (default: 0.20)",
    )
    ap.add_argument("--auto", action="store_true", help="start with auto-capture armed")
    ap.add_argument(
        "--check",
        metavar="IMAGE",
        default=None,
        help="grade one image through the live capture path and exit",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="probe camera indices 0..5 and exit",
    )
    args = ap.parse_args()

    if args.list:
        return list_cameras(args.width, args.height)

    template_path = Path(args.template)
    key_path = Path(args.key)
    if not template_path.exists():
        print(f"error: template not found: {template_path}", file=sys.stderr)
        return 2
    if not key_path.exists():
        print(f"error: key not found: {key_path}", file=sys.stderr)
        return 2

    template = Template.load(template_path)
    key = load_key(key_path, template)
    out_root = Path(args.out)

    if args.check:
        return run_check(args.check, template, key, out_root, args.min_blur)
    return run_camera(args, template, key)


if __name__ == "__main__":
    raise SystemExit(main())
