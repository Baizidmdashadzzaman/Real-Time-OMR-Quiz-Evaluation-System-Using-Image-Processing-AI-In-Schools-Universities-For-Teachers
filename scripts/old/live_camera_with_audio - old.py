#!/usr/bin/env python3
"""Continuous OMR scanner with voice mark announcements.

This is the batch-friendly companion to ``live_camera.py``.  It does not stop
on a result screen: the camera preview keeps running, each sheet is graded in a
background worker, and the full graded sheet (mark box plus green/red/blue
bubble marks) is shown for a short hold before returning to the live camera.
A text-to-speech queue announces the mark once per sheet.

Examples
--------
Laptop webcam::

    python scripts/live_camera_with_audio.py

Phone on the same Wi-Fi (Android IP Webcam)::

    python scripts/live_camera_with_audio.py --source "http://192.168.1.45:8080/video"

Quiet mode (no voice)::

    python scripts/live_camera_with_audio.py --no-voice

Test the voice engine only::

    python scripts/live_camera_with_audio.py --test-voice

Keys while running
------------------
SPACE  scan now; on the graded result, return to live immediately
P      pause / resume automatic scanning
V      mute / unmute voice
S      save the raw camera frame
Q/ESC  quit

How it avoids double announcements
----------------------------------
After a successful scan the scanner is "disarmed".  It re-arms when the sheet
has been absent for ``--absence-reset`` seconds, or when the detected sheet has
moved by more than ``--shift-reset`` (a new sheet placed in its place).  The same
sheet left in front of the camera is not scanned again unless SPACE is pressed.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from live_camera import (  # noqa: E402
    Assessment,
    CaptureResult,
    FrameGrabber,
    assess_frame,
    fit_to_box,
    list_cameras,
    open_camera,
    process_capture,
)
from omr.grade import load_key  # noqa: E402
from omr.template import Template  # noqa: E402

WINDOW = "LatticeLock OMR - Continuous"
GREEN = (40, 180, 40)
YELLOW = (0, 210, 230)
RED = (30, 30, 230)
ORANGE = (0, 140, 255)
WHITE = (245, 245, 245)
DARK = (24, 24, 24)


@dataclass
class ScanOutcome:
    ok: bool
    capture: CaptureResult | None
    error: str
    quad_norm: np.ndarray | None


class Announcer:
    """Speak mark phrases in a background thread, without blocking OpenCV."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable speech, starting the worker if it was never needed."""
        self.enabled = enabled
        if enabled and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        if self._thread is None or not self._thread.is_alive():
            self.set_enabled(True)
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Keep the newest announcement rather than falling behind.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                pass

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            self._speak(text)

    def _speak(self, text: str) -> None:
        try:
            if sys.platform.startswith("win"):
                powershell = shutil.which("powershell.exe") or shutil.which("powershell")
                if not powershell:
                    raise RuntimeError("powershell.exe not found")
                safe = text.replace("'", "''")
                command = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Speak('{safe}');"
                )
                subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                    check=True,
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                return

            if sys.platform == "darwin":
                say = shutil.which("say")
                if not say:
                    raise RuntimeError("macOS 'say' not found")
                subprocess.run(
                    [say, text],
                    check=True,
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return

            for executable in ("espeak", "spd-say"):
                path = shutil.which(executable)
                if path:
                    subprocess.run(
                        [path, text],
                        check=True,
                        timeout=20,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
            raise RuntimeError("no TTS engine found (install espeak/spd-say)")
        except Exception as exc:  # speech is optional; never stop scanning for it
            print(f"[voice] {text}  (speech unavailable: {exc})", file=sys.stderr)

    def close(self, wait: bool = False) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if wait:
            self._thread.join(timeout=20)


def _fmt_mark(value: float) -> str:
    """Match the one-decimal formatting used by the saved result image."""
    return str(int(round(value))) if float(value).is_integer() else f"{value:.1f}"


def _as_bgr(frame: np.ndarray | None) -> np.ndarray | None:
    """Normalise camera/video frames to 3-channel BGR."""
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _mark_phrase(capture: CaptureResult) -> str:
    result = capture.result
    return f"Mark {_fmt_mark(result.total)} out of {_fmt_mark(result.maximum)}."


def _worker(
    frame: np.ndarray,
    quad_norm: np.ndarray | None,
    template: Template,
    key: dict,
    out_root: Path,
    source: str,
    results: queue.Queue[ScanOutcome],
) -> None:
    try:
        capture = process_capture(frame, template, key, out_root, source)
        results.put(ScanOutcome(True, capture, "", quad_norm))
    except Exception as exc:
        results.put(ScanOutcome(False, None, str(exc), quad_norm))


def _draw_badge(
    view: np.ndarray,
    quad: np.ndarray | None,
    capture: CaptureResult | None,
    processing: bool,
    paused: bool,
    color: tuple[int, int, int],
    status_label: str = "SHOW SHEET",
) -> None:
    """Draw the mark on top of the detected sheet (or a short status)."""
    h, w = view.shape[:2]

    if processing:
        label, subtitle, badge_color = "READING...", "PLEASE WAIT", YELLOW
    elif capture is not None:
        label = f"{_fmt_mark(capture.result.total)}/{_fmt_mark(capture.result.maximum)}"
        subtitle = "MARK"
        if capture.result.review:
            badge_color = YELLOW
        elif capture.result.total <= 0:
            badge_color = ORANGE
        else:
            badge_color = GREEN
    elif paused:
        label, subtitle, badge_color = "PAUSED", "", YELLOW
    else:
        label, subtitle, badge_color = status_label or "SHOW SHEET", "", color

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.9, min(2.3, w / 720.0))
    thickness = max(2, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    sub_scale = max(0.45, scale * 0.36)
    (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    pad = int(round(18 * scale))
    gap = int(round(8 * scale)) if subtitle else 0
    box_w = max(tw, sw) + 2 * pad
    box_h = th + sh + gap + 2 * pad

    if quad is not None:
        top_mid = (
            float((quad[0][0] + quad[1][0]) / 2.0),
            float((quad[0][1] + quad[1][1]) / 2.0),
        )
        x = int(top_mid[0] - box_w / 2.0)
        y = int(top_mid[1] - box_h - 12)
    else:
        x = int(w / 2 - box_w / 2.0)
        y = 54
    x = max(8, min(x, w - box_w - 8))
    y = max(48, min(y, h - box_h - 48))

    overlay = view.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), DARK, -1)
    cv2.addWeighted(overlay, 0.80, view, 0.20, 0, view)
    cv2.rectangle(view, (x, y), (x + box_w, y + box_h), badge_color, 3)

    text_y = y + pad
    if subtitle:
        cv2.putText(
            view,
            subtitle,
            (x + pad, text_y + sh),
            cv2.FONT_HERSHEY_SIMPLEX,
            sub_scale,
            badge_color,
            1,
            cv2.LINE_AA,
        )
        text_y += sh + gap
    cv2.putText(
        view,
        label,
        (x + pad, text_y + th),
        font,
        scale,
        badge_color,
        thickness,
        cv2.LINE_AA,
    )
    _ = baseline


def draw_continuous(
    frame: np.ndarray,
    assessment: Assessment | None,
    capture: CaptureResult | None,
    last_mark: tuple[float, float] | None,
    fps: float,
    scans_done: int,
    processing: bool,
    paused: bool,
    voice_on: bool,
    notice: str,
) -> np.ndarray:
    view, scale = fit_to_box(frame, 1180, 760)
    h, w = view.shape[:2]

    quad = None
    if assessment is not None and assessment.quad is not None:
        quad = (assessment.quad * scale).astype(np.int32)
        color = assessment.color
        if capture is not None and not processing:
            color = GREEN if capture.result.review == 0 else YELLOW
        cv2.polylines(view, [quad], True, color, 3, cv2.LINE_AA)
        for point in quad:
            cv2.circle(view, tuple(point), 4, color, -1, cv2.LINE_AA)

    # Top status bar.
    bar_h = 44
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), DARK, -1)
    cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
    cv2.putText(
        view,
        "CONTINUOUS OMR",
        (16, 29),
        cv2.FONT_HERSHEY_DUPLEX,
        0.68,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    if last_mark is not None:
        cv2.putText(
            view,
            f"LAST {_fmt_mark(last_mark[0])}/{_fmt_mark(last_mark[1])}",
            (230, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            GREEN,
            1,
            cv2.LINE_AA,
        )
    right = f"{fps:4.1f} FPS   scans {scans_done}   voice {'ON' if voice_on else 'OFF'}"
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        view,
        right,
        (w - rw - 14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    # Bottom status / key help.
    bottom_h = 42
    overlay = view.copy()
    cv2.rectangle(overlay, (0, h - bottom_h), (w, h), DARK, -1)
    cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
    help_text = "SPACE scan now   |   P pause   |   V voice   |   S snapshot   |   Q quit"
    cv2.putText(
        view,
        help_text,
        (14, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    if notice:
        status, status_color = notice, YELLOW
    elif paused:
        status, status_color = "PAUSED - press P to resume", YELLOW
    elif processing:
        status, status_color = "Reading sheet in background...", YELLOW
    elif assessment is not None:
        status, status_color = assessment.label, assessment.color
    else:
        status, status_color = "Show a sheet", WHITE
    cv2.putText(
        view,
        status[:110],
        (14, h - bottom_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        status_color,
        1,
        cv2.LINE_AA,
    )

    if assessment is None:
        badge_status = "SHOW SHEET"
    elif assessment.ready:
        badge_status = "HOLD STILL"
    else:
        badge_status = assessment.label
    _draw_badge(view, quad, capture, processing, paused, WHITE, badge_status)
    return view


def draw_result_view(
    capture: CaptureResult,
    remaining: float,
    paused: bool,
    voice_on: bool,
    scans_done: int,
) -> np.ndarray:
    """Show the saved graded sheet (mark box + green/red/blue marks)."""
    canvas = np.full((800, 1180, 3), 245, np.uint8)
    graded, _ = fit_to_box(capture.image, 760, 690)
    gh, gw = graded.shape[:2]
    x = 20
    y = 58 + max(0, (690 - gh) // 2)
    canvas[y : y + gh, x : x + gw] = graded
    cv2.rectangle(canvas, (x - 1, y - 1), (x + gw, y + gh), DARK, 2)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 44), DARK, -1)
    cv2.rectangle(overlay, (0, 756), (canvas.shape[1], 800), DARK, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.putText(
        canvas,
        "GRADED SHEET",
        (16, 29),
        cv2.FONT_HERSHEY_DUPLEX,
        0.70,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    right = f"scans {scans_done}   voice {'ON' if voice_on else 'OFF'}"
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        canvas,
        right,
        (canvas.shape[1] - rw - 14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    result = capture.result
    color = YELLOW if result.review else GREEN
    if result.total <= 0:
        color = ORANGE
    panel_x = x + gw + 40
    cv2.putText(
        canvas,
        "MARK IN THE PRINTED BOX",
        (panel_x, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{_fmt_mark(result.total)}/{_fmt_mark(result.maximum)}",
        (panel_x, 185),
        cv2.FONT_HERSHEY_DUPLEX,
        2.8,
        color,
        4,
        cv2.LINE_AA,
    )
    rows = [
        ("correct", result.correct, GREEN),
        ("wrong", result.wrong, RED),
        ("blank", result.blank, ORANGE),
        ("review", result.review, YELLOW),
    ]
    for i, (label, value, row_color) in enumerate(rows):
        y_row = 255 + i * 42
        cv2.putText(
            canvas,
            label,
            (panel_x, y_row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(value),
            (panel_x + 170, y_row),
            cv2.FONT_HERSHEY_DUPLEX,
            0.72,
            row_color,
            2,
            cv2.LINE_AA,
        )

    if paused:
        next_text = "PAUSED - automatic scanning off"
    elif remaining > 0:
        next_text = f"Live camera in {remaining:.1f}s"
    else:
        next_text = "SPACE: return to live camera"
    cv2.putText(
        canvas,
        next_text,
        (panel_x, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        YELLOW if paused or remaining > 0 else WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "SPACE live view   |   P pause   |   V voice   |   Q quit",
        (panel_x, 535),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    if remaining > 0:
        cv2.putText(
            canvas,
            "Press SPACE to return now",
            (14, 785),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _save_snapshot(frame: np.ndarray, out_root: Path) -> Path:
    snap_dir = out_root / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = snap_dir / f"snapshot_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def run_continuous(args: argparse.Namespace, template: Template, key: dict) -> int:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cap, source_label, error = open_camera(
        args.source, args.width, args.height, args.fps
    )
    if cap is None or error:
        print(f"error: {error}", file=sys.stderr)
        print("\nPhone as a camera:")
        print("  Android: install 'IP Webcam', start the server, then run:")
        print("    python scripts/live_camera_with_audio.py --source http://PHONE_IP:8080/video")
        print("  USB/iPhone: use DroidCam/Irium/Camo, then try --source 1")
        print("  Both devices must be on the same Wi-Fi network.")
        return 2

    grabber = FrameGrabber(cap)
    announcer = Announcer(enabled=not args.no_voice)
    results: queue.Queue[ScanOutcome] = queue.Queue()
    worker: threading.Thread | None = None

    assessment: Assessment | None = None
    previous_quad_norm: np.ndarray | None = None
    stable_since: float | None = None
    last_assess = 0.0
    last_frame_time = time.time()
    fps = 0.0
    fps_t0 = time.time()
    fps_count = 0

    processing = False
    paused = bool(args.start_paused)
    armed = not paused
    voice_on = not args.no_voice
    scans_done = 0
    last_capture: CaptureResult | None = None
    result_until = 0.0
    last_mark: tuple[float, float] | None = None
    last_quad: np.ndarray | None = None
    last_scan_time = 0.0
    absence_since: float | None = None
    shift_since: float | None = None
    retry_at: float | None = None
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
        announcer.close()
        return 2

    def begin_scan(frame: np.ndarray, quad_norm: np.ndarray | None) -> None:
        nonlocal processing, armed, result_until, last_scan_time
        nonlocal worker
        if processing:
            return
        capture_frame = _as_bgr(grabber.latest(copy=True)[0])
        if capture_frame is None:
            capture_frame = _as_bgr(frame.copy())
        processing = True
        armed = False
        result_until = 0.0
        last_scan_time = time.time()
        print("scanning...")
        worker = threading.Thread(
            target=_worker,
            args=(
                capture_frame,
                None if quad_norm is None else quad_norm.copy(),
                template,
                key,
                out_root,
                f"continuous:{source_label}",
                results,
            ),
            daemon=True,
        )
        worker.start()

    try:
        while True:
            frame, _seq, read_error = grabber.latest()
            frame = _as_bgr(frame)
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

            # Collect a finished background scan without blocking the preview.
            try:
                outcome = results.get_nowait()
            except queue.Empty:
                outcome = None
            if outcome is not None:
                processing = False
                worker = None
                if outcome.ok and outcome.capture is not None:
                    last_capture = outcome.capture
                    result_until = now + max(0.0, args.result_hold)
                    last_quad = outcome.quad_norm
                    last_mark = (
                        outcome.capture.result.total,
                        outcome.capture.result.maximum,
                    )
                    scans_done += 1
                    print(
                        f"mark: {_fmt_mark(outcome.capture.result.total)}/"
                        f"{_fmt_mark(outcome.capture.result.maximum)}  "
                        f"review={outcome.capture.result.review}  "
                        f"saved={outcome.capture.out_dir}"
                    )
                    armed = False
                    absence_since = None
                    shift_since = None
                    retry_at = None
                    notice = f"saved {outcome.capture.out_dir.name}"
                    notice_until = now + 3.0
                    if voice_on:
                        announcer.say(_mark_phrase(outcome.capture))
                else:
                    last_capture = None
                    result_until = 0.0
                    last_quad = None
                    armed = False
                    retry_at = now + 2.0
                    notice = f"scan failed: {outcome.error}"
                    notice_until = now + 5.0
                    print(f"scan failed: {outcome.error}", file=sys.stderr)

            if paused:
                armed = False
            elif not processing:
                current_norm = assessment.quad_norm if assessment is not None else None

                # Re-arm after the sheet has been removed.
                if not armed:
                    if current_norm is None:
                        if absence_since is None:
                            absence_since = now
                        elif now - absence_since >= args.absence_reset:
                            armed = True
                            absence_since = None
                            shift_since = None
                            notice = "ready for the next sheet"
                            notice_until = now + 2.0
                    else:
                        absence_since = None
                        if last_quad is not None:
                            delta = float(np.max(np.abs(current_norm - last_quad)))
                            if delta > args.shift_reset:
                                if shift_since is None:
                                    shift_since = now
                                elif now - shift_since >= 0.30:
                                    armed = True
                                    shift_since = None
                                    notice = "new sheet detected"
                                    notice_until = now + 2.0
                            else:
                                shift_since = None
                        elif retry_at is not None and now >= retry_at:
                            armed = True

                if armed and assessment.ready and now - last_scan_time >= args.min_interval:
                    begin_scan(frame, assessment.quad_norm)

            if last_capture is not None and now < result_until:
                view = draw_result_view(
                    last_capture,
                    max(0.0, result_until - now),
                    paused,
                    voice_on,
                    scans_done,
                )
            else:
                badge_capture = (
                    last_capture
                    if last_capture is not None and not armed and not processing
                    else None
                )
                view = draw_continuous(
                    frame,
                    assessment,
                    badge_capture,
                    last_mark,
                    fps,
                    scans_done,
                    processing,
                    paused,
                    voice_on,
                    notice if now < notice_until else "",
                )
            cv2.imshow(WINDOW, view)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress in (27, ord("q")):
                break
            if keypress == 32:  # SPACE: return from result view, or scan now
                if last_capture is not None and now < result_until:
                    result_until = 0.0
                    notice = "live view"
                    notice_until = now + 2.0
                elif processing:
                    notice = "still reading the previous sheet"
                    notice_until = now + 2.0
                else:
                    if assessment is None:
                        assessment = assess_frame(
                            frame,
                            template,
                            previous_quad_norm,
                            stable_since,
                            now,
                            args.min_blur,
                        )
                    begin_scan(frame, assessment.quad_norm)
            elif keypress == ord("p"):
                paused = not paused
                if paused:
                    armed = False
                    notice = "paused"
                else:
                    assessment = None
                    last_assess = 0.0
                    absence_since = None
                    shift_since = None
                    armed = last_capture is None
                    notice = "resumed"
                notice_until = now + 2.0
            elif keypress == ord("v"):
                voice_on = not voice_on
                announcer.set_enabled(voice_on)
                notice = f"voice {'on' if voice_on else 'off'}"
                notice_until = now + 2.0
                if voice_on:
                    announcer.say("Voice on.")
            elif keypress == ord("s"):
                path = _save_snapshot(frame, out_root)
                notice = f"saved {path.name}"
                notice_until = now + 3.0
                print(f"snapshot: {path}")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cap.release()
        cv2.destroyAllWindows()
        announcer.close(wait=False)
        if worker is not None:
            worker.join(timeout=2.0)
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
    ap.add_argument("--out", default=str(ROOT / "results" / "live_audio"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-blur", type=float, default=45.0)
    ap.add_argument("--detect-interval", type=float, default=0.20)
    ap.add_argument(
        "--min-interval",
        type=float,
        default=1.0,
        help="minimum seconds between automatic scans (default: 1.0)",
    )
    ap.add_argument(
        "--absence-reset",
        type=float,
        default=0.45,
        help="seconds without a sheet before re-arming (default: 0.45)",
    )
    ap.add_argument(
        "--shift-reset",
        type=float,
        default=0.08,
        help="normalised corner movement treated as a new sheet (default: 0.08)",
    )
    ap.add_argument(
        "--result-hold",
        type=float,
        default=2.5,
        help="seconds to show the graded sheet after a scan; 0 disables (default: 2.5)",
    )
    ap.add_argument("--no-voice", action="store_true", help="disable speech")
    ap.add_argument(
        "--test-voice",
        action="store_true",
        help="speak one sample phrase and exit",
    )
    ap.add_argument(
        "--start-paused",
        action="store_true",
        help="open the camera but do not scan until P or SPACE",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="probe camera indices 0..5 and exit",
    )
    args = ap.parse_args()

    if args.list:
        return list_cameras(args.width, args.height)

    if args.test_voice:
        announcer = Announcer(enabled=not args.no_voice)
        if announcer.enabled:
            announcer.say("Voice ready. Mark 29 out of 30.")
            announcer.close(wait=True)
        else:
            print("voice is disabled by --no-voice")
        return 0

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
    return run_continuous(args, template, key)


if __name__ == "__main__":
    raise SystemExit(main())
