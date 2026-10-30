# OMR Scanner — Full Process Guide

Status: **M0 still-image reader + live preview/capture tester**
Last updated: 2026-10-30

This document is the operating manual for the project. It covers:

- installing and running the scanner;
- scanning a still image and a live camera/phone stream;
- checking that a result is trustworthy;
- understanding every output file;
- creating keys and changing marking policy;
- changing the printed OMR sheet design safely;
- diagnosing the common failures.

The design rationale and research are in [realtime-plan.md](realtime-plan.md);
the short overview is in [README.md](README.md).

---

## Contents

1. [What this system does](#1-what-this-system-does)
2. [Project map](#2-project-map)
3. [One-time setup](#3-one-time-setup)
4. [Ten-minute smoke test](#4-ten-minute-smoke-test)
5. [Running the scanner](#5-running-the-scanner)
6. [Command reference](#6-command-reference)
7. [How to check the scanner](#7-how-to-check-the-scanner)
8. [Understanding the output](#8-understanding-the-output)
9. [How grading works](#9-how-grading-works)
10. [Keys, marks, and policies](#10-keys-marks-and-policies)
11. [Changing the OMR sheet design](#11-changing-the-omr-sheet-design)
12. [Troubleshooting](#12-troubleshooting)
13. [Current limitations](#13-current-limitations)
14. [Housekeeping and privacy](#14-housekeeping-and-privacy)
15. [Quick reference](#15-quick-reference)

---

## 1. What this system does

The pipeline is:

```text
printed OMR PDF
      │
      │  compile_template.py  (once per design)
      ▼
templates/<name>.json          exact bubble centres in millimetres
      │
      │
camera / scan ──► align + perspective warp ──► canonical 200-DPI sheet
                                                   │
                                                   ▼
                                      read every bubble on that sheet
                                      (per-sheet calibration, no global threshold)
                                                   │
                                                   ▼
                                       answers + confidence + review flags
                                                   │
                                    key + marks + policy (grade.py)
                                                   │
                                                   ▼
                             result.jpg + result.json + audit_canonical.png
```

It reads **one captured sheet at a time**. `live_camera.py` grades one sheet
on `SPACE`; `live_camera_with_audio.py` repeats sheet-to-sheet for a stack,
but neither continuously tracks bubbles while the sheet moves. Results are
files on disk, not a database.

The supplied template is for the 30-question, 4-option, two-band
`q-paper.pdf` sheet. A different layout needs a new template and usually a new
key — see [section 11](#11-changing-the-omr-sheet-design).

---

## 2. Project map

```text
omr/
  template.py     compile/load the sheet template; mm geometry
  geometry.py     find the printed frame or page; resolve orientation; warp
  reader.py       measure bubbles; per-sheet GMM calibration; question states
  grade.py        load key; apply marks/policy; totals
  overlay.py      draw green/red/blue/orange/yellow/purple result
scripts/
  compile_template.py   PDF -> templates/<name>.json
  read_image.py         one still image -> result.jpg/json
  make_key.py           interactive teacher answer key -> keys/<name>.json
  live_camera.py        webcam/phone preview -> SPACE -> result
  live_camera_with_audio.py  continuous scan -> mark badge -> voice
  selftest.py           fixed acceptance checks for the supplied sheet
templates/
  q-paper.json               compiled geometry (the runtime source of truth)
  q-paper_reference.png      rendered reference at 200 DPI
keys/
  sample_key.json            teacher key for the supplied 30 questions
samples/
  q-paper_blank.png          blank control
  q-paper_mixed.png          blank/double/faint control
  q-paper_tilted.jpg         angled control
  test-new/                  rotated, cropped and skewed captures
results/                     generated output; safe to delete
requirements.txt
```

The important separation:

- **template** = where the bubbles are, in millimetres;
- **key** = which option is correct, by question number;
- **result** = what was read from this particular sheet.

Do not put answers in the template. Do not put geometry in the key.

---

## 3. One-time setup

Run every command from the project root (the folder containing `README.md`,
`omr/`, `scripts/`, `templates/`, and `keys/`).

### 3.1 Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Windows `cmd.exe`:

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Confirm:

```powershell
python --version
python -m pip --version
```

### 3.2 Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The runtime needs:

- `opencv-python` — image operations and the live window;
- `numpy` — arrays;
- `scikit-learn` — the two-component GMM calibration.

`pymupdf` is listed because it is used by the **template compiler**. It is not
needed to scan a sheet. If your Windows Application Control policy blocks
PyMuPDF's DLL, the scanner still runs; only `compile_template.py` needs it.
See [12.1](#121-pymupdf-is-blocked-by-windows).

### 3.3 Make sure OpenCV can open a window

The live tester needs the normal GUI build, not the headless build.

```powershell
python -c "import cv2; cv2.namedWindow('opencv check'); cv2.destroyAllWindows(); print('GUI OK')"
```

If that raises an error, run:

```powershell
python -m pip uninstall -y opencv-python-headless
python -m pip install --upgrade opencv-python
```

Re-run the check. You should see `GUI OK` and a window may flash briefly.

### 3.4 Check the template and key exist

```powershell
Test-Path templates\q-paper.json
Test-Path keys\sample_key.json
```

Both should print `True`.

The supplied `templates/q-paper.json` is already compiled. **You do not need to
run `compile_template.py` for the supplied sheet.**

---

## 4. Ten-minute smoke test

This verifies the whole still-image path before involving a camera.

### Step 1 — grade the flat sample

```powershell
python scripts/read_image.py "code try/q-paper_page-0001.jpg" --template templates/q-paper.json --key keys/sample_key.json --out results/smoke_flat --debug
```

Expected:

```text
warp: full_frame (confidence 0.90) -> 1433x2024
total: 29/30  correct=29 wrong=1 blank=0 multi=0 review=0
```

The one wrong answer is Q1: the sample marks **ক** (option 0), while the key
expects **খ** (option 1). That is intentional.

Open:

- `results/smoke_flat/result.jpg` — green/red overlay and total;
- `results/smoke_flat/audit_canonical.png` — the upright, warped sheet;
- `results/smoke_flat/result.json` — machine-readable answers.

### Step 2 — grade the angled sample

```powershell
python scripts/read_image.py samples/q-paper_tilted.jpg --template templates/q-paper.json --key keys/sample_key.json --out results/smoke_tilted --debug
```

Expected: `page_quad`, **29/30**.

### Step 3 — grade the blank control

```powershell
python scripts/read_image.py samples/q-paper_blank.png --template templates/q-paper.json --key keys/sample_key.json --out results/smoke_blank
```

Expected:

```text
total: 0/30  correct=0 wrong=0 blank=30 multi=0 review=0
```

If the blank sheet shows marks, stop and investigate before using real sheets.

### Step 4 — run the full acceptance test

```powershell
python scripts/selftest.py
```

Expected tail:

```text
PASS: all M0 checks
```

`selftest.py` is hard-coded to the supplied `q-paper` sheet and its samples.
It is a regression test for this template, not a generic test for a redesigned
sheet. See [7.1](#71-run-the-built-in-acceptance-test).

### Step 5 — check the camera without a sheet

```powershell
python scripts/live_camera.py --list
```

It probes camera indices 0–5. A working laptop camera should print something
like:

```text
camera: camera 0 -> 1280x720 @ 30 fps
  0: OK  1280x720
```

If no camera is found, fix that before trying the live window.

### Step 6 — run the live window

```powershell
python scripts/live_camera.py
```

Hold the printed sheet in front of the camera. Wait for the green
`READY - PRESS SPACE`, then press `SPACE`. The result should match the still
image for the same answers.

---

## 5. Running the scanner

### 5.1 One still image

```powershell
python scripts/read_image.py "path\to\sheet.jpg" --template templates/q-paper.json --key keys/sample_key.json --out results\my_scan --debug
```

Useful variations:

```powershell
# Exact flat scan: treat the whole image as the page, no page/frame search
python scripts/read_image.py "scan.png" --no-detect --out results\flat

# Force frame/page detection even if the image looks like a flat scan
python scripts/read_image.py "scan.png" --force-detect --out results\forced

# Supply the four full-page corners manually: TL, TR, BR, BL
python scripts/read_image.py "photo.jpg" --corners "40,30,1180,20,1200,1580,25,1600" --out results\manual

# Read at 300 DPI instead of the template's 200 DPI
python scripts/read_image.py "photo.jpg" --dpi 300 --out results\highdpi
```

`--out` behaviour:

- a directory (or no extension) → writes `result.jpg`, `result.json`,
  `audit_canonical.png` inside it;
- a path ending in `.jpg`, `.jpeg`, or `.png` → writes the result image there
  and puts `result.json` and `audit_canonical.png` beside it.

`--debug` additionally writes `debug_warped.png`.

The default output directory is `results/<input-file-stem>`.

### 5.2 Laptop webcam

```powershell
python scripts/live_camera.py
```

Useful variations:

```powershell
# Second camera / USB camera
python scripts/live_camera.py --source 1

# Lower resolution if the laptop struggles
python scripts/live_camera.py --width 1280 --height 720

# Start with automatic capture armed
python scripts/live_camera.py --auto

# Probe indices 0..5 and exit
python scripts/live_camera.py --list
```

While live:

| Key | Action |
|---|---|
| `SPACE` | capture the current frame, grade it, show the result |
| `A` | arm/disarm automatic capture |
| `S` | save the current raw camera frame |
| `Q` / `ESC` | quit |

On the result screen:

| Key | Action |
|---|---|
| `SPACE` | return to live for the next sheet |
| `S` | save a copy of the result image |
| `Q` / `ESC` | quit |

Auto-capture disarms itself after one capture. Press `A` again for the next
sheet, or use `SPACE`.

### 5.3 Phone as the camera

Android, using the **IP Webcam** app:

1. Install IP Webcam and open it.
2. Connect the phone and laptop to the same Wi-Fi network.
3. In the app, start the server (usually at the bottom).
4. Note the IP address shown, for example `192.168.1.45`; the default port is
   `8080`.
5. Test the stream in the laptop browser first:

   ```text
   http://192.168.1.45:8080/video
   ```

   You should see the phone's camera feed. If the browser cannot open it, the
   scanner cannot either.

6. Run:

```powershell
python scripts/live_camera.py --source "http://192.168.1.45:8080/video"
```

Replace the IP with the one shown in **your** app. Do not use the example
`192.168.0.23` unless that is actually your phone.

DroidCam commonly exposes:

```text
http://192.168.1.45:4747/video
```

USB phone cameras (DroidCam, Irium, Camo) often appear as a normal camera
index. Probe with `--list`, then use `--source 1` (or whichever index appears).

Checklist if the phone stream fails:

- phone and laptop are on the same Wi-Fi, not guest Wi-Fi with client
  isolation;
- the IP is current (it can change after reconnecting);
- the port is the one shown in the app;
- the phone's server is running and the screen is awake;
- Windows Firewall allowed Python on private networks;
- VPN is off;
- the URL works in a browser on the laptop.

### 5.4 Capture output and filenames

Each live capture gets a timestamp directory:

```text
results/live/20261030_153012_482/
  capture.jpg            raw camera frame
  result.jpg             canonical sheet + overlay + total
  result.json            warp, calibration, answers, grade
  audit_canonical.png    upright warped sheet, no overlay
```

Extra saves:

```text
results/live/snapshots/snapshot_<timestamp>.jpg   # S while live
results/live/saved/result_<timestamp>.jpg         # S on result screen
```

Use `capture.jpg` and `audit_canonical.png` together when diagnosing a bad
read: the first shows what the camera saw, the second shows what the aligner
produced.

### 5.5 Run the live capture path without a camera

```powershell
python scripts/live_camera.py --check "samples/test-new/Asset 1@4x.png"
```

This runs the same `process_capture()` used by `SPACE`, including
`force_detect=True`, then writes under `results/live/<timestamp>/`.

Expected:

```text
quality: HOLD STEADY  ... (a single still has no steadiness history)
warp: frame_quad (confidence 0.98) -> 1433x2024
total: 29/30  correct=29 wrong=1 blank=0 multi=0 review=0
```

`HOLD STEADY` is expected for a single still; it is not a failure.

### 5.6 Continuous voice scanner (many sheets)

For checking a stack of sheets without pressing `SPACE` for every one:

```powershell
python scripts/live_camera_with_audio.py
```

This is a separate script; it does not replace `live_camera.py`. It keeps the
camera preview running, grades each new sheet in a background worker, and by
default projects the result onto the physical sheet (AR overlay): the total
inside the printed mark box plus green circles for correct, red for wrong and
blue for the key answer. The total is also drawn as a badge on the live sheet,
and the voice speaks it once. Each result is saved under
`results/live_audio/<timestamp>/` with the same files as the normal live
tester:

```text
results/live_audio/<timestamp>/
  capture.jpg
  result.jpg
  result.json
  audit_canonical.png
```

| Key | Action |
|---|---|
| `SPACE` | scan the current sheet immediately; on the graded view, return to live |
| `P` | pause / resume automatic scanning |
| `V` | mute / unmute voice |
| `S` | save the raw camera frame |
| `Q` / `ESC` | quit |

The AR overlay is **on by default**; no flags are needed. To use the old
separate warped graded-sheet view instead:

```powershell
python scripts/live_camera_with_audio.py --no-ar --result-hold 2.5
```

`--result-hold` is the number of seconds that separate view stays on screen
(default **0**, so it is skipped). `--no-ar` turns the overlay off. Press
`SPACE` on the graded view to return to the live camera immediately; press
`SPACE` again if you want to rescan the same sheet.

#### AR overlay on the real sheet

The script maps each graded bubble and the total-marks region back through the
inverse homography and draws them on the **live camera frame**. Green = correct,
red = wrong, blue = key answer, purple = multi, orange = blank, yellow =
review; the total is drawn inside the printed mark box. The homography is
re-estimated from the current frame at the preview interval, so the overlay
follows the sheet as it moves.

This is a 2D projective overlay, not 3D AR. It works best when the printed frame
or page outline is visible and the sheet is reasonably flat. If the sheet is
lost, the overlay is hidden rather than placed incorrectly. Strong perspective,
glare, or lens distortion can reduce the precision of the circles and the
text; `result.jpg` is still the authoritative saved image.

If `--result-hold` is greater than zero, the separate graded view is shown
first and the AR overlay appears after that hold. This also works with the
default AR mode: `--result-hold 2.5` alone adds the graded view in front of the
overlay.

After a successful result the scanner is **disarmed**. It re-arms when the
sheet has been absent for about `--absence-reset` seconds, or when the detected
sheet moves by more than `--shift-reset` (a new sheet has been placed). The same
sheet left in front of the camera is not scanned again unless you press
`SPACE`. This is intentional: it prevents the voice from repeating the same
mark while the teacher reaches for the next sheet.

Phone camera:

```powershell
python scripts/live_camera_with_audio.py --source "http://192.168.1.45:8080/video"
```

Quiet mode and voice test:

```powershell
python scripts/live_camera_with_audio.py --no-voice
python scripts/live_camera_with_audio.py --test-voice
```

Voice backends:

- **Windows:** `System.Speech` through Windows PowerShell; no extra package.
- **macOS:** the built-in `say` command.
- **Linux:** `espeak` or `spd-say` if installed.
- If no voice engine is available, the mark is still displayed and saved; the
  scanner prints the phrase and continues.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--source` | `0` | camera index, video path, or phone URL |
| `--out` | `results/live_audio` | capture root |
| `--min-blur` | `45.0` | same focus gate as the normal live tester |
| `--detect-interval` | `0.20` | preview geometry/quality check interval |
| `--min-interval` | `1.0` | minimum seconds between automatic scans |
| `--absence-reset` | `0.45` | seconds without a sheet before re-arming |
| `--shift-reset` | `0.08` | normalised corner movement treated as a new sheet |
| `--result-hold` | `0` | seconds to show the separate graded sheet; `0` disables |
| `--ar` / `--no-ar` | on | project marks back onto the live sheet |
| `--no-voice` | off | disable speech |
| `--test-voice` | off | speak one sample phrase and exit |
| `--start-paused` | off | open the camera but wait for `P`/`SPACE` |
| `--list` | off | probe camera indices 0..5 and exit |

If the teacher leaves a sheet in place and wants it re-read after correcting a
mark, press `SPACE`; the badge changes to `READING...` and the new total is
announced when the background worker finishes.

### 5.7 Batch still images (optional, not a built-in command)

There is no still-image batch command yet. For a folder of still scans, a
PowerShell loop is the simplest workaround:

```powershell
Get-ChildItem "scans\*.jpg" | ForEach-Object {
  python scripts/read_image.py $_.FullName --template templates/q-paper.json --key keys/sample_key.json --out ("results\" + $_.BaseName)
}
```

This is a convenience loop, not a supported batch feature. Check each result;
the loop does not stop on the first failure. CSV export is not implemented.

---

## 6. Command reference

### 6.1 `scripts/read_image.py`

```text
python scripts/read_image.py IMAGE [options]
```

| Option | Default | Meaning |
|---|---|---|
| `IMAGE` | required | still image path |
| `--template` | `templates/q-paper.json` | compiled template |
| `--key` | `keys/sample_key.json` | teacher key |
| `--out` | `results/<stem>` | output directory or image path |
| `--corners` | none | manual full-page corners `x1,y1,...,x4,y4` in TL,TR,BR,BL order |
| `--no-detect` | off | treat the whole image as an already-aligned page |
| `--force-detect` | off | run frame/page detection even on a flat scan |
| `--dpi` | template value (200) | canonical raster DPI |
| `--debug` | off | also save `debug_warped.png` |

`--corners` maps the points to the **full page**, not to the printed frame.
Use it only when automatic alignment fails and you can identify the four page
corners.

### 6.2 `scripts/live_camera.py`

```text
python scripts/live_camera.py [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--source` | `0` | camera index, video path, or phone MJPEG URL |
| `--template` | `templates/q-paper.json` | compiled template |
| `--key` | `keys/sample_key.json` | teacher key |
| `--out` | `results/live` | capture root |
| `--width` | `1920` | requested capture width |
| `--height` | `1080` | requested capture height |
| `--fps` | `30` | requested capture FPS |
| `--min-blur` | `45.0` | Laplacian-variance threshold measured at 960 px |
| `--detect-interval` | `0.20` | seconds between preview geometry/quality checks |
| `--auto` | off | start with auto-capture armed |
| `--check IMAGE` | none | grade one still through the live capture path and exit |
| `--list` | off | probe camera indices 0..5 and exit |

The live app needs a GUI OpenCV build. `--check` and `--list` do not need a
window, but `--list` does need camera access.

### 6.3 `scripts/compile_template.py`

```text
python scripts/compile_template.py PDF [options]
```

| Option | Default | Meaning |
|---|---|---|
| `PDF` | required | vector OMR sheet PDF |
| `--out` | `templates/<pdf-stem>.json` | output template |
| `--id` | PDF stem | template id; must match the key |
| `--dpi` | `200` | canonical reading DPI |
| `--no-reference` | off | skip the reference PNG render |

Run this only when the sheet design or geometry changes. The compiler needs a
working PyMuPDF; the scanner does not.

### 6.4 `scripts/selftest.py`

```text
python scripts/selftest.py
```

No options. It uses the fixed `q-paper` template, `sample_key.json`, and the
sample images in `samples/` and `samples/test-new/`. Exit code `0` means pass;
`1` means at least one check failed.

### 6.5 `scripts/live_camera_with_audio.py`

```text
python scripts/live_camera_with_audio.py [options]
```

Continuous version of the live tester. It uses the same alignment, reader and
`process_capture()` path, but runs grading in a background thread so the
preview keeps moving. The full option table is in
[5.6](#56-continuous-voice-scanner-many-sheets).

---

## 7. How to check the scanner

### 7.1 Run the built-in acceptance test

```powershell
python scripts/selftest.py
```

Expected current output:

```text
flat:   mode=full_frame total=29 correct=29 wrong=1
tilted: mode=page_quad conf=0.83 total=29 correct=29
blank:  total=0 blank=30 marked_bubbles=0
mixed:  states={1: 'ANSWERED', 2: 'BLANK', 3: 'MULTI', 4: 'REVIEW', 5: 'ANSWERED'} review=[4]
Asset 1@4x.png     mode=frame_quad total=29 correct=29 wrong=1 answered=30
Asset 2@4x.png     mode=frame_quad total=29 correct=29 wrong=1 answered=30
Asset 3@4x.png     mode=frame_quad total=29 correct=29 wrong=1 answered=30
Asset 4@4x.png     mode=frame_quad total=29 correct=29 wrong=1 answered=30
orientation 0deg    mode=frame_quad k=0 total=29 answered=30
orientation 90ccw   mode=frame_quad k=1 total=29 answered=30
orientation 180deg  mode=frame_quad k=2 total=29 answered=30
orientation 90cw    mode=frame_quad k=3 total=29 answered=30

PASS: all M0 checks
```

What each check proves:

| Check | Proves |
|---|---|
| flat | flat-scan path and key/total |
| tilted | page-edge perspective warp |
| blank | no false positives on an empty sheet |
| mixed | blank / multi / review states |
| Asset 1–4 | printed-frame warp on rotated, skewed, cropped captures |
| orientation 0/90/180/270 | frame orientation via the three pink panel boxes |

If `selftest.py` fails, do not change the expected numbers to make it pass.
Fix the regression and rerun.

### 7.2 Check one image through the live path

```powershell
python scripts/live_camera.py --check "samples/test-new/Asset 1@4x.png"
```

This is the quickest check that the camera **capture → grade** half works
without opening a camera. It uses `force_detect=True`, so it prefers the
printed frame over the full-frame path.

### 7.3 Check the camera hardware

```powershell
python scripts/live_camera.py --list
```

This prints each available index. Close other apps that may be using the
camera. If a camera opens but returns no frame, try another index or a lower
resolution:

```powershell
python scripts/live_camera.py --source 1 --width 1280 --height 720
```

### 7.4 Check alignment visually

Always inspect these two files:

1. `audit_canonical.png` — the warped sheet, no overlay.
   - Is the sheet upright?
   - Is the whole answer grid visible?
   - Are the bubbles circular, not stretched into ovals?
   - Is the page cropped?
2. `result.jpg` — the overlay.
   - Do the circles sit on the printed bubbles?
   - Is the total in the marks panel?
   - Are wrong/blank/review markers on the right questions?

If `audit_canonical.png` is wrong, the problem is alignment, not the answer
key. If it is right but the overlay is offset, the problem is the template
coordinates or the wrong template. If both are right but the total is wrong,
check the key.

### 7.5 Check the JSON

Replace the path with your result directory:

```powershell
python -c "import json; d=json.load(open('results/smoke_flat/result.json', encoding='utf-8')); print('warp', d['warp']['mode'], d['warp']['page_confidence']); print('cal', d['calibration']); print('grade', {k:d['grade'][k] for k in ('total','maximum','correct','wrong','blank','multi','review')})"
```

Print every question's detected state:

```powershell
python -c "import json; d=json.load(open('results/smoke_flat/result.json', encoding='utf-8')); [print(q, a['state'], a['selected'], a['reason']) for q,a in d['answers'].items()]"
```

Check the template's page, grid, frame, and first question:

```powershell
python -c "import json; t=json.load(open('templates/q-paper.json', encoding='utf-8')); print('page', t['page']); print('grid', t['grid']); print('frame', t['frame']); print('regions', t['regions']); print('q1', t['questions'][0])"
```

### 7.6 Check a real printed sheet

Use this order every time you change paper, printer, or design:

1. **Blank sheet** — expect `0`, `blank = number of questions`, no marked
   bubbles.
2. **Known filled sheet** — mark answers from a known list with a dark pen;
   expect the exact total and zero `review`.
3. **Rotate the known sheet 90°/180°/270°** — with the printed frame visible,
   the reader should still grade it correctly. If it does not, check the
   `frame_orientation` value and the panel regions.
4. **Tilt the sheet** — hold it at roughly 20–30°; the frame path should
   produce an upright `audit_canonical.png`.
5. **Crop outside the frame** — the frame path should still work as long as
   all four frame edges are visible.
6. **Live camera** — repeat with the actual camera, lighting, and distance you
   will use in production.

For a production pilot, keep a human review step for every `REVIEW` outcome
and spot-check `WRONG` outcomes against the scanned image.

---

## 8. Understanding the output

### 8.1 `result.jpg`

The canonical sheet with:

- **green** circle = selected answer is correct;
- **red** circle = selected answer is wrong;
- **blue** circle = the key answer when the student was wrong;
- **orange `?`** = blank;
- **yellow `!`** = review / uncertain / unobserved;
- **purple `M`** = more than one selected;
- total written in the printed total-marks panel;
- footer with correct/wrong/blank/review counts.

If the total-marks region is wrong for a new design, the number can overlap the
answer area. Update `template.regions.total_marks` or remove that region; see
[11.6](#116-regions-frame-and-orientation).

### 8.2 `result.json`

Still-image result (the live result adds `source`, `captured_at`, and
`image_size`):

```json
{
  "image": "code try/q-paper_page-0001.jpg",
  "template_id": "q-paper",
  "warp": {
    "mode": "full_frame",
    "page_confidence": 0.9,
    "corners": [[0, 0], [1433, 0], [1433, 2024], [0, 2024]],
    "debug": {
      "source_size": [1433, 2024],
      "warped_size": [1433, 2024]
    }
  },
  "calibration": {
    "method": "gmm",
    "threshold": 0.13,
    "separation": 15.5,
    "mu_empty": 0.02,
    "mu_marked": 0.62,
    "marked_count": 30
  },
  "diagnostics": {
    "mean_ink": 12.4,
    "glare_fraction": 0.0,
    "ring_baseline": 0.25,
    "score_min": 0.0,
    "score_max": 0.8
  },
  "answers": {
    "1": {"selected": [0], "state": "ANSWERED", "confidence": 0.99, "reason": ""}
  },
  "grade": {
    "total": 29,
    "maximum": 30,
    "percentage": 96.67,
    "correct": 29,
    "wrong": 1,
    "blank": 0,
    "multi": 0,
    "review": 0,
    "review_questions": [],
    "questions": []
  }
}
```

Only the first question is shown above; the real file contains all questions.
The numeric calibration values vary per image; do not treat the example values
as expected thresholds.

### 8.3 `audit_canonical.png` and `debug_warped.png`

- `audit_canonical.png` is always written. It is the exact warped image the
  reader measured.
- `debug_warped.png` is written only with `--debug` on `read_image.py`; it is
  the same kind of image, kept as an explicit debugging artifact.

If the audit image is not upright and complete, the answer data is not
trustworthy regardless of the total.

### 8.4 Warp modes

| `warp.mode` | Meaning | When it is used |
|---|---|---|
| `frame_quad` | printed pink frame corners mapped to the template frame | preferred for camera captures; works on white backgrounds and cropped images |
| `page_quad` | page outline mapped to the full page | frame not found; needs a visible page edge, usually a dark background |
| `full_frame` | whole image treated as the page | flat scans and fallback |
| `manual` | four user-supplied full-page corners | `--corners` |

`page_confidence` is 0–0.99. Frame captures are usually 0.98–0.99; page
captures around 0.8; a flat scan that skips detection is 0.90; a complete
fallback with no detection is 0.0.

`warp.debug` can contain:

- `frame_orientation` — `0` upright, `1` corrected a 90° CCW input, `2` 180°,
  `3` 90° CW;
- `panel_distance` — how far the pink panel landed from the expected marks
  panel; lower is better; `>= 1.5` caps confidence at 0.75;
- `source_size` and `warped_size`.

### 8.5 Calibration fields

- `method: "gmm"` — the sheet's own 120 scores separated into empty/marked
  components. Preferred.
- `method: "fallback"` — too few distinct scores or a weak separation; a fixed
  threshold based on the ring baseline is used.
- `separation` — distance between the two component means in standard
  deviations. Higher is better. Values below 2.0 cause the GMM to be rejected.
- `threshold` — score boundary used for the final decisions.
- `marked_count` — number of bubbles classified as marked by the calibration.
- `gmm_rejected` / `gmm_error` — why fallback was used.

### 8.6 Question and grade states

Bubble level (`answers[q].state`):

| State | Meaning |
|---|---|
| `ANSWERED` | exactly one marked bubble, no uncertainty |
| `BLANK` | no marked and no uncertain bubbles |
| `MULTI` | more than one marked bubble |
| `REVIEW` | at least one uncertain bubble, but no clear multi/blank |
| `UNOBSERVED` | glare/occlusion covers at least one bubble |

Grade level (`grade.questions[].outcome`):

| Outcome | Meaning | Default award |
|---|---|---|
| `CORRECT` | one selected option equals the key | full marks |
| `WRONG` | answered with a different option | 0, or negative if configured |
| `BLANK` | no answer | 0 |
| `MULTI` | multiple options | 0 or negative; `partial` policy changes this |
| `REVIEW` | uncertain or unobserved | 0 |

`grade.total` is the sum of awarded marks; `grade.maximum` is the sum of the
key's `marks`; `percentage` is `total / maximum`.

---

## 9. How grading works

### 9.1 Alignment

`omr/geometry.py` tries, in order:

1. **Printed frame** — the large pink rectangle around the answer grid and
   marks panel. Its four corners map exactly to the frame coordinates stored
   in the template. This survives white-on-white renders, cropping outside the
   frame, rotation and moderate perspective.
2. **Page outline** — the largest convex quadrilateral matching the page
   aspect ratio. This needs a visible page edge, so a dark table/mat helps.
3. **Full frame** — the whole image is assumed to be the page. This is correct
   for a flatbed scan and wrong for a camera photo of a sheet on a desk.
4. **Manual corners** — your four page corners, when automatic detection fails.

The flat-scan guard is deliberate: a page-shaped image with a white border is
treated as already aligned, because its anti-aliased printed frame can bias the
quad by a pixel or two.

### 9.2 Orientation

In `frame_quad` mode, the three solid pink boxes in the marks panel decide
which corner assignment is upright. The reader can correct 0°, 90°, 180° and
270° automatically. `selftest.py` exercises all four.

`page_quad` and `full_frame` modes have no orientation signal. If a sheet is
sideways in those modes, rotate the image before scanning or fix the camera
orientation.

### 9.3 Bubble measurement

For each bubble:

- convert to RGB;
- compute ink as `255 - min(R,G,B)`, which suppresses pink print;
- estimate and subtract local paper brightness;
- measure an inner disk (`0.52 × radius`) and a ring annulus
  (`0.85–1.15 × radius`);
- combine fill, dark fraction and ring contrast into a score;
- flag specular glare;
- fit a two-component GMM to **that sheet's** scores;
- mark, leave empty, or flag uncertain/unobserved.

No single global pixel threshold is used. This is why faint pencil and
different lighting can still work, but it is also why a badly aligned or
heavily glared image can produce `REVIEW` rather than a confident guess.

### 9.4 Grading

`omr/grade.py` loads the key, checks that it contains every question, applies
`marks`, `negative_per_wrong`, and the multi-answer policy, then totals the
awarded marks. The key's `template_id` must match the template's
`template_id`; a mismatch is an error, not a warning.

---

## 10. Keys, marks, and policies

### 10.1 Create a new key

Use the key wizard. It reads the template, asks for the correct option of each
question and the marks, validates everything, and writes the key file:

```powershell
python scripts/make_key.py --out keys\class8_bangla.json
```

What it asks for:

1. **Output file** — where to save the key (default `keys/new_key.json`).
2. **Key ID, subject, set** — for your records; the set is metadata only.
3. **One answer per question** — `1`-`4` or the printed `ক`/`খ`/`গ`/`ঘ`.
   Enter keeps the existing answer when editing a key; `p` goes back one
   question; `q` cancels without writing anything.
4. **Marks** — one value for every question, then an optional line of
   per-question overrides such as `3=2,7=5`.
5. **Save** — answers and marks are checked against the template before and
   after writing.

You can paste the whole answer list at the first question, for example
`2,1,3,4,1,2,...`, instead of typing 30 answers one by one.

Bulk entry without prompts (useful when the answers are already in a list):

```powershell
python scripts/make_key.py --out keys\class8_math.json `
  --answers "2,1,3,4,1,2,3,4,..." --marks 1

# Different marks per question: one value, or one value per question
python scripts/make_key.py --out keys\class8_math.json `
  --answers "2,1,3,4,..." --marks "1,1,2,1,..."
```

Edit an existing key (answers, marks, and metadata are preloaded):

```powershell
python scripts/make_key.py --from keys\class8_bangla.json `
  --out keys\class8_bangla.json
```

Add `--negative 0.25` for negative marking and `--multi partial` (or
`review`) to change the multiple-mark policy. `--dry-run` validates and prints
the key without writing it; `--force` replaces an existing file in bulk mode.

A new key always covers every question in the template. The tool refuses an
out-of-range answer, a missing question, a non-numeric mark, or a mismatched
`template_id`.

#### Manual editing

You can still copy the supplied key and edit it by hand:

```powershell
Copy-Item keys\sample_key.json keys\my_key.json
```

Then edit `keys/my_key.json`:

```json
{
  "key_id": "class-8-bangla-set-a",
  "template_id": "q-paper",
  "subject": "বাংলা",
  "set": "A",
  "questions": {
    "1": { "answer": 1, "marks": 1 },
    "2": { "answer": 0, "marks": 1 },
    "3": { "answer": 2, "marks": 1 }
  },
  "policy": {
    "multi": "wrong",
    "blank": 0,
    "negative_per_wrong": 0.0,
    "partial_multi": 0.0
  }
}
```

Requirements:

- `template_id` must exactly equal the template's id;
- every question from `1` through the template's question count must be
  present;
- `answer` is the **zero-based** option index, not the printed label;
- `marks` defaults to `1` if omitted;
- `key_id` is for your records; it is not used by the grading math.

### 10.2 Answer indexing

For the supplied four-option sheet:

| Printed label | `answer` value |
|---|---|
| ক | `0` |
| খ | `1` |
| গ | `2` |
| ঘ | `3` |

If a future sheet has a fifth option, the compiler supports label `ঙ` and
index `4`, but validate that layout explicitly.

### 10.3 Policy fields — what actually works

| Field | Current behaviour |
|---|---|
| `negative_per_wrong` | positive number; a wrong answer awards `-negative_per_wrong × marks` |
| `multi: "wrong"` | multiple marks are `MULTI`; award 0 or the negative penalty |
| `multi: "partial"` | if the key answer is among the selected options, award `marks / number_selected`; otherwise 0 or negative |
| `multi: "review"` | multiple marks become `REVIEW`, award 0 |
| `blank` | **reserved; currently ignored** — blanks always award 0 |
| `partial_multi` | **reserved; currently ignored** — the partial calculation uses `marks / number_selected` |

Do not assume `blank` or `partial_multi` will change the total. They are
present in the sample key but not applied by the current `grade.py`.

### 10.4 Sets A and B

There is **no automatic set detection**. The `set` field is metadata only.
If the same layout has different question orders or answer keys for different
sets, create one key per set and pass the correct one:

```powershell
python scripts/live_camera.py --key keys\q-paper_set_A.json
python scripts/live_camera.py --key keys\q-paper_set_B.json
```

### 10.5 Marks and totals

- `marks` may differ per question;
- `maximum` is the sum of all key marks;
- the total can be negative if `negative_per_wrong` is used;
- the overlay displays an integer when the total is whole, otherwise one
  decimal place.

---

## 11. Changing the OMR sheet design

This is the most important section for maintaining the project. Change one
layer at a time: **PDF → template → key → validation**.

### 11.1 Decide what kind of change it is

| Change | New template? | New key? | Code change? |
|---|---|---|---|
| Logo, title, instructions, colors that do not move bubbles | Usually no; validate first | No | No |
| Move the bubble grid or resize the page | Yes | No, if question order is unchanged | No |
| Change the printed frame position | Yes, or edit `frame` | No | No |
| Move the marks panel to the left/top | Yes, or edit `regions` | No | Update `regions` |
| Change the frame color from pink to black/blue | No, but frame detection will fail | No | Yes, `_pink_mask` in `omr/geometry.py` |
| Change bubble diameter/shape (not 17 pt circles) | Compiler will miss them | No | Yes, compiler filter |
| Add/remove questions | Yes | Yes | Update `selftest.py` if replacing the sample |
| Change option count (4 → 5, etc.) | Yes | Yes | Maybe; compiler supports up to 5 labels |
| Change number of answer bands (2 → 3) | Compiler fails | Yes | Yes |
| Change question numbering/order | Yes, and verify `q` mapping | Yes | Maybe |
| Change from portrait to landscape | Yes | No | Validate; regions may need editing |
| Add QR/fiducials/student-ID OCR | Not used by M0 | No | Future feature |
| Different answer sets (A/B) | No | Yes, one key per set | No |
| Negative marking or different marks | No | Yes | No |

Rule of thumb: **if the PDF's bubble centres, frame, page size, or question
order changed, do not reuse the old template or key without validating.**

### 11.2 Back up the working version first

```powershell
Copy-Item templates\q-paper.json templates\q-paper_v1.json
Copy-Item keys\sample_key.json keys\q-paper_v1_key.json
```

Do not overwrite the working template until the new one has passed the checks.
Keep old templates and keys for old print runs.

### 11.3 Compile the new PDF

```powershell
python scripts/compile_template.py "design\new-sheet.pdf" --out templates\new-sheet.json --id new-sheet --dpi 200
```

Notes:

- run from the project root;
- use a new `--id` for a new design (`new-sheet`, `q-paper-v2`, etc.);
- `--dpi 200` is the tested default; 300 is acceptable but slower; below 150
  may lose small bubbles;
- the compiler writes `templates/new-sheet_reference.png` unless you pass
  `--no-reference` (not recommended);
- the compiler uses **page 1 only**;
- `pymupdf` must work for this command; if your policy blocks it, use an
  allowed environment or an existing template. Scanning itself does not need
  it.

### 11.4 Inspect the compiled template

Open `templates/new-sheet_reference.png`. It should show the sheet with the
bubble grid intact, at the requested DPI.

Inspect the JSON:

```powershell
python -c "import json; t=json.load(open('templates/new-sheet.json', encoding='utf-8')); print('page', t['page']); print('grid', t['grid']); print('frame', t['frame']); print('regions', t['regions']); print('q1', t['questions'][0])"
```

Check:

- `page.width_mm` / `height_mm` match the printed page;
- `grid.columns` is the number of question columns in one band;
- `grid.options` is the number of options per question;
- `grid.bands` is `2`;
- `grid.questions == columns × 2`;
- `frame` is not `null` if the sheet has a printed frame;
- `questions[0].q` is `1` and its options are in top-to-bottom order;
- `x_mm`/`y_mm` values are inside the page.

If the compiler's question numbering does not match the printed numbering, do
not use the output as-is. See [11.5](#115-what-the-compiler-assumes).

### 11.5 What the compiler assumes

`scripts/compile_template.py` → `omr/template.py:compile_template()` currently
assumes all of the following:

1. **One page** — it reads `doc[0]`.
2. **Stroked circles** — each bubble is a separate drawing whose bounding box
   is between **16 and 18 points** wide/high, nearly square
   (`abs(width - height) <= 1.0`), and has a stroke color.
3. **A regular grid** — x and y centres cluster within 4 points.
4. **Exactly two answer bands** — the largest vertical gap splits the rows into
   two groups.
5. **Equal option counts** — both bands have the same number of rows.
6. **At most five options** — labels are ক, খ, গ, ঘ, ঙ.
7. **Band-major numbering** — question `q = band × columns + column + 1`:
   - top band: Q1..Q<columns> left-to-right;
   - bottom band: Q<columns+1>..Q<2×columns> left-to-right.
8. **A large printed frame** — a stroked rectangle with a near-white fill,
   at least 50% of page width and 20% of page height, but not the full page.
   It is used for the preferred `frame_quad` alignment.
9. **Default regions** — the marks panel is assumed near the right side of the
   page, at the proportional positions in `default_regions()`. These are used
   by the orientation resolver.

If any assumption is false, the compiler may fail loudly or produce a template
that looks plausible but reads the wrong bubbles. Always inspect the reference
image and test a known sheet.

### 11.6 Regions, frame, and orientation

The template's `regions` are millimetres from the top-left of the page:

```json
"regions": {
  "student_id":  {"x_mm": 134.0, "y_mm": 38.0, "w_mm": 44.0, "h_mm": 20.0},
  "subject_code":{"x_mm": 134.0, "y_mm": 66.0, "w_mm": 44.0, "h_mm": 18.0},
  "total_marks": {"x_mm": 134.0, "y_mm": 94.0, "w_mm": 44.0, "h_mm": 30.0}
}
```

- `total_marks` is the only region currently used by the overlay. The other
  two are reserved for future OCR.
- `_choose_frame_orientation()` uses the centres of **all** regions to decide
  which side is the marks panel. If the panel moves, update these rectangles.
- If `regions` is empty, orientation falls back to `0` (upright).
- If the total-marks box is wrong, the total may be drawn over the bubbles.
  Fix or remove `total_marks`; the grade JSON is unaffected.

Frame format:

```json
"frame": {"x0_mm": 5.0, "y0_mm": 51.9, "x1_mm": 177.6, "y1_mm": 132.0}
```

- coordinates are the outer frame's top-left and bottom-right in mm;
- the compiler finds a large white-filled stroked rectangle automatically;
- if the compiler misses the frame, you can add it manually, but
  `find_frame_quad()` still needs the printed frame to be **pink** under the
  current `_pink_mask` thresholds;
- if there is no frame, set `"frame": null`; alignment falls back to the page
  outline/full frame.

How to measure a region from the 200-DPI reference image:

```text
mm = pixels × 25.4 / 200
```

For example, 1000 px from the left = 127.0 mm.

### 11.7 Create the new key

```powershell
Copy-Item keys\sample_key.json keys\new-sheet_key.json
```

Edit `template_id` to the new template id. Replace every `answer` with the
correct zero-based option. Ensure the key contains every question from `1` to
the new template's question count. If the new template has 40 questions, the
key needs 40 entries.

For a quick structural check:

```powershell
python -c "import json; k=json.load(open('keys/new-sheet_key.json', encoding='utf-8')); print('template', k['template_id'], 'questions', len(k['questions']), 'ids', sorted(map(int,k['questions']))[:3], '...', sorted(map(int,k['questions']))[-3:])"
```

The key check is strict about `template_id` and missing questions, but it does
not reject extra questions. Remove extras to avoid confusion.

### 11.8 Test the blank sheet first

Print the new sheet without marks, scan or photograph it, then:

```powershell
python scripts/read_image.py "samples\new-blank.png" --template templates\new-sheet.json --key keys\new-sheet_key.json --out results\new_blank --debug
```

Expected:

- total `0`;
- `blank` equals the number of questions;
- no `marked` bubbles;
- `audit_canonical.png` upright and complete.

If a blank sheet shows marks, do not proceed. Check alignment, print quality,
and the bubble coordinates.

### 11.9 Test a known filled sheet

Mark a known answer list with a dark pen, then read it:

```powershell
python scripts/read_image.py "samples\new-known.png" --template templates\new-sheet.json --key keys\new-sheet_key.json --out results\new_known --debug
```

Check:

- every question is `ANSWERED`;
- `selected` matches your marked options;
- the total equals the expected total;
- `review` is `0`;
- the overlay circles sit on the bubbles.

If the total is right but some selections are wrong, your key is probably
wrong. If the selections are wrong, inspect the template coordinates and the
warp.

### 11.10 Test rotation, tilt, and crop

For a frame-based design, rotate a known still:

```powershell
python -c "import cv2; im=cv2.imread('samples/new-known.png'); cv2.imwrite('samples/new-known-180.png', cv2.rotate(im, cv2.ROTATE_180))"
python scripts/read_image.py "samples/new-known-180.png" --template templates\new-sheet.json --key keys\new-sheet_key.json --out results\new_known_180 --debug
```

Expected: same total, `frame_quad`, and `warp.debug.frame_orientation` equal to
`2` for the 180° case.

Also test:

- 90° CCW (`cv2.ROTATE_90_COUNTERCLOCKWISE`);
- 90° CW (`cv2.ROTATE_90_CLOCKWISE`);
- a tilted photo (about 20–30°);
- a crop that removes paper outside the frame but keeps all four frame edges.

If rotation works only for the still but not live, check that the live app is
receiving the full frame and that the three pink panel boxes are visible.

### 11.11 Test the live path

```powershell
python scripts/live_camera.py --template templates\new-sheet.json --key keys\new-sheet_key.json
```

First run `--check` on a known still:

```powershell
python scripts/live_camera.py --check "samples\new-known.png" --template templates\new-sheet.json --key keys\new-sheet_key.json
```

Then use the real camera and sheet. Expected: green `READY`, capture, correct
total, and an upright `audit_canonical.png`.

### 11.12 If you are replacing the default sheet

The scripts default to `templates/q-paper.json` and
`keys/sample_key.json`. You can either:

- always pass `--template` and `--key` (recommended for multiple designs); or
- change the defaults in `scripts/read_image.py`, `scripts/live_camera.py`, and
  `scripts/selftest.py`.

If you change the defaults, update `selftest.py` expectations to match the new
sheet. Do not leave the old sample assertions in place and call them a test of
the new design.

### 11.13 Validation checklist

Use this before trusting a redesigned sheet:

- [ ] new PDF compiles without errors;
- [ ] reference PNG looks like the printed sheet;
- [ ] `grid.questions == columns × 2`;
- [ ] `questions[0]` is Q1 with options top-to-bottom;
- [ ] `frame` is correct or intentionally `null`;
- [ ] `regions.total_marks` is inside the marks panel;
- [ ] blank sheet → 0 marks, all blank;
- [ ] known sheet → exact expected total, all answered, zero review;
- [ ] 90°/180°/270° known sheet → same total;
- [ ] tilted/cropped known sheet → same total;
- [ ] live `--check` → correct;
- [ ] real camera → green READY, correct total, upright audit image;
- [ ] old template/key still available for old print runs.

### 11.14 If the compiler cannot handle the new design

The current compiler is intentionally narrow. Use this decision guide:

| Symptom / design | What to do |
|---|---|
| `no bubble circles found` | bubbles are not 16–18 pt stroked circles; adjust the PDF or extend the compiler filter |
| `could not form a grid` | bubble centres do not cluster at 4 pt; check the PDF or extend clustering |
| `expected two answer bands` | layout has a different band structure; redesign to two bands or extend the compiler |
| `answer bands have different option counts` | make both bands equal, or extend the compiler |
| `unsupported option count` | more than 5 options; extend `OPTION_LABELS` and validate the reader/key |
| `grid produced N questions, expected M` | numbering/columns mismatch; inspect the grid and q assignment |
| `pymupdf` blocked | compile in an environment where the DLL is allowed; scanning still works with an existing template |
| Frame not found | make the frame a large white-filled stroked rectangle, or add `frame` manually; frame detection still requires pink |
| Orientation wrong | update `regions` to the actual panel position; or ensure the three panel boxes are visible; or rotate input for page/full modes |
| Need QR/fiducials or student-ID OCR | not implemented in M0; this is a future milestone |

A manual template is possible for advanced layouts, but it must match the
loader's schema. The minimum shape is:

```json
{
  "template_id": "custom",
  "page": {"width_mm": 210, "height_mm": 297, "width_pt": 595.28, "height_pt": 841.89},
  "canonical_dpi": 200,
  "grid": {"columns": 2, "options": 4, "bands": 1, "questions": 2},
  "questions": [
    {
      "q": 1,
      "band": 0,
      "column": 0,
      "options": [
        {"label": "ক", "x_mm": 20, "y_mm": 40, "r_mm": 3},
        {"label": "খ", "x_mm": 20, "y_mm": 50, "r_mm": 3},
        {"label": "গ", "x_mm": 20, "y_mm": 60, "r_mm": 3},
        {"label": "ঘ", "x_mm": 20, "y_mm": 70, "r_mm": 3}
      ]
    }
  ],
  "regions": {},
  "frame": null
}
```

The loader assigns `option` from the array order, so keep options sorted
top-to-bottom. Run a blank and a known sheet through the manual template before
using it. Manual templates bypass the compiler's safety checks, so the
validation is entirely on you.

---

## 12. Troubleshooting

### 12.1 PyMuPDF is blocked by Windows

Typical error:

```text
ImportError: DLL load failed while importing _mupdf:
An Application Control policy has blocked this file.
ModuleNotFoundError: No module named 'mupdf'
```

What it means: Windows security policy is blocking PyMuPDF's native DLL.

What to do:

- **Scanning does not need PyMuPDF.** `read_image.py` and `live_camera.py`
  load an existing `templates/*.json` and work without it.
- Only `compile_template.py` needs PyMuPDF. Run the compiler on a machine or
  environment where the signed DLL is allowed, or ask IT for an exception.
- Do not disable security policy just to compile; use an approved environment.
- If you run the compiler while it is blocked, the current code raises a clear
  `RuntimeError` explaining that it is compile-only.

### 12.2 OpenCV cannot open a window

Typical error:

```text
error: this OpenCV build has no GUI support (opencv-python-headless).
```

Fix:

```powershell
python -m pip uninstall -y opencv-python-headless
python -m pip install --upgrade opencv-python
```

`--check` and `read_image.py` can work without a window; the live preview
cannot.

### 12.3 Camera not found

```text
error: could not open '0'
```

Try:

```powershell
python scripts/live_camera.py --list
python scripts/live_camera.py --source 1
```

Close other apps using the camera. Check Windows camera privacy settings. For a
USB camera, reconnect it and probe again. For a phone, use the stream URL and
test it in a browser first.

### 12.4 Camera opens but returns no frames

```text
error: camera opened but returned no frames
```

Another app may hold the camera, the USB connection may be unstable, or the
phone stream may have stopped. Close other apps, reconnect, restart the phone
server, or lower the requested resolution:

```powershell
python scripts/live_camera.py --width 1280 --height 720
```

### 12.5 Phone stream will not open

Check, in this order:

1. the URL works in the laptop browser;
2. the IP is the phone's current IP;
3. the port matches the app;
4. both devices are on the same Wi-Fi and not an isolated guest network;
5. the phone server is running;
6. Windows Firewall allows Python on private networks;
7. VPN is off.

Common URLs:

```text
IP Webcam:  http://PHONE_IP:8080/video
DroidCam:   http://PHONE_IP:4747/video
```

#### `Error number -138 occurred`

Typical output:

```text
[tcp @ ...] Connection to tcp://192.168.1.45:8080 failed:
Error number -138 occurred
```

This is FFmpeg failing to open a TCP connection to the phone. It is a
network/endpoint failure, not a template, camera-focus, or bubble-reading
failure. Do not change the template or key to fix it.

Run these checks in PowerShell, replacing the IP and port with the values
shown in the phone app:

```powershell
# 1. Is the port reachable at all?
Test-NetConnection 192.168.1.45 -Port 8080

# 2. Does the stream answer HTTP directly (bypassing any system proxy)?
curl.exe --noproxy "*" -v --max-time 5 "http://192.168.1.45:8080/video" -o NUL

# 3. Is the laptop on the same subnet?
ipconfig
```

How to read the result:

| Result | Meaning | Fix |
|---|---|---|
| `TcpTestSucceeded : False` | phone is not listening on that IP/port, or the network blocks it | start the phone server, use the exact IP/port from the app, fix Wi-Fi isolation |
| `curl` prints `Connection refused` | server is not listening on that port | start/restart the app, change the port in the app and use the new URL |
| `curl` times out | wrong subnet, firewall, VPN, or AP/client isolation | compare `ipconfig` addresses; use phone hotspot |
| `curl` prints `HTTP/1.1 200 OK` and `multipart/x-mixed-replace` | stream is reachable; OpenCV should be able to open it | retry the scanner; if it still fails, try `/videofeed`, VLC, or DroidCam |
| `curl` prints `404 Not Found` | server is up but the path is wrong | use the URL shown in the app; try `/videofeed` |
| `curl` cannot resolve the host | typo or phone IP changed | re-read the IP in the app |

If the port is reachable but OpenCV still fails:

- try the alternate IP Webcam endpoint:

  ```powershell
  python scripts/live_camera.py --source "http://192.168.1.45:8080/videofeed"
  ```

- test with VLC (Media → Open Network Stream); if VLC plays it but OpenCV does
  not, try the DroidCam app instead;
- check proxy environment variables; local phone traffic should bypass them:

  ```powershell
  $env:NO_PROXY = "192.168.1.45"
  ```

- allow `python.exe` through any third-party firewall/antivirus network shield.

If the router isolates clients (common on guest/college Wi-Fi), the simplest
reliable fix is the **phone hotspot**:

1. turn on the phone's hotspot;
2. connect the laptop to that hotspot;
3. start the IP Webcam server;
4. use the IP shown in the app (often `192.168.43.1`) and port `8080`.

This bypasses router client isolation and is the most reliable test setup.

### 12.6 `NO SHEET`

The preview cannot find the printed frame or the page outline.

- Show the whole sheet, including all four frame edges.
- Put the sheet on a dark table/mat so the page edge is visible.
- Avoid white paper on a white desk.
- Keep the sheet inside the camera frame.
- If the frame is washed out, use more diffuse light and less glare.

### 12.7 `MOVE CLOSER` / `MOVE BACK`

The detected sheet is too small/large in the frame.

- `MOVE CLOSER`: fill more of the frame with the sheet.
- `MOVE BACK`: the sheet is too close; keep the whole sheet visible.
- `CENTER THE SHEET`: move it toward the middle, with all edges inside.

### 12.8 `TOO BLURRY`

- Hold still and let autofocus settle.
- Clean the lens.
- Add light; blur often follows low light.
- Keep the camera parallel to the sheet.
- `--min-blur` lowers the gate, but do not lower it for production just to
  make the banner green. Fix the focus.

### 12.9 `TOO DARK` / `TOO BRIGHT`

- Too dark: add diffuse light; avoid shadows from your hand.
- Too bright: reduce direct light; tilt away from a reflection; avoid glossy
  paper.
- Glare on a bubble can produce `UNOBSERVED`/`REVIEW`; reposition the light.

### 12.10 `HOLD STEADY`

The preview needs a short stable period (about 0.45 s) before it turns green.
Hold the sheet still. A single still fed to `--check` always shows
`HOLD STEADY`; that is expected.

### 12.11 Live capture fails with “sheet not found”

```text
Capture failed: sheet not found - fill the frame, use a dark background, or tap to focus
```

The live capture path uses `force_detect=True`. If neither the frame nor a
page-shaped flat scan is found, it refuses to grade the whole camera frame.
Fix the framing/background; this guard prevents silent garbage.

### 12.12 Key/template mismatch

```text
ValueError: key template_id 'old-sheet' does not match template 'new-sheet'
```

The key belongs to a different design. Use the matching key, or update
`template_id` in the key if you intentionally copied it. Do not bypass this
without understanding that the question mapping may be wrong.

### 12.13 Key is missing questions

```text
ValueError: key is missing questions: [...]
```

Every question in the template must exist in the key. Add the missing entries
with `answer` and optional `marks`.

### 12.14 Blank sheet shows marks

Do not use the scanner for real grading yet. Check:

- `audit_canonical.png`: is the sheet aligned and complete?
- Is the printed frame/page edge fully visible?
- Is the image overexposed or shadowed?
- Does the template's `grid` match the actual sheet?
- Are the bubble coordinates inside the printed circles?
- Is `calibration.method` `fallback` with a very low threshold?

A blank sheet is the strongest false-positive test. Fix it before proceeding.

### 12.15 One question is always `REVIEW`

Typical causes:

- two bubbles partially filled;
- a faint pencil mark;
- glare or shadow over one bubble;
- a print defect or stray mark;
- the bubble sits near the edge of the template circle.

Inspect the `audit_canonical.png` and the per-question `answers` reasons. If it
is a real ambiguity, keep it as `REVIEW` and resolve it manually. Do not hide
it by lowering thresholds.

### 12.16 Selections are shifted or all wrong

Check in this order:

1. `audit_canonical.png` is upright and not stretched.
2. The overlay circles sit on the printed bubbles.
3. `template_id` matches the sheet version.
4. The key's question numbering matches the printed numbering.
5. The key's `answer` values are zero-based.
6. The compiler's band-major numbering matches the printed layout.
7. The template's `grid.options` matches the sheet.

If the audit image is correct but every selection is offset by one question,
the template's q assignment is likely wrong. If every selection is offset by
one option, the option order or key indexing is wrong.

### 12.17 Total is wrong but selections look right

- Is the key answer for that question correct?
- Is `marks` correct?
- Is `negative_per_wrong` set when you expected no penalty?
- Is `multi` set to `partial` or `review`?
- Is the question in `review_questions` and therefore awarded 0?
- Is `maximum` what you expect?

### 12.18 `frame_quad` confidence is low or orientation is wrong

- Is the whole printed frame visible?
- Are the three solid pink panel boxes visible?
- Do the template's `regions` match the panel's actual location?
- Is `warp.debug.panel_distance` large (≥1.5)?
- Is the frame color still pink? `_pink_mask` is tuned for the supplied print.
- Is the sheet rotated? Frame mode should correct 0/90/180/270 only when the
  panel is found.

### 12.19 A flat scan is detected as `page_quad` or `frame_quad`

By default, a page-shaped scan with a white border uses `full_frame`. If yours
does not, try:

```powershell
python scripts/read_image.py "scan.png" --no-detect --out results\flat
```

If you need to test detection on a flat scan:

```powershell
python scripts/read_image.py "scan.png" --force-detect --out results\forced
```

### 12.20 Output folder is not where expected

`--out` ending in an image extension writes the image there and puts JSON/audit
beside it. Any other `--out` is a directory. With no `--out`, the default is
`results/<input-stem>`.

### 12.21 `selftest.py` fails after a redesign

`selftest.py` is hard-coded to the supplied `q-paper` sheet, sample images, and
expected totals. If you changed the design, either:

- keep the old template/key/samples and run the old selftest separately; or
- update `selftest.py` with the new template, key, samples, and expected
  values, and add new checks for the new layout.

Do not change expected values merely to silence a real failure.

---

## 13. Current limitations

These are known, intentional boundaries of the current M0 + live tester:

- **One captured sheet per worker.** `live_camera_with_audio.py` continues
  across sheets and re-arms automatically, but each sheet is still graded as a
  single captured frame; there is no queue/database of a whole exam session and
  no CSV export.
- **No continuous bubble overlay.** The preview detects frame/quality, but
  bubbles are read only on capture. The continuous voice script displays the
  mark badge, not live per-bubble marks. This keeps the preview responsive.
- **Voice is best-effort.** It depends on `System.Speech`/`say`/`espeak`;
  if audio is unavailable, the mark is still displayed and saved.
- **No multi-frame fusion.** A single frame is graded; there is no temporal
  voting across frames.
- **No QR / ArUco / fiducials.** Sheet identity is manual (template/key
  selection).
- **No student-ID or subject-code OCR.** Those regions exist in the template
  but are not read.
- **Orientation is automatic only in `frame_quad` mode.** Page and full-frame
  captures assume the sheet is upright.
- **One homography.** Lens distortion, paper curvature, and strong perspective
  are not modeled beyond the four-point warp.
- **No per-question multi-answer key.** `answer` is a single option index.
- **No automatic set A/B detection.** Use separate keys.
- **`blank` and `partial_multi` policy fields are not applied.**
- **Template-specific compiler.** It expects two bands, 16–18 pt stroked
  circles, up to five options, and a large white-filled frame.
- **PyMuPDF is compile-only and AGPL/commercial.** Check the licence before
  distributing a closed-source build.
- **Speed.** A capture is roughly 1.0–1.3 s on the development machine;
  the continuous script grades in a background thread, but a new mark still
  takes that long to appear. Preview detection is roughly 5 Hz; a 30 fps
  bubble overlay is not available.
- **No database, CSV export, or packaged installer.** Results are files on
  disk.

Do not use the output as the only record for high-stakes grading. Keep the
`audit_canonical.png`, the raw capture, and a human review path for `REVIEW`
and disputed `WRONG` answers.

---

## 14. Housekeeping and privacy

- `results/` is generated output and safe to delete; it can contain student
  names/IDs if they appear on the sheet. Treat it as student data.
- `templates/` and `keys/` are the reusable inputs. Back them up and version
  them by design (`q-paper_v1`, `q-paper_v2`, etc.).
- Do not edit a template while a live session is running; restart the app after
  changing template/key files.
- The app writes captures locally; it does not upload them.
- If you share a `result.json` for debugging, remember it contains the
  per-question answers and the scan path. Remove student-identifying paths if
  needed.

Useful cleanup commands (only for generated output):

```powershell
Remove-Item -Recurse -Force results\live
Remove-Item -Recurse -Force results\smoke_flat
```

Do not delete `templates/`, `keys/`, or `samples/` unless you have backups.

---

## 15. Quick reference

### Normal operation

```powershell
# Activate
.\.venv\Scripts\Activate.ps1

# One still
python scripts/read_image.py "sheet.jpg" --template templates/q-paper.json --key keys/sample_key.json --out results\scan --debug

# Laptop camera
python scripts/live_camera.py

# Continuous voice scanner for a stack of sheets
python scripts/live_camera_with_audio.py

# Phone camera (replace the IP)
python scripts/live_camera.py --source "http://192.168.1.45:8080/video"

# Camera probe
python scripts/live_camera.py --list

# Live path without a camera
python scripts/live_camera.py --check "samples/test-new/Asset 1@4x.png"

# Acceptance test
python scripts/selftest.py
```

### Design change

```powershell
# Back up
Copy-Item templates\q-paper.json templates\q-paper_v1.json
Copy-Item keys\sample_key.json keys\q-paper_v1_key.json

# Compile the new PDF
python scripts/compile_template.py "design\new-sheet.pdf" --out templates\new-sheet.json --id new-sheet --dpi 200

# Inspect
python -c "import json; t=json.load(open('templates/new-sheet.json', encoding='utf-8')); print(t['page']); print(t['grid']); print(t['frame']); print(t['regions'])"

# New key: copy sample and edit template_id + answers
Copy-Item keys\sample_key.json keys\new-sheet_key.json

# Blank + known sheet + live check
python scripts/read_image.py "samples\new-blank.png" --template templates\new-sheet.json --key keys\new-sheet_key.json --out results\new_blank --debug
python scripts/read_image.py "samples\new-known.png" --template templates\new-sheet.json --key keys\new-sheet_key.json --out results\new_known --debug
python scripts/live_camera.py --check "samples\new-known.png" --template templates\new-sheet.json --key keys\new-sheet_key.json
python scripts/live_camera.py --template templates\new-sheet.json --key keys\new-sheet_key.json
```

### The three files to inspect when something is wrong

1. `capture.jpg` — what the camera saw.
2. `audit_canonical.png` — what alignment produced.
3. `result.json` — `warp`, `calibration`, `answers`, `grade`.

If those three agree with your eyes, the scanner is working. If they do not,
fix alignment/template/key before changing thresholds.
