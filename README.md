# LatticeLock OMR — M0 + live tester

This is the working still-image reader plus a live preview/capture tester for
the pipeline described in [realtime-plan.md](realtime-plan.md). The full
operating manual — setup, checks, troubleshooting, and redesigning the sheet —
is in [documentation.md](documentation.md). The Android product and engineering
plan — platform choice, paper/print pipeline, database, and delivery phases —
is in [mobile-app-plan.md](mobile-app-plan.md).

The pipeline:

1. compile the exact bubble geometry from `q-paper.pdf` into `template.json`;
2. read a still image (flat scan, angled capture or cropped render);
3. preview a webcam or phone stream, guide framing/focus/exposure, capture on
   SPACE and grade it against a teacher key;
4. render a result image in the same green/red/blue language as
   `code try/omr_result_v2.jpg`.

It does **not** include QR/fiducials, lattice refinement, multi-frame fusion or
a packaged desktop UI yet. Those are M2–M4.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. (Optional) Compile the sheet geometry once (120 exact circles -> template.json)
#    Skip this if templates/q-paper.json already exists.
python scripts/compile_template.py "code try/q-paper.pdf" \
  --out templates/q-paper.json --id q-paper

# 2. Create your own answer key for a test (interactive)
python scripts/make_key.py --out keys/my_key.json

# 3. Read and grade the filled sample
python scripts/read_image.py "code try/q-paper_page-0001.jpg" \
  --template templates/q-paper.json \
  --key keys/sample_key.json \
  --out results/sample_scan

# 4. Try the synthetic angled capture
python scripts/read_image.py samples/q-paper_tilted.jpg \
  --template templates/q-paper.json \
  --key keys/sample_key.json \
  --out results/sample_tilted

# 5. Live webcam (SPACE = capture & grade, Q = quit)
python scripts/live_camera.py

# 5b. Continuous scanner with voice marks for a stack of sheets
python scripts/live_camera_with_audio.py

# 6. Phone on the same Wi-Fi (Android IP Webcam app)
#    Replace 192.168.0.23 with the IP shown in the app.
python scripts/live_camera.py --source "http://192.168.0.23:8080/video"

# 7. Test the live capture path without a camera
python scripts/live_camera.py --check "samples/test-new/Asset 1@4x.png"

# 8. Run all acceptance checks
python scripts/selftest.py
```

Expected result: **29/30** on the flat sample and on the tilted capture;
**0/30** on the blank sheet; Q4 is flagged for review on the mixed sheet.

---

## Answer keys for each test

Answers are not stored in the sheet template, so the same printed design can
be used for every subject and test. Each test gets its own key file:

```bash
python scripts/make_key.py --out keys/class8_bangla.json
```

The wizard asks for the correct option of each question (`1`-`4` or
`ক`/`খ`/`গ`/`ঘ`), then the marks. Enter keeps an existing answer when editing,
`p` goes back, `q` cancels without saving. You can paste a whole answer list
at the first question (`2,1,3,4,...`). Per-question marks can be set in the
same run (`3=2,7=5`), and negative marking is available with `--negative`.

Use the key on any scanner:

```bash
python scripts/live_camera_with_audio.py --key keys/class8_bangla.json
python scripts/read_image.py "sheet.jpg" --key keys/class8_bangla.json
```

To change an existing key, preload it with `--from`:

```bash
python scripts/make_key.py --from keys/class8_bangla.json --out keys/class8_bangla.json
```

For a list of answers without prompts, use `--answers` (and optionally
`--marks` for one value or one per question). The tool validates the answers,
marks, and template id before writing, and re-reads the file afterwards. The
wizard shows option positions 1-4; inside the JSON those are stored zero-based
(`ক` = 0, `খ` = 1), but you never need to type that by hand. See
[documentation.md](documentation.md#101-create-a-new-key) for the manual JSON
format and the marking policy fields.

---

## Live webcam / phone camera

Start with the laptop camera:

```bash
python scripts/live_camera.py
```

Use `--source 1` for a second/USB camera, or pass a phone stream URL. The
preview draws the detected sheet frame and shows one of:

- `READY - PRESS SPACE` (green frame)
- `MOVE CLOSER`, `MOVE BACK`, `CENTER THE SHEET`
- `TOO BLURRY`, `TOO DARK`, `TOO BRIGHT`, `HOLD STEADY`
- `NO SHEET` (use a dark background so the white page edge is visible)

| Key | Action |
|---|---|
| `SPACE` | capture the current frame, grade it, show the result |
| `A` | arm/disarm automatic capture when the frame is ready |
| `S` | save the current camera frame |
| `Q` / `ESC` | quit |

On the result screen, `SPACE` returns to live, `S` saves a copy, `Q` quits.
Each capture is saved under `results/live/<timestamp>/` with `capture.jpg`,
`result.jpg`, `result.json` and `audit_canonical.png`. Grading uses
`keys/sample_key.json` by default; pass `--key your_key.json` to use your own
answer key. The overlay still shows every detected selection either way.

Practical setup:

- fill roughly 15–80% of the frame with the sheet, fully inside the image;
- a dark table/mat gives the best page-edge fallback if the pink frame is
  washed out;
- avoid direct light reflections on the bubbles; hold still until the green
  `READY` banner;
- the preview checks geometry/focus/steadiness, but reads bubbles only on
  capture, so the preview stays responsive;
- use `--list` to probe camera indices, and `--check IMAGE` to verify the whole
  capture path without opening a window.

Use the normal OpenCV wheel, not the headless one, for the live window:

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install --upgrade opencv-python
```

If the installed OpenCV lacks GUI support, the script prints this instruction
instead of crashing.

---

## Continuous voice scanner (many sheets)

For checking a stack of sheets without pressing a key for every one:

```bash
python scripts/live_camera_with_audio.py
```

It keeps the camera preview running, grades each new sheet in a background
worker, and by default **projects the result onto the physical sheet** (AR
overlay): the total inside the printed mark box plus green circles for correct,
red for wrong and blue for the key answer. There is no separate result screen,
so the live view keeps running. The voice speaks the mark once per sheet.
Every scan is saved under `results/live_audio/<timestamp>/` with the same
`capture.jpg`, `result.jpg`, `result.json` and `audit_canonical.png` as the
normal live tester.

| Key | Action |
|---|---|
| `SPACE` | scan the current sheet immediately; on the graded view, return to live |
| `P` | pause/resume automatic scanning |
| `V` | mute/unmute voice |
| `S` | save the raw camera frame |
| `Q` / `ESC` | quit |

**AR overlay is the default**, so the command above needs no flags.

### Turning the AR overlay off

To use the old separate graded-sheet view instead of drawing on the physical
sheet:

```bash
python scripts/live_camera_with_audio.py --no-ar --result-hold 2.5
```

`--result-hold` controls how long that separate view stays on screen
(default `0`, i.e. no separate view). `--no-ar` disables the overlay; use both
together to get the previous behavior. Press `SPACE` on the graded view to
return immediately.

The overlay is recalculated from the current camera frame, so it follows the
sheet as it moves. This is a 2D projective overlay (homography), not a separate
warped result image.

AR needs the printed frame or page outline to be visible. If the sheet cannot
be located, the overlay is hidden rather than drawn in the wrong place. Very
strong perspective, glare, or lens distortion can still make circles or text
less exact; the saved `result.jpg` remains the authoritative visual record.

After a result it waits for the sheet to be removed, or for a new sheet to be
placed in its position, before scanning again. That prevents the same sheet
from being announced repeatedly. Press `SPACE` if you want to rescan the same
sheet.

Phone camera:

```bash
python scripts/live_camera_with_audio.py --source "http://PHONE_IP:8080/video"
```

Voice can be disabled or tested independently:

```bash
python scripts/live_camera_with_audio.py --no-voice
python scripts/live_camera_with_audio.py --test-voice
```

On Windows the voice uses the built-in `System.Speech` engine through
Windows PowerShell. If speech is unavailable, the mark is still displayed and
saved; the scanner does not stop.

`pymupdf` is only needed by `scripts/compile_template.py`. The live scanner and
`scripts/read_image.py` load an existing `templates/*.json` and do not import
it, so a blocked or missing `pymupdf` DLL cannot stop scanning:

```text
ImportError: DLL load failed while importing _mupdf:
An Application Control policy has blocked this file.
```

If you see that error while running a **compile** command, use an existing
template or run the compiler in an environment where the DLL is allowed.
Scanning itself does not require it.

---

## Outputs

Each run writes:

```
results/<name>/
  result.jpg           # canonical sheet + green/red/blue overlay + total
  result.json          # warp mode, calibration, answers, grade
  audit_canonical.png  # lossless canonical warp
```

`result.json` contains the per-question selected options, state, confidence,
the GMM/fallback calibration parameters and the full grade breakdown.

---

## How it works

### Template compilation

`omr/template.py` opens the PDF with PyMuPDF and finds the stroked
**17 × 17 pt** circle paths. Your `q-paper.pdf` has exactly 120, arranged as
8 rows × 15 columns on a 182 × 257 mm page. The compiler clusters them into
two 4-option bands and writes millimetre coordinates:

```
question q = 1..30
band      = 0 if q <= 15 else 1
column    = (q - 1) % 15
option    = 0..3   (ক, খ, গ, ঘ)
```

No 75/25 split, no equal-column assumption, no page-size guess.

### Reading

`omr/geometry.py`:

- first looks for the thin printed frame around the answer grid; mapping its
  four corners is exact, works on white-on-white renders and survives cropping
  outside the frame;
- resolves 0°/90°/180°/270° orientation by locating the three solid pink boxes
  in the marks panel and checking which corner assignment puts them on the
  right. `scripts/selftest.py` exercises all four rotations on a base capture;
- keeps a page-shaped flat scan on the full-frame path, because its
  anti-aliased frame can bias the quad by a pixel or two;
- otherwise falls back to page-quad detection and a perspective warp to
  200 DPI (1433 × 2024).

`omr/reader.py`:

- converts to true RGB and computes `ink = 255 - min(R,G,B)`, which suppresses
  the pink print;
- estimates paper locally and subtracts it (`ink_corr`), preserving the
  difference between faint pencil and black pen;
- measures an inner disk (`0.52 r`) and a ring annulus (`0.85–1.15 r`) per
  bubble;
- fits a two-component GMM to **that sheet's** 120 scores, instead of using a
  global pixel threshold;
- emits `MARKED`, `EMPTY`, `UNCERTAIN` (and `UNOBSERVED` if glare covers a
  bubble), then aggregates per question as `ANSWERED`, `BLANK`, `MULTI`,
  `REVIEW` or `UNOBSERVED`.

`omr/grade.py` applies the key, per-question marks and policy. `omr/overlay.py`
draws the result: green = correct, red = wrong selection, blue = key answer,
orange = blank, yellow = review, purple = multi-mark.

---

## M0 verification results

| Input | Warp | Result |
|---|---|---|
| `code try/q-paper_page-0001.jpg` (flat, 29 marks, Q1 wrong) | full_frame | **29/30**, 29 correct, 1 wrong |
| `samples/q-paper_tilted.jpg` (synthetic perspective) | page_quad, confidence 0.83 | **29/30**, 29 correct |
| `samples/q-paper_blank.png` (all bubbles cleared) | full_frame | **0/30**, 30 blank, 0 marked |
| `samples/q-paper_mixed.png` (blank + double + faint) | full_frame | Q1 answered, Q2 blank, Q3 multi, Q4 review, Q5 answered |
| `samples/test-new/Asset 1–4` (rotated, skewed, cropped) | frame_quad, confidence 0.98–0.99 | **29/30** on each, 30 answered, 0 blank/review |
| live camera 0/1 (1280×720 / 640×480) | camera opened and read | preview loop started; no sheet was in front of the camera for the automated run |
| `--check "samples/test-new/Asset 1@4x.png"` | frame_quad, 0.98 | **29/30**, same as the still-image path |

`python scripts/selftest.py` checks all eight, plus synthetic 0°/90°/180°/270°
rotations, automatically.

Calibration on the full sample: GMM separation **15.5σ**, threshold **0.130**,
30 marked bubbles. The four rotated/cropped assets separate at **29.5–40.7σ**
and all choose the correct orientation. Blank sheet: fallback threshold, **0**
marked bubbles.

---

## Key format

```json
{
  "key_id": "q-paper-sample-29",
  "template_id": "q-paper",
  "questions": {
    "1": { "answer": 1, "marks": 1 },
    "2": { "answer": 0, "marks": 1 }
  },
  "policy": {
    "multi": "wrong",
    "blank": 0,
    "negative_per_wrong": 0.0
  }
}
```

`answer` is the zero-based option index (0 = ক, 1 = খ, 2 = গ, 3 = ঘ). The key
is bound to `template_id`; a mismatch is rejected.

---

## Current limitations

- **Live capture, not continuous fusion.** The preview detects the sheet and
  checks focus/exposure; bubbles are read once on SPACE. Multi-frame log-odds
  fusion and a continuous bubble overlay are M3.
- **No QR/ArUco.** Sheet identity is a file name / manual session; the
  generator upgrade in `realtime-plan.md` §7.1 is M2.
- **Orientation only when the frame is found.** The marks panel resolves
  0°/90°/180°/270° for frame-aligned captures. A page-quad or full-frame
  capture still assumes upright until QR/fiducials arrive in M2.
- **No lattice bundle adjustment.** A single homography handles the tested
  tilt; curvature and lens distortion are M2.
- **Flat-scan gate is heuristic.** A page-shaped scan uses full-frame mapping
  and skips the frame quad; a page on a pure-white desk at an angle may also
  need `--force-detect` (or a dark mat).
- **Template-specific.** Coordinates are for the supplied Al-Quran Academy
  30-question sheet. Compile a different PDF for a different layout.
- **PyMuPDF is AGPL/commercial.** Check the licence before distributing a
  closed-source build; the template compiler can be replaced with a permissive
  PDF parser or run as an internal tool.
- **Speed.** A capture takes about 1.0–1.3 s on this machine, dominated by the
  81 × 81 paper estimate. Preview detection runs at roughly 5 Hz; that is
  enough for guidance but not for a 30 fps bubble overlay.

---

## What comes next

- **M1 (partly done):** `scripts/live_camera.py` covers camera capture, frame/
  focus/exposure guidance and Capture → grade; `scripts/live_camera_with_audio.py`
  adds continuous sheet-to-sheet scanning, mark badges and voice totals.
  Remaining: continuous bubble overlay, CSV export and a packaged loop.
- **M2:** QR + four fiducials, lattice bundle adjustment, glare/shadow handling
  under camera exposure.
- **M3:** multi-frame log-odds fusion, review queue, audit/duplicate handling.
- **M4:** Wild-OMR/Tamaulipas evaluation, latency tuning, packaging.

The design and prior-art evidence are in [realtime-plan.md](realtime-plan.md)
and [papers.md](papers.md).
