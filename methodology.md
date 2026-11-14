# Verification-Driven, Template-Anchored Optical Mark Recognition for Smartphone Cameras

*Methodology of the OMR reader implemented in `android-app/assets/omr-engine.js` and `android-app/assets/app.js` (checker app, v1.17-single-sheet). Written to serve as the Method section of a research paper; every constant quoted here exists in code and is exercised by `test/pipeline-test.js` and `test/conditions-test.js`.*

---

## 1. Overview

The system reads four-option bubble answer sheets photographed by a smartphone (live camera and uploaded photos) and grades them against a teacher-defined answer key. The pipeline is entirely classical computer vision — no training data, no network calls, runs offline in a WebView — and is organised around one design principle:

> **Nothing is read until the geometry it depends on has been verified, and anything that cannot be verified is refused with a reason instead of graded.**

The stages are:

1. **Template-anchored geometry** (Section 2) — the sheet design is measured once, in millimetres, and shared by the printer and the reader.
2. **Sheet location** (Section 3) — pink-frame / page-outline connected components → convex hull → dominant quad → homography.
3. **Orientation disambiguation** (Section 4) — *Ring-Likelihood Orientation Disambiguation (RLOD)*, the principal novel contribution: the four turns of the sheet are scored as competing hypotheses by a printed-ring likelihood plus a printed-band asymmetry witness, and an ambiguous orientation refuses to grade.
4. **Ring-likelihood alignment** (Section 5) — windowed ink sampling with joint translation+scale search corrects residual warp error before any bubble is measured; a global *ring fit* scalar quantifies how well the grid landed on the print.
5. **Per-bubble reading with local calibration** (Section 6) — annulus-local paper reference, adaptive Otsu/k-means thresholding, MARKED / EMPTY / UNCERTAIN states.
6. **Live-mode temporal fusion** (Section 7) — per-bubble median over a sliding window with agreement streaks; misaligned frames are dropped, never averaged.
7. **Photo-mode rotation loop and abstention** (Section 8) — orientation-first reading, best-of-rotations fallback, calibrated refusal.

---

## 2. Template-anchored geometry

The app supports exactly **one** printed design, the page in `print/v3.pdf` (A4, 210 × 297 mm; 15 question columns per row; four options A–D stacked per question, 9.0 mm option pitch, 3.56 mm bubble radius; four rows = 60 question slots; a pink alignment frame around the answer table; Student ID / Subject Code / Total Marks boxes beside it). Every measurement below was taken from that page at 150 dpi; an exam with fewer than 60 questions uses the top rows and the reader grades only the exam's questions (`gradeRead` clips the count to the key length). The same template drives:

* the **printer** (HTML/CSS sheet generator, positions in mm),
* the **reader** (all template coordinates converted to page pixels at a fixed 150 dpi model resolution, `PAGE_W = 1240`, `PAGE_H = 1754`).

This eliminates the classic template/registration mismatch: what is printed and what is measured come from one source of truth. The grid constants measured on `print/v3.pdf` are: column centres `13.21 … 148.59 mm` (15 columns), option-A centres `78.74 / 122.09 / 165.61 / 208.96 mm`, option pitch `9.0 mm`, bubble radius `3.56 mm`, answer-table border `x 8.37 … 153.43`, `y 64.35 … 244.01 mm`, numbered band `7.88 mm` tall sitting `1.52 mm` above option A.

Derived geometric facts kept in closure:

* bubble lattice positions `BUBBLES[i] = (q, opt, x, y, r)`;
* the **frame destination quad** `FRAME_DST` (the pink frame rectangles, in page pixels) and the **page destination quad** `PAGE_DST` (the full page corners);
* lattice centroid `(GRID_CX, GRID_CY)` — all alignment corrections (translation and scale) are applied about the centroid so the grid grows or shrinks in place;
* the **answer region** `omrRegion(padMm)` — the axis-aligned bounding box of all bubbles padded by 3 mm, used to exclude header, handwriting and classroom furniture from live measurement;
* **numbered-band geometry** for RLOD (Section 4.2): the witness strip runs `BAND_TOP_MM = 11.0` to `11.0 − 4.9 = 6.1 mm` above each row's option-A centre (inside the printed pink band, `11.44 … 5.08 mm` on v3), and the white strip below a row's option D is sampled at `0.85 ± 0.35 mm`.

---

## 3. Sheet location

Detection runs on a downscaled copy (max side 1200 px) for speed, upscaled coordinates returned.

**(a) Pink-frame detection.** A colour mask isolates pink ink; a 3×3 dilation closes anti-aliasing gaps; connected components are extracted. A component is a frame candidate when its bounding box is large enough relative to the photo (≥ 18 % of width, ≥ 4.5 % of height), not more than quarter-filled (a solid pink info box is `fill ≈ 1` and is excluded — it is *not* the frame), and its reduced-quad side-length ratio matches the known frame aspect within 0.10. The convex hull of the component is reduced to a dominant quadrilateral (`reduceToQuad`, iterative minimum-area removal of collinear vertices).

**(b) Flat-scan short-circuit.** If the input's aspect ratio matches the page within 1.5 % and the border pixels are white (border median ≥ 245), the input is a flat scanner image and the identity mapping is used directly.

**(c) Page fallback.** When no pink frame is usable (washed out by light, printed greyscale), the *brightest* large component is the paper: an adaptive Otsu cut (clamped to [90, 205]) splits paper from desk, and the largest bright component matching either page or frame aspect (within 0.15) is reduced to a quad. Size floors are deliberately loose (≥ 14 % of frame per side) so a sheet filling only a fifth of the photo is still found; verification later rejects false positives, not detection.

**(d) Homography.** The winning quad and its destination (`FRAME_DST` or `PAGE_DST`) define a homography mapping page pixels ↔ image pixels. Everything downstream measures in page space; the live path samples the image *through* the homography without warping (Section 7.1), and photo mode warps a full page once (Section 8).

**Fusion of the two geometries.** In photo mode both hypotheses (frame-interior and whole-page) are warped and read; the one whose *printed rings line up better* wins — never the one with more detected marks (which would reward a lucky wrong grid).

---

## 4. Orientation disambiguation (novel)

### 4.1 Problem

A photographed sheet can sit in four turns (0°, 90°, 180°, 270°). Because the pink frame is a plain rectangle and the bubble lattice is symmetric at full-row counts (30, 60), the *geometry alone* cannot separate the front of the page from its half-turn, and cannot reliably separate the two side-up motifs. Published systems recover orientation from corner fiducials or learned region detectors and then **assume** the matrix is correct; a mis-recovered orientation silently reads the wrong bubbles — the failure mode reported in the field as "the scan confuses vertical and horizontal mode."

### 4.2 Ring-Likelihood Orientation Disambiguation (RLOD)

RLOD treats the four turns as **competing hypotheses scored by printed evidence**:

**Hypothesis scoring.** For roll `k ∈ {0,1,2,3}` of the ordered detected quad, let `H_k` be the homography from template page coordinates to the image under that roll. Two observations are recorded:

1. **Ring likelihood** — `fit_k = ringFit(image, H_k)` (Section 5.3): the median, over an evenly spread bubble sample, of the ink found at the printed ring circumference of each bubble under hypothesis `k`. A wrong turn maps the lattice off the printed rings and scores near the paper's ink floor; the correct turn scores the ring contrast (≈ 0.13–0.9 depending on print and light).

2. **Band asymmetry** — the sheet prints a *numbered band above every question row* and leaves blank paper in the equal-sized strip between a row's option D and the next band. Under hypothesis `k`, for every row except the last, we sample both strips through `H_k` and vote `vote_r = (dark_above − dark_below) + (pink_above − pink_below)`, both means over the grid width, `dark = 255 − min(R,G,B)` and `pink = R − G`; the hypothesis score uses the median vote, `band_k`. **Both** channels are measured because they fail in opposite conditions: a dim classroom photo keeps the colour difference of the pink band after the brightness difference has gone (band `R − G ≈ 33` against paper `≈ 3`, while both are almost equally dark), whereas a bright screenshot of a sheet keeps the brightness difference and little colour. Scoring either alone made the same sheet read upright on one frame and upside-down on the next — the field symptom of a live scan that suddenly drops to a handful of correct answers. This is a **printed asymmetry witness**: correct orientation put the band above the rows (`band ≈ +0.1…0.3`); the half-turn puts it below (`band ≈ −…`); side turns land both strips on symmetric content and the vote collapses towards zero while `fit_k` separates them. When the lattice itself is asymmetric (counts 15, 20, 25, 40, 45, 50), the ring likelihood separates all four turns on its own.

**Decision rule.** With `score_k = fit_k + 0.9 · band_k`, ordered by score:

```text
margin = score(1st) − score(2nd)

margin ≥ 0.03                     → orientation = 1st (ring evidence decisive)
margin <  0.03 and panel anchor    → orientation = panel's choice (anchor breaks the tie)
margin <  0.03 and no panel anchor → REFUSE TO GRADE (orientation unresolved)
no second hypothesis                → REFUSE TO GRADE
```

The **panel anchor** is the geometric check already used for orientation: the pink info boxes beside the table must project, under the correct roll, to their known template positions (`chooseOrientation`); it is cheap, direct, and its expected position is a hard geometric constraint, not a pixel heuristic.

**Abstention as a first-class outcome.** An unresolved orientation is reported, not guessed: photo scans return an explicit refusal message, live mode resets fusion and displays *"Turn the sheet upright"*. This converts the silent mis-grade (the worst possible exam outcome) into a calibrated abstention.

**Which anchor is used.** The reader prefers the **printed answer table** (Section 2's pink frame) over the page outline whenever the table was found at all. The table is the rectangle that is actually read, its size on the print is known, and it is unique on a desk — a page outline is a shape that a second sheet, a notebook or the edge of a table imitates, which is the report of "it counts our side object as rectangle". The page outline is kept only for a sheet whose table fills the frame edge-to-edge.

### 4.3 Positioning among published work

State of the art (2023–2026) either (a) relies on corner/border fiducials and unverified homographies (the fault-tolerant OMR line: Hafeez et al., *CMC* 74(2) 2023 and *IEEE Access* 12, 2024, smartphone port), or (b) replaces fiducials with learned region detection whose internal geometry is then assumed (YOLOv8+classical hybrid: Nath et al., *IJERT* 14(12) 2025; YOLOv8+DBSCAN as surveyed in Hernández-Mier et al., *J. Imaging* 11(9):308, 2025; OMR-Exam, *IEEE Access* 2026). OMRNet (*MTAP* 2023) supplies learned per-box confidence but requires per-layout training data and performs no orientation reasoning. None of the surveyed systems scores the four in-plane turns as hypotheses nor gates grading on a printed-ring geometric likelihood. RLOD requires **no training data** and is **layout-portable** (any template registered through Section 2).

---

## 5. Ring-likelihood alignment and the ring-fit scalar

### 5.1 Why alignment comes first

A phone photo is routinely a millimetre or two off the template — a third of a bubble. Measuring before correcting contaminates every score with that residual.

### 5.2 The estimator

`alignOffset` searches translation `(dx, dy)` jointly with isotropic scale `s ∈ [0.94, 1.045]` (the plausible envelope of printer shrink/fit-to-page), coarse then locally refined (±2 px at step 2 then 1; scale refined at 0.0075).

**The scored statistic depends on how the sheet was printed**, and the print is identified before the search: `bandPink` measures the pink of the printed numbering band, which is ≈ 0.10 on the classroom print and ≈ 0.00 on a grey laser print or a black-and-white render. A pink print is matched by **ring colour contrast** — `mean(pink on the ring) − max(pink inside the bubble, pink just outside)` with `pink = R − G`, measured within one bubble so any cast from the room light cancels. A grey print is matched by the **darkness of the ring annulus**, the measurement this section used to use for everything.

The split is not a refinement, it is required. On a pink print the letters printed inside every bubble (`a`/`b`/`c`/`d`) are darker than the printed ring, and the darkness score rises whenever a hypothesis slides the ring samples towards them; measured on 33 real frames of a hand-held pink sheet (480 × 854, WhatsApp-compressed) the darkness search chose corrections of up to a whole option pitch (`[-4,-20,0.993]`, `[-16,-27,0.97]`) on nearly every frame and read the sheet one row off — 1 of 40 sampled frames read correctly, which is the field report of "only 4 out of 60". With the colour channel the same frames give 28 exact reads, 31 correct orientations, and 26 of 30 consecutive frames of the user's video read all 60 marks exactly. Conversely a black-and-white render has no colour at all: scoring it by pink returns zero everywhere, leaves the grid unaligned, and a sheet printed at 96% or 104% then reads 23/30.

`ringFit` (Section 5.3) is still darkness, and is still only a gate: it says "is there print where the rings should be", not "is this the best alignment".

Windowed maxima were retained for `ringFit` (Section 5.3), which is a *gate*, not an estimator: single-pixel sampling there would let a half-pixel geometry error read as "no ring" and refuse a good sheet.

**When the search runs.** In photo mode, always. In live mode, only when the location-implied geometry is *not a local peak of ring ink*. An absolute cut cannot decide this: a pink classroom print scores about a third of a black one, and the trained lattice is dense (9 mm between bubbles, 3.6 mm rings), so almost any shift lands on *some* ring. The test is therefore relative — the fit at the sheet's own position against the fit at the same grid nudged by 1.4 mm (a third of a ring). A real alignment collapses under the nudge (0.50 → 0.05); a bogus one does not. When the test fails, the previous frame's winning correction is tried first (a hand that is not moving keeps the same residual), then a coarse-to-fine search. With this, a correctly framed live frame costs ~55 ms of measurement instead of ~810 ms — the difference between 1 and 7 frames per second, and the reason a small hand movement no longer loses the sheet.

### 5.3 Ring fit

`ringFit(data, w, h, H, offset) = median(bubble score)` where a bubble's score is the 70th-percentile ink in a 12-direction × {0.95 r, 1.05 r} × 3×3-window sample of its printed circumference, divided by 255. Informally: *how well does the template's bubble lattice land on the photo's printed rings, from 0 (nothing but paper) to ≈ 0.9 (crisp print)?*

**Ring fit is the confidence measure for every downstream decision:**

| Gate | Value | Meaning |
|---|---|---|
| Live alignment trigger | `fit < 0.30` | only shaky geometry pays for the search |
| Live frame acceptance | `fit ≥ 0.10` | below this a frame is dropped, never averaged in |
| Photo refinement | best read wins by `fit` | ring evidence beats mark counting |
| Photo wide-research retry | `fit < 0.15` | extend search to ±48 px before giving up |
| Photo refusal | `fit < 0.10` | grade withheld with the reason |

Because ring fit is a printed-structure likelihood, it is invariant to which answers are marked — an empty sheet and a full one are judged by the same ruler, so the gate cannot be gamed by or fooled into grading a mis-placed grid.

---

## 6. Per-bubble reading with local calibration

Every bubble is measured in three regions: interior (`r ≤ 0.52 r`), printed ring (annulus `0.85–1.15 r`), local paper reference (outside `1.35 r`). Using the median of the *outer region* as the bubble's own paper reference makes the score robust to uneven illumination (a shadow across half the page does not produce false marks).

Per bubble: `score = 0.70 · fillRel + 0.30 · darkFrac`, where `fillRel = max(0, (median interior − paper) / 255)` and `darkFrac` is the interior fraction darker than paper + 70/255.

**Threshold calibration** avoids any fixed global cut: per-sheet Otsu/k-means on the score distribution assigns scores to "inked" and "paper" clusters and accepts the split only when clusters separate (`high − low > 0.08`, `separation ≥ 2σ`, ≥ 3 bubble cluster); otherwise a floor-relative fallback is used. A mark is `MARKED` when `score ≥ threshold` and `score ≥ 0.15` and `fill ≥ 0.40`; borderline bubbles become `UNCERTAIN` (graded as review, never as a confident answer).

Per-question states: `ANSWERED` (exactly one mark), `MULTI`, `REVIEW` (uncertainty), `BLANK`.

---

## 7. Live mode: gated temporal fusion

### 7.1 Region-cropped projected reading

Each frame is located once (detection + smooth corner EMA with jump detection), then only `omrRegion` is cropped *through the homography* and magnified up to 3× (bounded by a 1400 px cap; the enlargement cap keeps printed rings several pixels thick at long range, and the alignment search inside the crop scales with the magnification). Bubbles are sampled in the camera image via the homography — no full-page warp per frame, which is what makes 12–15 fps achievable on a phone.

### 7.2 Frame gating

A frame is **dropped, not averaged**, when any of: detection fails; the answer region leaves the frame; the ring fit of the crop is below 0.10; the orientation is ambiguous (Section 4.2). A run of drops (sheet moved away) resets fusion rather than blocking forever.

### 7.3 Fusion

`createFusion({ need: 5, agree: 3, window: 9 })`: each bubble keeps a sliding window of the last 9 frame scores; its state follows the *median*. Inside the window, thresholding uses the running per-frame calibrated cut (EMA α = 0.35), and the ink test is a *difference from the frame's own paper level* (the median fill of the frame, minus the bubble) rather than an absolute brightness — an absolute floor silently turned real marks off whenever the light or the focus changed mid-scan. Frame-level disagreement with the already-read sheet (> 25 % of bubbles) causes the frame to be skipped — six consecutive skips mean the sheet truly changed and history is cleared. A flipped state must hold its direction for 3 consecutive frames (`agree`) before committing. The result is presented when `frames ≥ need` and nothing has changed for ≥ 2 frames (`isSettled`).

This outvotes shaken, blurred, half-aligned frames (a smearing hand means the *median* never breaks).

### 7.4 One sheet, one result

**The tracker decides *when*, the photo reader decides *what*.** When fusion settles, the current frame is graded by the full `scanRGBA` pipeline — the same engine that reads an uploaded photo — and that result is locked: the loop stops scheduling reads (`live.frozen`), the marked frame and the total stay on screen, and a "Scan next sheet" button re-arms the reader. If the full read refuses the frame, nothing is shown and the camera keeps trying: a live scan and an uploaded photo of the same paper now give the same answer by construction, and a doubtful frame can never become a number. Continuous re-reading of the same paper was the source of drifting answers during long classroom sessions, so a sheet is graded exactly once; the next sheet is an explicit action by the teacher.

---

## 8. Photo mode: orientation-first reading with abstention

Uploaded photos are read as they arrive, then — only when needed — at 180°, 90° and 270° (`drawPhoto` canvas rotation loop). Because RLOD resolves in-plane turns inside the engine (Section 4), the **first** angle now normally succeeds (a rotated sheet is read from its unrotated photo); the loop remains as a fallback for EXIF-disastered uploads. Early exit requires a definite grid (`ring fit ≥ 0.35` with at least one mark). Candidate selection (`betterScan`) orders evidence: ring fit first (±0.05 band), then marked count, then detector confidence. Both page and frame geometries of the same detector are scored and the better-aligned one wins.

Refusal states returned to the user carry operational reasons: no sheet found; grid found but rings do not line up (straighten / re-shoot / print at 100 %, never "fit to page"); orientation unresolved (turn upright); over-bright / over-blurred prints.

Uploads follow the same one-scan rule: the result panel is shown once the photo has been graded, and scanning again is an explicit new upload rather than a continuous loop.

---

## 9. Evaluation methodology

`test/pipeline-test.js` (main suite) and `test/conditions-test.js` (camera-condition matrix) render the true sheet design from the same template (printed bands and info boxes included), then photograph it into a synthetic desk scene **through a perspective quad** — real homography distortion rendered pixel-for-pixel, not a simulated rectified approximation — and grade question-by-question against the key.

Variation axis: brightness (0.62× + a moving shadow, +28 % overexposure), 1- and 2-pass 3×3 blur, ±22 sensor noise, 7° hand tilt, strong asymmetric angle, sheet at 20 % frame size, quarter turns (90°/180°/270°), 96 % re-print, greyscale print, blank control; live runs detect → crop → 8 jittered frames (±3–5 px) → fusion → grade.

`test/live-video-test.js` (added after the classroom reports) is a **virtual video** suite rather than stills: 45-frame (1.5 s at 30 fps) hand-held sequences at 960×720 with slow sway, nervous tremble, knock bumps, motion blur on a third of the frames, drifting shadow, sensor noise, a sheet held at 9°, a hand moving closer and a sheet held upside down. Each frame goes through the app's live loop — detect, hold the last geometry through unreadable frames, require the sheet to be steady, then grade the settled frame with the full reader.

**Current measured result** (engine v1.17-single-sheet, single v3 design):

* Main suite (`pipeline-test.js`): **all checks pass**.
* Photo reader: **13 of 15 conditions perfect at 30/30**, including all three turns, 96 % re-print, grey print, far-away sheet, 7° hand tilt; blank control correctly 0/30.
* Live reader (stills matrix): **8 of 9 live cases settle at 30/30**.
* Live reader (video suite): **8 of 8 videos read 30/30** — including the shaky hand (settles once the hand steadies, 0.8 s), blur on a third of the frames, dim room with drifting shadow, 9° tilt, hand coming closer and an upside-down hold. Full read costs 0.31–0.56 s per locked sheet; tracking costs ~55–95 ms per frame.
* The one remaining stress case — a strong asymmetric left-side angle (top edge 460 px vs bottom edge 676 px over a 930 px page) — is **refused** in both modes ("found but did not line up") rather than graded; it is a safe abstention, not a wrong read.
* Real print verification: `print/v3.pdf` rendered at 150 dpi with 30 marked answers reads **30/30, align [0, 0, 1]**, grading exactly the 30 exam questions on the 60-slot sheet.

**Untested limitation (stated honestly):** all evidence is synthetic. The evaluation quantifies robustness to modeled perspective, light, blur, noise, distance, print scale and rotation; two or three real classroom photos at working distance remain the intended confirmation step.

---

## 10. Reproducibility

```bash
cd android-app
node test/pipeline-test.js      # main suite: all sheet sizes, live fusion, prints
node test/conditions-test.js    # camera-condition matrix incl. orientation turns
node test/live-video-test.js    # virtual hand-held video sequences for live mode
```

Key tunable constants (engine, values quoted from code): `ORI_MARGIN = 0.03`; band witness `BAND_TOP_MM = 11.0`, `BAND_LEN_MM = 4.9`; alignment reach `28 px` with an Occam bias `0.03 · (|dx|+|dy|)/reach` that prefers the smallest correction; ring-fit gates `1.20 × nudge (live refine trigger) / 0.10 (photo accept) / 0.15 (wide-research retry)`; fusion `{ need: 5, agree: 3, window: 9 }` with a paper-relative ink floor `0.10`; live crop magnification `≤ 3×` and only when a bubble is under 3.4 px; structurally: windowed ring sampling (`±2 px` search window), local paper annulus (`1.35 r`), and compensation-free mark counting (marks never decide geometry).

---

## References (positioning)

1. Q. Hafeez, W. Aslam, M. I. Lali, S. Ahmad, M. Alqahtani, M. Shafiq, "An Enhanced Fault Tolerance Algorithm for Optical Mark Recognition Using Smartphone Cameras," *IEEE Access*, vol. 12, pp. 121305–121319, 2024. — state-of-the-art smartphone fiducial-based pipeline; orientation trusted after detection.
2. Q. Hafeez et al., "Fault Tolerant Optical Mark Recognition," *CMC*, vol. 74, no. 2, pp. 3829–3847, 2023. — Hough-based skew correction; artifact taxonomy of fiducial fragility.
3. Y. Hernández-Mier, M. A. Nuño-Maganda, S. Polanco-Martagón, G. Acosta-Villarreal, R. Posada-Gómez, "Unsupervised Optical Mark Recognition on Answer Sheets for Massive Printed Multiple-Choice Tests," *J. Imaging*, 11(9):308, 2025. — classical CV, 96.15 % error-free sheets / 99.95 % answers; argues classical pipelines' per-stage explainability.
4. A. Nath, A. Nath, B. J. Omotayo, "A Web-Based Automated OMR Evaluation System Using YOLO and Image Processing Techniques," *IJERT*, 14(12), 2025. — YOLOv8 region detection + classical fill-density; internal geometry assumed upright after detection.
5. S. Mondal, P. De, S. Malakar, R. Sarkar, "OMRNet: a lightweight deep learning model for optical mark recognition," *Multimedia Tools and Applications*, 2023. — learned per-box confidence (avg. 95.96 %) with per-layout training data; no orientation reasoning.
