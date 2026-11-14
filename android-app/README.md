# OMR Checker — Android app

An offline Android app for answer sheets with **15, 25, 30, 40 or 50
questions**. A teacher keeps one key per subject and exam — with its own marks
per question — and then either points the live camera at a sheet, or scans a
photo. Everything is stored in the phone's own SQLite database; nothing is
uploaded.

**APK:** [OMR-Checker-pilot.apk](OMR-Checker-pilot.apk) (version 1.2-subjects-db)

## Install on the phone

1. Copy `OMR-Checker-pilot.apk` to the Android phone (USB, Drive, or any file
   transfer method).
2. Open it. Android will ask to allow installing from that app; allow it for
   this one install.
3. The app is called **OMR Checker** and needs Android 7.0 or newer.

The build is signed with a local debug key. It is for testing, not a Play Store
release. The camera permission is requested the first time **Start live scan**
is tapped. Upgrading over the previous pilot keeps the database and its keys.

## Subjects, exams and marks

The first tab manages the database.

- **Add a subject** (for example *Bangla 1st paper*). Tap **Rename** or
  **Delete** to change it; a second tap on Delete confirms, because a WebView
  cannot show a native confirm dialog.
- Each subject holds any number of **exams**. When adding one, pick the question
  count that matches the sheet you will print: 15, 25, 30, 40 or 50.
- Deleting a subject removes its exams and their answer keys (SQL `ON DELETE
  CASCADE`), so no orphan rows are left behind.

Open an exam to edit its key. Every question row has the four options and a
**marks box** — set 1 mark, 2 marks, halves, whatever the paper uses. Marks are
stored per question, so totals are exact.

The key is only written when you press **Save to phone**. Until then the edits
are on screen only; the button stays disabled when nothing has changed. Answers
can be tapped one by one or pasted (`223432…`, `2,2,3,4,…`, or `A B C D`).

## Question counts and printed sheets

`assets/sheet-designs.js` holds the printed sheet, measured from the PDFs in
`layout-omr/`. It is **one design**: 15 question columns per row, each column a
stack of four bubbles A–D, and a new row of 15 questions underneath when the exam
is longer. Nothing else changes with the count — the columns, the bubble size
(3.45 mm radius) and the spacing (9.01 mm A→D, 43.45 mm between question rows) are
the same on every sheet:

| Count | Rows | Bubbles |
|---|---|---|
| 15 | 1 row of 15 | 60 |
| 20 | 2 rows: 15 + 5 | 80 |
| 25 | 2 rows: 15 + 10 | 100 |
| 30 | 2 rows: 15 + 15 | 120 |
| 40 | 3 rows: 15 + 15 + 10 | 160 |
| 45 | 3 rows: 15 + 15 + 15 | 180 |
| 50 | 4 rows: 15 + 15 + 15 + 5 | 200 |
| 60 | 4 rows: 15 + 15 + 15 + 15 | 240 |

The printed layout and the reader come from the **same numbers**, so a sheet
printed from the app always lines up. Use **Print sheet** on an exam row and
print at **100% / actual size** — never "fit to page". Android's print service
offers a printer or *Save as PDF*.

A photo is measured as it is first. Only when that finds almost nothing does the
reader line the printed bubble rings up with the template (a search of about
±1.2 mm, with a small size correction) and keep the better of the two reads, so a
page outline that is a millimetre out no longer skews every bubble score.
The sheet carries the title, name/class/subject lines, the answer table with its
numbers and A–D letters, and the Student ID, Subject Code and Total Marks boxes
down the right side, exactly where the paper sheets put them.


## Answer key and grading

- Grading is per question: correct counts the question's marks, wrong/blank/
  double-marked/review count as zero.
- Questions with no key show as `NO KEY` instead of being guessed, and the
  maximum only includes questions that have an answer.
- A scan does not need a complete key; a partly filled key simply grades fewer
  questions.

## Live AR scan

1. Open an exam, check the key, then open **Live**.
2. Tap **Start live scan** and allow the camera.
3. Hold the whole pink frame in view. The status line shows
   `Reading… 40%` while several frames are averaged; it reaches 100% and the
   mark locks in.
4. Green rings = correct, red = wrong, blue = the key answer, purple =
   double-marked, amber = blank/review. The total is drawn in the printed mark
   box and spoken aloud.

What changed for a moving hand:

- **Multi-frame fusion.** Each bubble keeps a running average of its fill plus a
  hysteresis band. A decision changes only after *three consecutive* frames push
  it the same way, and the sheet is only reported when nothing has moved for two
  frames.
- **No full-page warp while scanning.** Bubbles are sampled straight from the
  camera frame through the sheet homography
  (`readProjected`), which is what makes ~10 frames per second affordable.
- **Corner smoothing.** The four corners are EMA-filtered; small tremors no
  longer re-solve the geometry, while a real move snaps and clears the buffer.
- **Motion gate.** Frames taken while the sheet is moving are skipped instead of
  being read, and the status says *Hold still…*.
- **Too far away is rejected** instead of guessed: if a bubble is under ~2 px in
  the frame the app asks you to move closer.

The old behaviour — a single reading as soon as the corners looked stable — is
what made a slightly moving sheet produce a wrong mark.

## Photo scan

The **Photo** tab keeps the still-image path: take or choose a photo, the sheet
is located from the pink frame (or the page outline), warped to the canonical
page and read once, with a full breakdown and a marked preview.

## Where the data lives

`OmrDatabase` (SQLite, `omr.db`) with foreign keys enabled:

```sql
subjects(id, name UNIQUE, created_at)
exams(id, subject_id -> subjects, name, template_id, question_count,
      option_count, created_at, UNIQUE(subject_id, name))
answer_keys(exam_id -> exams, question_no, correct_option, marks)
```

`DatabaseBridge` exposes it to the WebView as JSON through `window.OmrDatabase`.
Inside a plain browser the same API is emulated with localStorage
(`assets/store.js`) so the screens can be developed without a device.

Upgrades are additive only — `onUpgrade` must never drop the tables, or a
teacher loses every class list.

## Verification done

- `node test/pipeline-test.js` — renders each design (15/25/30/40/50), fills a
  key and grades it:
  - every question correct, total equals maximum, a different key scores 0, a
    blank sheet scores 0;
  - the live projected read finds every mark in a 1280-px camera frame;
  - per-question marks produce the expected maximum (15 questions with five
    2-mark questions = 20);
  - fusion settles, ignores two blurred frames, and does not flip an answer on a
    swapped frame. **All checks pass.**
- `bash build_apk.sh` — compiles, dexes, aligns, signs and verifies the APK
  (version 1.2-subjects-db).
- The page loads without console errors and the subject/exam flow works in the
  browser fallback store.

A physical phone camera, printed sheets at each count, and a real classroom
stack still need testing. The synthetic renders are geometry checks, not a
substitute for paper.

## Limits of this build

- Results are not saved; the database holds subjects, exams, keys and marks.
- No student ID, CSV export or result history yet.
- The 30-question design can only be printed from the original PDF.
- Live mode needs the pink frame reasonably large and in focus.
- The spoken phrase is English only.
- Sheet geometry changes need a new entry in `sheet-designs.js`; the engine reads
  any template with the same shape.

## Build

Prerequisites on this machine:

- Android SDK with build-tools and a platform (the script selects the newest).
- Android Studio's bundled JDK at
  `C:\Program Files\Android\Android Studio\jbr`.

```bash
bash android-app/build_apk.sh
node android-app/test/pipeline-test.js
```

The script compiles resources with `aapt2`, compiles the Java shell, dexes with
`d8`, aligns, signs with `debug.keystore`, and writes
`android-app/OMR-Checker-pilot.apk`. The first build creates the debug keystore;
keep it so later APKs update the installed app instead of requiring an
uninstall.

The app is plain Java plus local HTML/CSS/JS; there is no Gradle project and no
third-party runtime dependency.
