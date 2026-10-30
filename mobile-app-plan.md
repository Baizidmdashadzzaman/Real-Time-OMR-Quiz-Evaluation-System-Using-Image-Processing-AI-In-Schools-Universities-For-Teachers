# Mobile OMR — Android Product & Engineering Plan

**Status:** proposal for review  
**Date:** 30 October 2026  
**Scope:** Android-only teacher app for creating MCQ exams, printing OMR sheets,
scanning/evaluating them, and storing results.  
**Starting point:** the working Python scanner in this repository.

---

## 1. The short answer

Build a **native Android app in Kotlin + Jetpack Compose**, with the OMR
computer vision running **on the device** through **OpenCV for Android**.

Do **not** wrap the existing Python program as the app. Keep Python on the
**server** for three jobs it is good at:

1. generating print-perfect PDFs (question paper + OMR answer sheet);
2. being the reference implementation for regression tests while the Kotlin
   engine is ported;
3. re-evaluating a disputed sheet on the server from the original capture.

React Native is a legitimate second choice, but only if the UI is built in
React Native and the camera/OpenCV work still lives in a **native Kotlin
module**. There is no useful way to run OpenCV on live camera frames in
JavaScript, so React Native does not remove the hard native work; it adds a
second runtime around it.

A Python-on-Android wrapper (Chaquopy, Kivy, BeeWare, PySide6) is suitable for a
prototype, not for a production school app. The reasons are concrete, not
stylistic: PDF rendering with PyMuPDF has no practical Android wheel, Chaquopy's
OpenCV is stuck on an old 4.5 build, camera integration still has to be written
in Kotlin/Java, and the resulting app is larger and harder to maintain.

The recommended shape:

```text
Android app (Kotlin/Compose)
  ├── exam builder + roster + results
  ├── CameraX scanner
  ├── OpenCV OMR engine (on-device, offline)
  ├── Room database (local source of truth)
  └── print/share PDF
             │ HTTPS
             ▼
Supabase (Auth + Postgres + Storage + RLS)      Python PDF service
  └── teachers, classes, students, exams,         └── WeasyPrint + Noto Bengali
      submissions, answers, audit images              + template geometry
```

MVP can be **local-data** (Room database, no teacher account, export/backup).
It still calls the PDF service to generate final papers, so that one step needs
a connection; scanning, grading, review, and export work offline. Cloud sync is
a later phase, not a prerequisite for grading a classroom set of papers.

---

## 2. What already exists and what carries over

The current repository is a working reference implementation, not throwaway
code. These pieces are valuable:

| Existing piece | Role in the mobile system |
|---|---|
| `omr/template.py` + `templates/q-paper.json` | Exact bubble geometry in millimetres. The app consumes the JSON; only the server compiles new PDFs. |
| `omr/geometry.py` | Frame/page detection, orientation, homography. Port to Kotlin/OpenCV. |
| `omr/reader.py` | Bubble features, per-sheet GMM calibration, question states. Port to Kotlin. |
| `omr/grade.py` | Answer key, per-question marks, policies, totals. Port to Kotlin; the logic is small. |
| `omr/overlay.py` | Green/red/blue result drawing. Reimplement in Compose Canvas; AR later. |
| `scripts/read_image.py` | Golden-output harness. Keep it to generate expected JSON for the Kotlin tests. |
| `scripts/make_key.py` | The answer-entry UX concept. In the app this becomes part of the exam builder, not a JSON editor. |
| `scripts/selftest.py` + samples | Regression corpus seed. Expand it substantially. |

The Python implementation should stay runnable for the whole project. Every
Kotlin result is compared against it. When a field sheet looks wrong, the same
capture can be re-run through Python to decide whether the app or the sheet is
at fault.

---

## 3. Platform decision

### 3.1 Native Kotlin (recommended)

**Why it wins for this product**

- Android-only is an explicit requirement. There is no cross-platform payoff
  to paying for a cross-platform runtime.
- The hard parts are camera buffers, OpenCV, file/print APIs, background sync,
  and battery/memory behaviour. Kotlin is the shortest path to all of them.
- One language, one debugger, one build. No JS/native bridge to diagnose when a
  frame is dropped or a bitmap is copied.
- OpenCV's Java/Kotlin API is sufficient for this workload: a few hundred
  bubbles and one homography per capture, not a neural network.
- Smaller app, fewer native libraries, simpler Play Store compliance.

**Stack**

- Kotlin, Jetpack Compose, Navigation Compose, Material 3.
- CameraX for preview + `ImageAnalysis` + `ImageCapture`.
- OpenCV Android `org.opencv:opencv` (4.14.x; evaluate 5.x only after checking
  16 KB page alignment).
- Room + SQLCipher for encrypted local data.
- WorkManager for sync/export jobs.
- Kotlinx serialization, Ktor/Retrofit, Coil, DataStore.
- ML Kit barcode scanning for the exam QR (optional; a fallback manual
  selection must exist).
- Min SDK 26, target/compile SDK 36 (Play requirement in 2026).

**Cost:** the UI is forms and lists; Compose is fast to build. The CV engine is
the same work in any stack.

### 3.2 React Native (viable alternative, with conditions)

React Native can absolutely host this app, but the architecture must be honest:

- Use **React Native 0.86+ / New Architecture**. Legacy architecture is gone.
- Use **react-native-vision-camera v5** with a frame output. In v5 the API is
  `useFrameOutput`, not the old `useFrameProcessor`; frames are worklet-owned
  and must be disposed.
- Put OpenCV behind a **native Kotlin/C++ Nitro module** (or
  `react-native-fast-opencv`, which is JSI + the OpenCV C++ API). Never attempt
  OpenCV.js/WASM for live frames: it copies every camera buffer into a WASM
  heap and Hermes/WASM is not a production path for this.
- Stream YUV, not RGB, and downscale before contour detection. A 4K RGB frame
  is roughly 34 MB; copying that per frame at 30 fps is not viable.
- Use an App Bundle and restrict ABIs. The full OpenCV AAR is roughly 116 MB
  (4.13) / 146 MB (5.0) before per-ABI delivery; the installed download is much
  smaller, but CI and local APKs are not.
- Verify **16 KB page-size alignment** for every shipped `.so`. Play's deadline
  for updates without 16 KB support is 1 February 2027; OpenCV builds from older
  NDKs are a known failure point.
- Expo Go cannot be used. You need a development build / prebuild.

**When React Native is the right call**

- The team is TypeScript-first and would struggle to staff Kotlin.
- iOS is a funded requirement within the next 12 months.
- The app is mostly dashboards/forms and the CV module can be owned as a
  separate native library.

**When it is the wrong call**

- Android-only, one small team, CV and print reliability are the product.
  You will write the same Kotlin OpenCV engine either way, then debug it through
  a bridge.

If React Native is chosen, the plan in this document barely changes: the OMR
engine, PDF service, database, and print strategy are identical. Only the UI
layer and the camera host change.

### 3.3 Python wrapper (not recommended for production)

| Option | What works | What breaks | Verdict |
|---|---|---|---|
| **Chaquopy** (CPython inside a normal Android app) | Python 3.10–3.15, NumPy, OpenCV wheel, CameraX frames can be passed to Python | OpenCV wheel is **4.5.1.48**, old API; **PyMuPDF has no usable Android wheel**; camera/UI still Kotlin; JNI copies and GIL; larger app | Best Python option, still not the right app core |
| **Kivy + python-for-android** | OpenCV 4.12 built from source; AAB output | Camera integration is DIY; the mature `camera4kivy` glue is archived (2023); UI/UX and Play polish are a large extra project | Prototype only |
| **BeeWare / Briefcase** | Python UI, AAB, CI-tested | Toga camera is still-photo oriented, no live frame stream into `cv2`; PyMuPDF absent | Wrong tool for live scanning |
| **PySide6 Android** | Qt UI, OpenCV via p4a recipes | Build host must be Linux/macOS; huge runtime; pinned NDK/interpreter; camera via Qt/JNI | Overkill |

**Conclusion:** wrapping Python does not avoid native work. You still need
Kotlin/Java for camera, permissions, printing, and Play packaging, while giving
up current OpenCV and a practical PyMuPDF path. Use Python on the server and
for the reference engine.

### 3.4 Decision matrix

| Criterion | Native Kotlin | RN + native CV | Python embedded |
|---|---|---|---|
| Android-only fit | Best | Good | Poor |
| Camera/OpenCV control | Best | Good, through native module | Poor/DIY |
| UI speed for forms | Fast | Fast | Slow |
| App size / native risk | Lowest | Medium | High |
| Play Store path | Straightforward | Straightforward, more native libs | Fragile |
| iOS later | No | Yes | No |
| Team fit if JS-first | Requires Kotlin | Best | Requires Python |
| **Verdict** | **Build this** | Choose only for JS team/iOS | Prototype only |

### 3.5 If you are a solo Python developer

This is the practical question behind the platform choice. None of the mobile
options let you keep Python as the app's main language without a large tax:

- **Kotlin:** you learn Compose and CameraX, then port the OMR math. The port is
  mechanical because the Python reference and JSON schema already exist. Expect
  roughly 1,500–2,500 lines of Kotlin for the engine and scanner glue, spread
  over 4–6 weeks.
- **React Native:** you also learn TypeScript/React Native state and native
  modules, and you still write the OpenCV module in Kotlin. More surface, not
  less.
- **Chaquopy/Kivy:** you keep some Python math, but still write camera, UI,
  permissions, printing, and packaging in Kotlin/Java or a niche framework;
  you also accept old OpenCV and no practical PyMuPDF.

A workable solo path:

1. Spend 2–3 weeks on Kotlin + Compose basics and one CameraX sample.
2. Build the OMR engine as a JVM/Android library with a command-line harness
   before touching the UI; compare against Python on every image.
3. Build the PDF service in Python first; it is familiar ground and unblocks
   paper testing.
4. If the budget allows, contract an Android developer for the camera/port
   phase while you own the paper pipeline and backend. This is usually faster
   than learning both the CV port and Compose under schedule pressure.

Android Studio on Windows is sufficient; no macOS is needed for Android-only.

---

## 4. Product scope

### 4.1 MVP (what ships first)

A single teacher can, on one Android phone/tablet:

1. create a class and import a student roster (CSV or manual);
2. create an MCQ exam offline: title, subject, date, instructions, questions,
   four options, correct option, marks per question;
3. publish it, generating a print-ready question paper PDF and a fixed OMR
   answer-sheet PDF (needs the PDF service);
4. print/share the PDFs;
5. scan answer sheets with the camera, one at a time or in a batch;
6. grade on-device against the published key;
7. review uncertain/multi/blank answers and correct them;
8. see per-student and class results, and export CSV/PDF;
9. keep everything in a local database, with a manual backup/export.

No teacher account is required for this MVP; the PDF endpoint can use a device
token until cloud sync arrives. "Saved in a database" means a real local SQLite
database on the device, not a text file.

Authoring, scanning, grading, review, and export are offline. Final PDF
generation is the one online step: if the teacher is offline, the exam stays a
draft and the PDF is generated when a connection returns. A fully on-device PDF
renderer is a later enhancement, not an MVP dependency.

### 4.2 Explicit non-goals for MVP

- Free-form drag-and-drop paper designer.
- Arbitrary page layouts or bubble geometry created on the phone.
- iOS, web admin, student-facing app.
- Online exams, timer/lockdown browser, essay grading.
- AI answer grading or handwriting recognition.
- Real-time multi-device collaboration.
- Payments/subscriptions.
- AR overlay (the existing Python AR is a phase-2 feature; the core result
  overlay is enough for MVP).

### 4.3 Later phases

- Cloud sync and multi-device (phone + tablet).
- Sets A/B with shuffled questions/options.
- 50/100-question templates.
- Question images, equations, rich text.
- Per-student QR sheets.
- Direct IPP printing with scaling disabled.
- AR mark overlay and voice announcements in the app.
- School/organisation accounts and teacher roles.
- Analytics across terms/classes.

---

## 5. Users and end-to-end flows

### 5.1 Teacher setup

1. Install the app, accept privacy notice.
2. Create a class (name, section, year).
3. Add students manually or import CSV: `student_code, roll, name`.
4. Optionally create a school/profile.

### 5.2 Create an exam

1. **Details:** title, subject, class, date, duration, instructions.
2. **Capacity:** choose an OMR template tier that fits the question count
   (30, 50, 100). The template fixes the bubble geometry; the teacher never
   moves bubbles.
3. **Questions:** enter or import questions. For 30–50 questions, bulk import is
   essential; typing 50 MCQs on a phone is not acceptable UX.
4. **Answers and marks:** pick the correct option (1–4) and marks per question;
   the UI mirrors the existing key wizard. Show a running total.
5. **Preview:** generate a draft PDF, check pagination and Bengali text. If
   offline, save the draft and generate the preview when connected.
6. **Publish:** freeze the exam version, generate the final PDFs and the answer
   key version. Editing after publish creates a new version and requires
   reprinting.

### 5.3 Print

1. App downloads/caches:
   - question paper PDF (variable pages);
   - OMR answer sheet PDF (one page per student);
   - optional teacher answer-key PDF (never given to students).
2. Teacher prints the question paper once and the answer sheet N times.
3. App shows a print checklist:
   - A4, actual size / 100%, do not "fit to page";
   - black-and-white is fine if the frame remains dark enough;
   - print the calibration page once per printer and measure the 100 mm bar.
4. App stores the printed template version and key version.

### 5.4 Exam day

1. Students write name/roll and fill the student-ID bubbles.
2. Teacher opens the exam and taps **Scan**.
3. Camera preview shows frame detection and a readiness state.
4. Capture is automatic when the sheet is sharp, stable, and inside the guide;
   a shutter button is always available.
5. The result appears in under a second: total, correct/wrong/blank/unclear.
6. The sheet is saved locally with the capture and result image.
7. The scanner re-arms when the sheet is removed or replaced; the same sheet is
   not graded twice unless the teacher forces it.

### 5.5 Review and results

1. Submissions with `REVIEW`, `MULTI`, blank, or low confidence go to a queue.
2. The teacher sees the bubble crop, the detected answer, and the key answer;
   tapping changes the answer, with an audit record.
3. The teacher can mark a submission unassigned/duplicate and reassign a student.
4. Class results: score list, distribution, per-question accuracy, export CSV,
   share a result PDF.
5. If online, the sync worker uploads exams, submissions, answers, and images;
   if offline, everything remains usable.

---

## 6. System architecture

```text
┌──────────────────────────── Android app ────────────────────────────┐
│ Compose UI                                                          │
│  Exams · Roster · Scan · Review · Results · Settings                │
│                                                                     │
│  ExamRepository        ScanRepository        ResultRepository       │
│        │                    │                       │               │
│  Room DB (encrypted)   OmrEngine (Kotlin+OpenCV)   Export/PDF viewer│
│        │                    │                       │               │
│  SyncWorker (WorkManager)  CameraX (YUV analysis, capture)          │
│        │                    │                                       │
│  PDF client            ML Kit QR (optional)                         │
└────────┼────────────────────┼───────────────────────────────────────┘
         │ HTTPS              │ local files
         ▼                    ▼
┌──────────────────┐   ┌──────────────────────────────┐
│ Supabase         │   │ Python PDF service            │
│ Auth             │   │ FastAPI + WeasyPrint          │
│ Postgres + RLS   │   │ Noto Sans Bengali             │
│ Storage (images) │   │ template geometry + QR        │
└──────────────────┘   └──────────────────────────────┘
         │
         ▼
  Python OMR reference / audit re-evaluation
```

**Key principle:** the phone is the source of truth for scanning and grading.
The network is an enhancement, not a dependency. In the local-data MVP the
Supabase box is absent; the app talks only to the PDF service, and only when a
paper is generated or downloaded.

---

## 7. Paper and OMR design

### 7.1 Separate the question paper from the answer sheet

Do not keep the current combined layout for the mobile product. A 30-question
paper with long Bengali questions will not fit reliably around a fixed bubble
grid. Use:

- **Question paper:** normal A4 pages, variable length, no machine-readable
  geometry required.
- **OMR answer sheet:** exactly one fixed-layout page per student, generated
  from a versioned template.

This makes pagination, printing, and reprints much simpler, and lets the same
answer sheet serve multiple question papers of the same capacity.

### 7.2 Fixed template families

The teacher chooses a capacity, not a layout:

| Template | Questions | Options | Status |
|---|---|---|---|
| `omr-30-v1` | 30 | 4 | MVP, based on current `q-paper` geometry |
| `omr-50-v1` | 50 | 4 | Phase 3; same frame/ID design |
| `omr-100-v1` | 100 | 4 | Later; two pages or smaller bubbles |
| `omr-30-a-v1` | 30 | 5 | Only if a real exam needs five options |

Each template has:

- page size A4 (210 × 297 mm);
- outer alignment frame (current pink frame, or a high-contrast black frame);
- bubble rows grouped in bands/columns;
- question numbers printed next to rows;
- student-ID grid (8 digits, 0–9 bubbles per column);
- exam QR (exam id, template id/version, set, checksum);
- name/roll writing area;
- instructions and a calibration ruler;
- template id/version in human-readable text.

**Never** let the app stretch, reflow, or auto-fit bubble geometry. A new design
is a new template version compiled on the server and validated before release.

### 7.3 Student identity

A result cannot be saved to a student without an identifier. Three options:

| Option | How | Trade-off |
|---|---|---|
| **Roster order** | Scan sheets in class roll order; app assigns next unassigned student | No template change, but one out-of-order sheet corrupts the rest |
| **Student-ID bubbles** | 8-digit code filled by student (recommended) | One extra field to fill; misbubbling possible |
| **Per-student QR** | Each student's sheet printed with a unique QR | Best identity integrity; printing/collation is harder |

**Recommendation:** student-ID bubbles on a generic sheet for MVP, with an
"unassigned" queue and duplicate detection. Add per-student QR as an option for
schools that can manage personalized printing.

If the ID grid is blank or invalid, the submission is saved as unassigned and
must be matched manually. Never silently attach it to the next student.

### 7.4 Sets A/B

MVP: one set. Later:

- shuffle question order and/or option order with a seeded algorithm;
- print a set letter bubble and encode the set in the QR;
- keep a separate key per set;
- the answer sheet remains the same template;
- results are comparable because the key maps printed question numbers back to
  the canonical question ids.

### 7.5 Print-accurate PDF generation

Use **WeasyPrint** on the server (HTML/CSS paged media) with static
**Noto Sans Bengali** fonts. WeasyPrint uses Pango/HarfBuzz, so Bengali
conjuncts and vowel signs shape correctly; ReportLab would require a custom
HarfBuzz shaping pass.

Layout rules:

- `@page { size: 210mm 297mm; margin: 0; }`; inner content uses explicit
  padding, not browser defaults.
- Keep all critical marks ≥ 8 mm from the physical edge. Cheap printers have
  non-printable margins; Android's print path may clip, not scale, at the
  printer's minimum margins.
- Embed fonts (subset) and verify with `pdffonts`/`pdfinfo`.
- Add a 100 mm calibration bar and a 10 mm grid in the footer.
- QR: 20 × 20 mm, high error correction, payload versioned.
- Output A4 must measure 595.276 × 841.89 pt. Validate in the PDF service tests.

Do not generate the final PDF with headless Chromium unless there is a strong
reason. Chromium defaults to US Letter, and `preferCSSPageSize`/`@page` mistakes
silently scale the page. WeasyPrint gives deterministic A4 from CSS.

### 7.6 Printing from Android

Android's `PrintManager` does not expose an "actual size / 100%" switch. The
print service and printer can fit the page to the printable area. Consequences:

- the PDF page size being correct does **not** guarantee the physical print is
  1:1;
- a uniform scale (a few percent) is largely absorbed by the homography, but
  cropping, non-uniform scaling, or aspect-ratio changes are not;
- always show the calibration bar and a print checklist;
- test on the actual printers used by the pilot schools.

For exactness later, submit directly over IPP with `print-scaling = none` and
`media = iso_a4_210x297mm`, after checking the printer's supported attributes.
That is a phase-3/4 option, not MVP.

### 7.7 Paper versioning

- `template_version` is immutable once used for a real exam.
- `exam_version` increments on any content/key change after publish.
- Scans store both versions and the key checksum.
- Changing an answer key after printing is allowed only as a new key version,
  with an audit event and a clear "re-grade existing scans?" prompt. Existing
  results are never silently rewritten.
- Old templates remain downloadable so old papers can be reprinted.

---

## 8. Data model

Use UUIDs (UUIDv7 where possible), UTC timestamps, and explicit status fields.
The local Room schema mirrors the server subset; server-only tables are marked.

### 8.1 Core tables

| Table | Important columns | Notes |
|---|---|---|
| `schools`* | id, name, region | Optional for single-teacher MVP |
| `teachers`* | id (= auth uid), school_id, name, phone, role | Server auth owner |
| `classes` | id, teacher_id, name, section, academic_year | RLS root |
| `students` | id, class_id, student_code, roll_no, name, active | `student_code` matches ID bubbles |
| `templates` | id, version, capacity, options, page_mm, geometry_json, checksum | Read-only on device |
| `exams` | id, teacher_id, class_id, template_id, title, subject, exam_date, status, active_questions, total_marks, version, published_at | `DRAFT → PUBLISHED → SCANNING → CLOSED` |
| `exam_questions` | id, exam_id, ordinal, prompt, options_json, correct_index, marks, active | Key lives here; never in student PDF |
| `exam_sets` | id, exam_id, label, mapping_json | Phase 2 |
| `submissions` | id, exam_id, student_id, set_label, template_version, key_version, status, score, maximum, correct, wrong, blank, multi, review, capture_path, result_path, captured_at, device_id, reviewed_at | One row per scanned sheet |
| `submission_answers` | submission_id, q, selected_json, state, confidence, awarded, reason, reviewed | Per-question detail |
| `review_events` | id, submission_id, q, old_value, new_value, teacher_id, created_at | Append-only audit |
| `devices`* | id, teacher_id, model, app_version, last_seen | Sync/debug |
| `sync_outbox` (local) | id, entity, entity_id, operation, payload, attempts, next_attempt_at | WorkManager queue |

### 8.2 Constraints that matter

- Unique `(exam_id, student_id)` for assigned submissions; partial unique index
  so multiple unassigned sheets can wait for matching.
- `submissions` are immutable after sync; corrections create `review_events` and
  update the derived answer/score in a transaction.
- `exam_questions.correct_index` is 0-based internally, 1-based in the UI.
- `active_questions` allows a 30-question test on a 50-question template without
  changing geometry; grading ignores inactive rows.
- All score fields are derived and re-computable from answers + key version.
- Store `engine_version` and `key_checksum` on each submission for audit.

### 8.3 Local vs cloud

MVP: every table except `schools`, `teachers`, and `devices` exists locally.
Cloud adds teacher ownership, RLS, cross-device sync, and image backup.

---

## 9. Backend, sync, and security

### 9.1 Recommendation: Supabase, introduced in phase 5

Supabase is the best fit for this data shape:

- Postgres, not a document store: exams, questions, rosters, scores, and
  analytics are relational.
- Row-level security (RLS) expresses "a teacher sees only their classes/exams"
  in SQL and enforces it in the database.
- Auth, private object storage, and signed URLs are included.
- Data export is plain `pg_dump`; no proprietary document format.

Caveats to plan for:

- The **free tier is not enough** for scanned images: 500 MB database, 1 GB
  storage, projects pause after a week of inactivity. Budget **Supabase Pro
  (~$25/month)** for a real pilot.
- Supabase has no built-in offline sync. For MVP, use a **Room outbox** with
  client-generated UUIDs and idempotent upserts. If multi-device sync becomes a
  hard requirement, adopt **PowerSync** (Postgres → per-user SQLite) rather than
  hand-rolling conflict resolution forever. `supabase-offline-kt` exists but is
  pre-1.0; do not make it a critical dependency yet.
- Database backups do **not** include Storage objects. Keep an independent copy
  of scanned images or accept that originals can be lost.

Firebase is the alternative if the team wants zero backend ops and live sync
out of the box. It gives offline-by-default Firestore and Cloud Storage, but the
document model and Security Rules make relational grade queries and
teacher→class→exam→submission access awkward and read-billed. For this app,
Postgres + RLS is the cleaner long-term model.

### 9.2 Sync strategy

- **Local-first:** scan, grade, review, and export work with no network.
- **Outbox:** each write gets a UUID and a monotonic local revision; sync pushes
  in dependency order (exam → submission → answers → images).
- **Idempotency:** server `PUT` by client UUID; retries cannot duplicate rows.
- **Conflicts:** published exams are immutable on the server; submissions are
  append-only; review events are append-only. Draft exams use last-write-wins
  with a conflict warning.
- **Images:** compress to JPEG/WebP quality ~85, upload in the background,
  checksum, retry with exponential backoff. Keep local until the server confirms.
- **WorkManager:** constraints `NETWORK_CONNECTED`, backoff, unique work name
  per teacher. Expect Doze to delay sync on cheap tablets; never block grading.
- **Status UI:** "Saved on this device", "Waiting to sync", "Synced", with a
  manual retry.

### 9.3 Security and privacy

Students are minors; treat the data accordingly.

- Auth: email/phone OTP for teachers; school invites later. No student accounts
  in MVP.
- RLS on every table, with `revoke` before grants; test policies with pgTAP.
- Storage: private buckets, path scoped by `teacher_id/exam_id/submission_id`,
  short-lived signed URLs. No public buckets.
- Local DB: SQLCipher/Room open-helper with a key stored in Android Keystore.
- Transport: TLS only; certificate pinning optional.
- Answer keys: stored locally encrypted; downloaded only after publish; never
  embedded in a student PDF.
- Data minimisation: no address/phone for students; only name, roll/code.
- Retention: configurable (e.g. images 90 days, results 3 years); export and
  delete flows; audit events for review changes.
- Analytics/crash reporting: off by default; opt-in and no student names.
- If cloud is used, check local data-residency requirements and sign the
  provider's DPA. Supabase offers region choice; Firebase offers regional
  buckets/databases. Get this reviewed before a school pilot.

---

## 10. Android app architecture

### 10.1 Modules

```text
:app
:core:model          entities, value objects, JSON schema
:core:db             Room, SQLCipher, DAOs, migrations
:core:omr            Kotlin + OpenCV engine (no UI)
:core:network        Supabase/PDF clients, sync DTOs
:core:design         theme, components
:feature:exams       exam list/editor/preview
:feature:roster      classes/students/import
:feature:scan        camera, batch scanning, result overlay
:feature:results     class/student results, review, export
:feature:settings    printer test, storage, account
```

`:core:omr` must be testable without an Activity: input `Mat`/`Bitmap`, output a
result object identical to the Python JSON schema.

### 10.2 Camera pipeline

1. CameraX Preview + ImageAnalysis (`YUV_420_888`, target 1280×720,
   `STRATEGY_KEEP_ONLY_LATEST`) + ImageCapture (max resolution).
2. Downscale analysis frame to ~640 px wide; convert YUV→RGB/gray in native code.
3. Detect pink/black frame or page contour; require four corners, reasonable
   area, and no clipping.
4. Compute orientation and page quality (blur, glare, brightness).
5. Show guide + state: `READY`, `MOVE CLOSER`, `HOLD STEADY`, `TOO BLURRY`,
   `NO SHEET`.
6. Auto-capture after ~300 ms of stability, or on shutter.
7. Process the full-resolution capture on a single background executor:
   homography → warp to canonical 200 DPI → read bubbles → grade.
8. Save `capture.jpg`, `warped.png`, `result.jpg`, `result.json` atomically.
9. Re-arm only after the sheet is removed or replaced; `SPACE`/shutter can force
   a re-read.

Never process frames on the main thread. Reuse `Mat` buffers. Close
`ImageProxy` in `finally`. If analysis exceeds the frame budget, drop frames
rather than queueing them.

### 10.3 Screens

- **Home:** recent exams, scan shortcut, sync status.
- **Classes/Students:** list, import CSV, search.
- **Exam editor:** stepper (details → questions → answers/marks → preview).
- **Question editor:** prompt, four options, correct option, marks; bulk import.
- **Preview/print:** PDF preview, print, share, calibration checklist.
- **Scan:** camera, batch counter, last result, review badge.
- **Review:** list of flagged submissions, bubble crop, override.
- **Results:** sortable scores, distribution, per-question stats, export.
- **Settings:** printer test, storage usage, backup/export, account/sync.

### 10.4 Performance targets

| Metric | Target |
|---|---|
| Camera preview | 30 fps preview, analysis ≥ 10 fps on mid-range |
| Detection | < 60 ms per analysis frame at 640 px |
| Full grade | < 500 ms on a mid-range phone (2–3 year old) |
| Capture-to-result | < 1.5 s including save |
| Batch | 100 sheets < 15 min including handling |
| Cold start | < 2 s to home |
| Memory | < 300 MB during scanning |

These are acceptance targets, not guesses to be discovered after launch. Measure
on a low-end device early; the OpenCV work is small, but camera copies and
bitmap allocation are where apps slow down.

---

## 11. Porting the OMR engine

### 11.1 Module mapping

| Python | Kotlin |
|---|---|
| `template.py` (PyMuPDF compile) | Server only; app loads `template.json` |
| `geometry.py` | `FrameDetector`, `PageWarper` (OpenCV) |
| `reader.py` | `BubbleReader`, `Calibrator`, `QuestionResolver` |
| `grade.py` | `Grader` |
| `overlay.py` | Compose Canvas / Bitmap renderer |
| `read_image.py` | `OmrCli` JVM harness for golden tests |

### 11.2 Port order

1. **Template loading and pixel coordinates.** Assert bubble centres/radii match
   Python at 200 DPI.
2. **Warp only.** For a set of hand-picked images, compare the warped canonical
   image against Python's `debug_warped.png` (mean absolute difference, corner
   alignment).
3. **Bubble features and calibration.** Compare per-bubble scores, threshold,
   and state. The GMM is a 2-component 1D fit; implement EM directly rather than
   pulling a large math library.
4. **Question states and grading.** Compare `ANSWERED/BLANK/MULTI/REVIEW`,
   selected options, awarded marks, total.
5. **Frame detection and orientation.** This is the highest-variance part; test
   rotations, crops, glare, and page-only fallback.
6. **Overlay.** Draw from the result object, not from re-reading bubbles.

### 11.3 Golden corpus (the most important test asset)

Build a corpus of at least **300 images**:

- the existing `samples/` and `samples/test-new/` images;
- printed sheets from at least 3 printers;
- 5 phone cameras, including a low-end device;
- angles 0–30°, all four rotations, partial crops, shadows, glare, dim light;
- blank, faint pencil, pen, double-marked, erased, and corrected sheets;
- 30/50-question templates as they are added.

For every image, store:

```text
corpus/<id>/capture.jpg
corpus/<id>/expected.json     # Python output: warp, answers, grade
corpus/<id>/notes.md          # device, printer, lighting, known issues
```

The Kotlin harness must reproduce `expected.json` exactly (or within an
explicit tolerance for confidence values). Any mismatch is a bug or a documented
improvement; never update the expectation just to make the test green.

### 11.4 Packaging OpenCV

- Prefer `org.opencv:opencv:4.14.x` from Maven Central.
- `ndk.abiFilters` for `arm64-v8a` (add `armeabi-v7a` only if old devices are
  required).
- Use an Android App Bundle; Play delivers only the needed ABI.
- Verify 16 KB page alignment before every Play upload:
  `check_elf_alignment.sh` / `llvm-objdump`.
- Do not ship OpenCV 5.x until its AAR and all native dependencies pass the
  16 KB check. Early 5.x builds have had alignment problems.
- Keep a minimal OpenCV build option for later if app size becomes a problem;
  the current pipeline needs only core, imgproc, imgcodecs, and calib3d.

### 11.5 Acceptance for the port

- 100% agreement with Python on the curated corpus for selected answers and
  totals; confidence values within a small tolerance.
- Zero crashes on malformed/blank images; always return a `REVIEW` result rather
  than an exception.
- All results include `engine_version`, `template_version`, and diagnostics.
- The same result JSON schema is used by the app, the server audit, and tests.

---

## 12. PDF and question-paper service

### 12.1 Why a service

Generating print-perfect Bengali PDFs inside the Android app is possible but
expensive and hard to test: fonts, shaping, pagination, and exact A4 geometry
are easier in a controlled Linux container. Paper creation happens before the
exam, usually with network available; scanning happens offline. This is why
the PDF service is the one online dependency in the local-data MVP.

### 12.2 Service design

- Python FastAPI (or Flask) in a Docker image based on Debian with Pango,
  HarfBuzz, fontconfig, and Noto Sans Bengali.
- Jinja2/HTML + CSS Paged Media → WeasyPrint 70 (or current stable).
- Template geometry loaded from the same JSON used by the app.
- Endpoints:
  - `POST /v1/papers` — exam JSON → `{question_pdf_url, answer_sheet_pdf_url, sha256, pages}`;
  - `GET /v1/papers/{id}` — signed download;
  - `POST /v1/audit` — optional re-evaluate a capture with Python.
- Auth: Supabase JWT; service-role key only on the server.
- Store generated PDFs in private object storage with a content hash; regenerate
  deterministically if missing.
- Rate-limit and cap question count/page count.

### 12.3 PDF validation

Automated tests on every deploy:

- page size exactly A4 (595.276 × 841.89 pt);
- fonts embedded and subset;
- no text overflow outside the safe area;
- bubble coordinates within tolerance of the template (parse the PDF drawing
  operators or render and compare);
- Bengali shaping screenshot tests for a fixed set of words;
- QR decodes to the expected exam/template/version;
- calibration bar is 100 mm.

Manual print test: print on 3 printers, scan, and run the existing Python
reader. If the reader cannot find the frame or bubbles, the print pipeline is
not ready.

### 12.4 Deployment

- Container host (Fly.io/Render/Railway/your VPS) with a persistent object store.
- Do not run WeasyPrint natively on Windows in development; use Docker or WSL.
  The user's current PyMuPDF DLL issue is a reminder that native dependencies on
  Windows are fragile.
- Keep the PDF service stateless; generated files live in storage.

---

## 13. Delivery plan

Assumes **2 developers** (1 Android/CV, 1 backend/PDF) plus part-time design/QA.
A solo developer should multiply the calendar by roughly 1.6–1.8.

| Phase | Duration | Deliverable | Exit criteria |
|---|---|---|---|
| **0. Specification & print pilot** | 1–2 weeks | Frozen answer-sheet spec; printed samples; corpus started | 3 printers tested; 100-image corpus; decision on 30/50 capacity |
| **1. OMR engine port** | 4–6 weeks | `:core:omr` + JVM harness | 100% agreement on corpus; < 500 ms per sheet |
| **2. App foundation & exam builder** | 3–4 weeks | Compose app, Room, classes/students, exam editor, key entry | Create and edit an exam fully offline |
| **3. PDF service & printing** | 3–4 weeks | WeasyPrint service, template rendering, print flow, calibration | A4/fonts/QR validated; 3-printer print-scan pass |
| **4. Scanning, grading, review** | 4 weeks | Camera pipeline, batch scan, result overlay, review queue | 100-sheet batch; no duplicate scans |
| **5. Results, export, local DB hardening** | 2–3 weeks | Results screens, CSV/PDF export, backup, encryption | Scan/grade/review/export offline; paper generation via PDF service |
| **6. Cloud sync & auth** | 3–4 weeks | Supabase schema/RLS/storage, outbox, sync status | Two devices; offline/online reconciliation; RLS tests |
| **7. Pilot & hardening** | 3–4 weeks | Play internal build, pilot with 2 teachers, fixes | 500 sheets, accuracy targets, crash-free sessions |

**MVP (phases 0–5, local data):** roughly **17–23 weeks (4–5.5 months)** with
2 developers, assuming the OMR port and PDF service overlap.  
**Full cloud product (0–7):** roughly **23–31 weeks (5.5–7 months)**.  
**Solo:** plan for **7–10 months** for a local-data MVP and **9–13 months** to
the cloud version, again assuming the CV port goes smoothly.

### 13.1 Budget (pilot, excluding labour)

| Item | Cost |
|---|---|
| Google Play developer account | $25 one-time |
| PDF container host | ~$10–20/month |
| Supabase Pro (phase 6 onward) | ~$25/month |
| Domain/email (optional) | ~$20/year |
| Test printers/paper/phones | one-time, school-dependent |
| Fonts | free (Noto OFL) |
| **Total** | **~$10–20/month for local-data MVP; under ~$100/month with cloud** |

### 13.2 Release readiness

- Signed AAB with a backed-up upload keystore; Play App Signing enabled.
- Target/compile SDK 36; all native libraries 16 KB aligned.
- Privacy policy and Play Data Safety form covering student names, IDs, and
  scanned images.
- Content rating and app-access declarations; no ads by default.
- Closed testing with 2 teachers before production; staged rollout after.
- Crash reporting opt-in, no student data in logs.
- Export/delete paths tested before the first school uses the app.

---

## 14. Testing and acceptance

### 14.1 Test layers

- **Unit:** geometry math, bubble features, calibration, grading, marks policy.
- **Golden:** Python vs Kotlin on the corpus; exact answers/totals.
- **Instrumented:** camera lifecycle, permissions, rotation, save/load, Room
  migrations, encrypted DB.
- **PDF:** page size, fonts, QR, overflow, rendered screenshots.
- **End-to-end:** create exam → generate PDF → print (or use a pre-printed
  fixture) → scan → review → export.
- **Field:** real classrooms, real printers, real lighting.

### 14.2 Acceptance metrics for the pilot

| Metric | Target |
|---|---|
| Bubble state accuracy on clean sheets | ≥ 99.5% |
| Silent misgrades (wrong total with no review flag) | 0 in pilot |
| Review rate | < 3% of answers |
| Duplicate scans | 0 unassigned duplicates |
| Capture-to-result | < 1.5 s mid-range |
| Crash-free sessions | ≥ 99.5% |
| Sync success (when online) | ≥ 99% within 5 minutes |
| Teacher completion | 100 sheets scanned and exported without support |

"Silent misgrade" is the key safety metric: an uncertain read must become a
review flag, never a confident wrong mark.

### 14.3 Pilot protocol

1. Pick 2 teachers, 2 classes, 2 printers, 2 phone models.
2. Print a 30-question exam; scan 100 sheets.
3. Manually verify 100 sheets against the paper; log every disagreement.
4. Repeat with 50-question template and a low-end phone.
5. Measure time per sheet, review rate, and sync behaviour in airplane mode.
6. Fix accuracy/UX issues before expanding.

---

## 15. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Printer scaling/cropping | High — bubbles/frame lost | Calibration bar, safe margins, print checklist, 3-printer pilot; direct IPP later |
| CV port diverges from Python | High — wrong marks | Golden corpus, Python oracle, staged port, no expectation edits |
| Student ID misbubbled/blank | High — results attached to wrong student | ID grid + unassigned queue + duplicate detection + manual assign; per-student QR later |
| Offline sync conflicts | Medium — lost/duplicated data | Local-first, UUID idempotency, immutable submissions, append-only reviews, outbox tests |
| Student privacy | High — legal/trust | Local-first, encryption, RLS, private storage, retention/delete, DPA review |
| Play 16 KB/native libs | Medium — release blocked | OpenCV 4.14, verify every `.so`, target SDK 36, test AAB |
| Bengali shaping/fonts | Medium — unreadable paper | WeasyPrint/Pango, Noto static fonts, visual regression tests |
| Scope creep (layout editor) | High — delays | Fixed templates only; new design = new compiled template |
| Camera/lighting variance | Medium — review rate | Quality gates, auto-capture only when stable, manual shutter, field corpus |
| Low-end device performance | Medium — slow scanning | 640 px analysis, frame dropping, buffer reuse, early device testing |
| Cloud cost/pausing | Low–medium | Local-first MVP; Pro plan for pilot; export/backup |
| Answer-key leakage | Medium — exam integrity | Key never in student PDF; encrypted local storage; server RLS |

---

## 16. Alternatives considered

### 16.1 Thin client: upload every sheet to Python

**Fastest to build** because the existing Python engine stays the only engine.
The Android app becomes a camera + upload + results viewer.

**Why not the default:** exam halls often have poor Wi-Fi; upload latency and
privacy are worse; a 100-sheet batch becomes a network dependency. It is a good
**fallback mode** or **audit path**, not the primary scanner.

### 16.2 React Native

Discussed in section 3.2. Viable with VisionCamera v5 + a native OpenCV module;
the CV port is unchanged. Choose it for a JS team or a funded iOS roadmap, not
for Android-only CV work.

### 16.3 Python embedded

Discussed in section 3.3. Chaquopy is the least bad, but old OpenCV, no practical
PyMuPDF, native camera still required, and app-size/JNI costs. Kivy/BeeWare/
PySide6 are worse fits for live scanning. Not recommended.

### 16.4 Fully cloud, no local DB

A web app with a phone camera could work for a demo, but offline grading and
privacy suffer. The local database is cheap and makes the product reliable.

### 16.5 No-code / generic form app

Cannot do frame detection, homography, bubble calibration, print geometry, or
offline batch scanning. Not a fit.

---

## 17. First two weeks (concrete)

### Days 1–2 — freeze the paper

- Decide capacity tiers: 30 for MVP, 50 next.
- Decide student identity: ID bubbles (recommended) vs roster order.
- Decide question-paper/answer-sheet separation.
- Write a one-page answer-sheet spec: A4, frame, bubbles, ID grid, QR,
  calibration bar, safe margins.

### Days 3–5 — print reality check

- Print the current sheet and the proposed answer sheet on 3 printers.
- Print at 100% and at "fit to page"; measure the frame and calibration bar.
- Scan both with the current Python tool; record failures.
- Photograph the results into the corpus.

### Days 6–8 — corpus

- Collect 100 images: rotations, angles, shadows, glare, faint pencil, pen,
  blank, double marks.
- Run `read_image.py` and store `expected.json` for each.
- Document known-hard images.

### Days 9–10 — Kotlin spike

- New Android project; add OpenCV.
- Implement template load + warp + bubble features for one image.
- Compare to Python; measure time on a real device.
- Do not build UI yet.

### Days 11–12 — PDF spike

- WeasyPrint in Docker with Noto Sans Bengali.
- Render one A4 page with question text + answer sheet geometry + QR.
- Verify page size, fonts, Bengali shaping, and print on one printer.

### Days 13–14 — decision gate

- Review spikes: accuracy, performance, print quality, effort.
- Confirm native Kotlin (or switch to RN only if the team constraint is real).
- Turn the phase table into a backlog with estimates.

---

## 18. Open decisions

1. **Capacity:** 30 only for MVP, or 30 + 50 from the start?
2. **Identity:** student-ID bubbles, roster order, or per-student QR?
3. **Device:** phone-first authoring, or recommend a 10-inch tablet for exam
   creation and a phone for scanning?
4. **Cloud timing:** local-data MVP first (recommended), or Supabase from day 1?
5. **Sets:** single set for MVP, or A/B immediately?
6. **Question import:** CSV only, or also paste-from-Word/plain text?
7. **Printing:** system print dialog only, or invest in direct IPP for exact
   scale?
8. **Retention:** how long are capture images kept, and who can delete them?

Answering 1–4 before phase 1 prevents most rework.

---

## 19. Research basis

Platform and packaging:

- React Native + VisionCamera v5 frame output:
  https://visioncamera.margelo.com/docs/frame-output
- VisionCamera native frame plugins:
  https://visioncamera.margelo.com/docs/native-frame-processor-plugins
- OpenCV Android:
  https://docs.opencv.org/4.13.0/d5/df8/tutorial_dev_with_OCV_on_Android.html
- Android 16 KB page sizes:
  https://developer.android.com/guide/practices/page-sizes
- Chaquopy Android docs:
  https://chaquo.com/chaquopy/doc/current/android.html
- Chaquopy package index (OpenCV version):
  https://chaquo.com/pypi-13.1/
- PyMuPDF Android discussion:
  https://github.com/pymupdf/PyMuPDF/discussions/4253
- Kivy/python-for-android OpenCV recipe:
  https://github.com/kivy/python-for-android/tree/develop/pythonforandroid/recipes/opencv
- Briefcase camera limitation:
  https://github.com/beeware/briefcase/discussions/1921
- PySide6 Android deployment:
  https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-android-deploy.html

PDF, fonts, printing:

- WeasyPrint first steps (Pango/HarfBuzz):
  https://doc.courtbouillon.org/weasyprint/stable/first_steps.html
- HarfBuzz shaping:
  https://harfbuzz.github.io/what-does-harfbuzz-do.html
- Android custom printing:
  https://developer.android.com/training/printing/custom-docs
- Android `PrintedPdfDocument`:
  https://developer.android.com/reference/android/print/pdf/PrintedPdfDocument
- IPP print scaling:
  https://www.pwg.org/ipp/ippguide.html

Backend and sync:

- Supabase RLS:
  https://supabase.com/docs/guides/database/postgres/row-level-security
- Supabase pricing:
  https://supabase.com/pricing
- Supabase storage access control:
  https://supabase.com/docs/guides/storage/security/access-control
- PowerSync + Supabase:
  https://docs.powersync.com/integrations/supabase/guide
- Firebase offline persistence:
  https://firebase.google.com/docs/firestore/manage-data/enable-offline
- Firebase pricing:
  https://firebase.google.com/pricing

---

## 20. One-sentence recommendation

If the goal is a dependable Android product for real classrooms, build the
native Kotlin app, port the OMR engine against the Python reference, keep
Python for print-perfect PDFs and audits, store data locally first, and add
Supabase sync only after the scanner and paper workflow pass a real pilot.
