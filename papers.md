# Optical Mark Recognition (OMR) — Open-Access Paper & Dataset Review

**Compiled:** 28 October 2026  
**Scope:** recent (2019–2026) peer-reviewed work on reading bubble/answer sheets, with emphasis on camera-based, real-time and template-independent OMR.  
**Rule:** every paper listed below has a freely readable full text or at minimum a free publisher page with the complete abstract. Paywalled papers are excluded and listed separately at the end.

> **How to read the accuracy column:** accuracy is **not comparable across papers**. Some report *per-bubble* accuracy, some *per-question*, some *whole-sheet exact match*, and some *detection mAP*. A system with 99.9% bubble accuracy can still score only ~90% of sheets perfectly because 30 questions compound errors. Wherever possible the exact metric is named. Figures marked ⚠️ were taken from the publisher's PDF/index snippet, not read in full text.

> **How this list was built:** searched OpenAlex, Crossref, Semantic Scholar, PubMed Central, J-STAGE, MDPI, IEEE Access, Zenodo and Mendeley Data for 2019–2026 OMR/answer-sheet work, then opened each primary page to confirm title, open-access status, method summary and at least one quantitative result. This is a **curated set, not a claim of exhaustive coverage**: OMR is published across many small venues, several relevant papers are paywalled, and many report no numeric accuracy at all. The exclusions are listed in §7 so you can see what was left out and why.

---

## 1. Summary table

| # | Year | Title (short) | Venue | Method family | Reported accuracy | Public dataset? |
|---|------|---------------|-------|---------------|-------------------|-----------------|
| 1 | 2025 | Unsupervised OMR on Answer Sheets for Massive Printed MCQ Tests | J. Imaging (MDPI) | Classical CV: homography, adaptive threshold, morphology, pixel+grey classification | **99.95%** of answers; **96.15%** of exams graded without error; 1.04 s/sheet | ✅ Tamaulipas (Mendeley) |
| 2 | 2025 | Bayesian-Optimization-Based Auto Optical Mark Array Recognition | IEICE Trans. Inf. Syst. | Bayesian global optimisation + multi-layer feature energy for irregular marks | **97.07%** localization; **97.80%** recognition | ❌ Not released |
| 3 | 2024 | Webcam-Based Robust and Affordable OMR for Teachers | IJEEI | Preprocessing, segmentation, recognition; webcam | **100%** (self-reported; protocol not detailed) | ❌ Not released |
| 4 | 2024 | Enhanced Fault Tolerance Algorithm for OMR Using Smartphone Cameras | IEEE Access | Smartphone pipeline focused on illumination robustness | No single % in abstract; "comparable to scanners" | ❌ Not released |
| 5 | 2023 | Fault Tolerant OMR | Computers, Materials & Continua | Hough-transform skew correction + improved marker detection | Improvements per artefact class; no aggregate % in abstract | ❌ Not released |
| 6 | 2025 | OMR Techniques for Multiple-Choice Tests: A Comprehensive Review | IEEE Access | Survey of classical + deep methods | N/A (review) | N/A |
| 7 | 2026 | OMR-Exam: Lightweight Framework via YOLO26, EfficientNet, DCGAN | IEEE Access | Object detection + classifier + GAN augmentation | **88.98% ± 3.92%** mean sheet accuracy ⚠️ | ❌ Not released |
| 8 | 2026 | Federated OMR for Exam Paper Digitization and Assessment | Applied Sciences (MDPI) | Federated learning across institutions | **98.41%** question-level; **84.62%** sheet-level; 99.42% ROI (project-reported) | ❌ Not verified public |
| 9 | 2026 | SurveyNet: Unified Deep Learning for OCR + OMR Survey Digitization | J. Imaging (MDPI) | 5-stage OCR+OMR deep pipeline | **50–97%** depending on task ⚠️ | ❌ Private survey data |
| 10 | 2024 | Automated Paper-based MCQ Scoring using Fast Object Detection | IJACSA | YOLO-style marker detection | **96.5%** ⚠️ | ❌ Not released |
| 11 | 2024 | Automated Entrance Examination Checker Using OMR | J. Computer, Software & Program | Camera + microcontroller + LCD, 250-item sheets | No numeric accuracy reported | ❌ Not released |

Supporting geometry/rectification work is in [§4](#4-supporting-work-geometry-dewarping-and-fiducials), public datasets in [§5](#5-public-datasets-worth-using), and open-source baselines in [§6](#6-open-source-baselines-worth-reading).

---

## 2. Paper details

### 1. Unsupervised Optical Mark Recognition on Answer Sheets for Massive Printed Multiple-Choice Tests

- **Authors:** Yahir Hernández-Mier, Marco Aurelio Nuño-Maganda, Said Polanco-Martagón, Guadalupe Acosta-Villarreal, Rubén Posada-Gómez
- **Year / venue:** 2025, *Journal of Imaging* 11(9), 308
- **Open-access link:** [MDPI](https://www.mdpi.com/2313-433X/11/9/308) · [PubMed Central full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12470569/)
- **DOI:** [10.3390/jimaging11090308](https://doi.org/10.3390/jimaging11090308)
- **Methodology:** A deliberately non-neural pipeline for scanned sheets: orientation and size normalisation, rectangle/contour detection, homography-based perspective correction, adaptive thresholding, morphological cleanup, then mark classification by combining pixel-count and grey-level evidence. No training data required.
- **Reported accuracy:** **99.95%** of four-option answers classified correctly; **96.15%** of exams graded with no error at all; average **1.04 s per sheet**. Evaluated on **6,029 real exam sheets / 564,040 answers** from Tamaulipas, Mexico.
- **Dataset:** [Tamaulipas Multiple-Choice-Question Exam Image Dataset](https://data.mendeley.com/datasets/djmynjwjpy/2) — 5,721 scanned sheets, 535,020 labelled items, human-observed labels. (The paper's 6,029-sheet evaluation and the public 5,721-sheet release are related but not identical; check the record.)
- **Why it matters for us:** This is the strongest evidence that a **classical CV pipeline with proper homography + adaptive thresholding** can reach production accuracy on real, messy sheets — without a GPU or training set. It is scanner-based, not real-time camera, which is exactly the gap our project fills.
- **Limitations:** Desktop/scanned input; no phone-angle, glare or hand-held evaluation; template-specific layout.

### 2. Bayesian-Optimization-Based Auto Optical Mark Array Recognition for Flexible Paper Answer Sheet

- **Authors:** Chenbo Shi, Wenxin Sun, Jie Zhang, Junsheng Zhang, Chun Zhang, Changsheng Zhu
- **Year / venue:** 2025, *IEICE Transactions on Information and Systems* E108.D(7), 685–696
- **Open-access link:** [J-STAGE (FREE ACCESS)](https://www.jstage.jst.go.jp/article/transinf/E108.D/7/E108.D_2024IIP0010/_article/-char/en)
- **Methodology:** Models the mark array as a flexible structure and uses Bayesian global optimisation plus a multi-layer feature-energy model to jointly localise and segment marks that are irregularly shaped or non-uniformly spaced. Recognition is based on pixel occupancy inside the recovered marks.
- **Reported accuracy:** **97.07%** single-mark localisation accuracy and **97.80%** recognition accuracy on **31,940 mark instances**.
- **Dataset:** Not publicly released.
- **Why it matters for us:** Directly addresses the failure mode of fixed grid assumptions — skewed, shifted or non-uniform bubbles. Its "find the array, then read it" idea supports our lattice-fitting stage, even though we will use our own generator's exact geometry.
- **Limitations:** No public data/code; not real-time; evaluated on a proprietary set.

### 3. Webcam Based Robust and Affordable Optical Mark Recognition System for Teachers

- **Authors:** Effat Somaiya, Alifa Sun Mim, Mohammed Abdul Kader
- **Year / venue:** 2024, *Indonesian Journal of Electrical Engineering and Informatics (IJEEI)* 12(4), 870–882
- **Open-access link:** [IJEEI article page](https://section.iaesonline.com/index.php/IJEEI/article/view/5856)
- **DOI:** [10.52549/ijeei.v12i4.5856](https://doi.org/10.52549/ijeei.v12i4.5856)
- **Methodology:** Webcam capture followed by a pipeline of image enhancement, segmentation and mark recognition, designed to remove dependence on dedicated OMR hardware. Targets inconsistent handwriting and varying lighting.
- **Reported accuracy:** **100%** (stated in the abstract). The evaluation set size and protocol are not prominent; treat this as a demonstration until the full protocol is checked, not as a benchmark.
- **Dataset:** Not released.
- **Why it matters for us:** Confirms demand and feasibility of low-cost webcam OMR in a school context; also shows the literature often reports perfect accuracy on small self-collected sets, so our evaluation must be much stricter (multi-device, multi-condition, sheet-level).
- **Limitations:** No public data; no latency/FPS report; no angle stress test.

### 4. An Enhanced Fault Tolerance Algorithm for Optical Mark Recognition Using Smartphone Cameras

- **Authors:** Qamar Hafeez, Waqar Aslam, Romana Aziz, Ghadah Aldehim
- **Year / venue:** 2024, *IEEE Access*
- **Open-access link:** [IEEE Access (DOI)](https://doi.org/10.1109/ACCESS.2024.3451972)
- **DOI:** [10.1109/ACCESS.2024.3451972](https://doi.org/10.1109/ACCESS.2024.3451972)
- **Methodology:** An OMR algorithm designed specifically for uncontrolled smartphone capture. The abstract identifies illumination variation as the key failure source and proposes a fault-tolerant recognition scheme intended to match scanner-based results.
- **Reported accuracy:** No single aggregate accuracy is given in the abstract; the authors report results "comparable to those obtained from the usage of optical scanners". Read the full text for the per-condition tables.
- **Dataset:** Not released.
- **Why it matters for us:** It validates the core premise — phone cameras are the hard part, and illumination (not just geometry) is the dominant error source. Our pipeline therefore treats photometric normalisation and glare masking as first-class stages.
- **Limitations:** Android/iOS portability focus rather than a real-time video system; no public benchmark.

### 5. Fault Tolerant Optical Mark Recognition

- **Authors:** Qamar Hafeez, Waqar Aslam, M. Ikramullah Lali, Shafiq Ahmad, Mejdal Alqahtani, Muhammad Shafiq
- **Year / venue:** 2023, *Computers, Materials & Continua* 74(2), 3829–3847 (received 2021, issue 2022/2023)
- **Open-access link:** [TechScience full text](https://www.techscience.com/cmc/v74n2/50190/html)
- **DOI:** [10.32604/cmc.2023.026422](https://doi.org/10.32604/cmc.2023.026422)
- **Methodology:** Classifies OMR error types caused by printing/scanning artefacts, then proposes an improved Hough-transform-based skew-correction algorithm and better marker detection.
- **Reported accuracy:** Reports accuracy improvement over existing algorithms for each identified artefact class; the abstract does not give one aggregate percentage.
- **Dataset:** Not released.
- **Why it matters for us:** Useful taxonomy of real failure modes (skew, print offset, marker loss) and a reminder that **skew correction alone is not enough** — it cannot fix perspective, which is why we use a full homography plus lattice refinement.
- **Limitations:** Scanner-era assumptions; no camera/video evaluation.

### 6. Optical Mark Recognition Techniques for Multiple-Choice Tests: A Comprehensive Review

- **Authors:** Mohammed Ali, Suhel Hammoud, Alida Esper
- **Year / venue:** 2025, *IEEE Access* 13, 156407–156423
- **Open-access link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11129039)
- **Methodology:** Systematic review of OMR approaches — classical image processing, template matching, object detection and deep learning — with attention to preprocessing, localisation and mark classification.
- **Reported accuracy:** Not applicable (review).
- **Dataset:** N/A.
- **Why it matters for us:** Best single entry point to the field and a source of comparison tables; use it to justify design choices and to find additional baselines.
- **Limitations:** Review, not a new system; conclusions are only as good as the underlying papers' protocols.

### 7. OMR-Exam: An Efficient Lightweight Framework for Automated Exam Scoring via YOLO26, EfficientNet, and DCGAN-Enhanced Recognition

- **Authors:** Nguyen Thi Hong Ngan, Le Duc Thuan
- **Year / venue:** 2026, *IEEE Access* 14, 154198
- **Open-access link:** [IEEE Access PDF](https://ieeexplore.ieee.org/iel8/6287639/6514899/11720321.pdf)
- **DOI:** [10.1109/ACCESS.2026.3740415](https://doi.org/10.1109/ACCESS.2026.3740415)
- **Methodology:** End-to-end learned pipeline: YOLO26 localises bubbles/marks without a per-bubble template, EfficientNet classifies them, and a DCGAN augments scarce training data.
- **Reported accuracy:** **88.98% ± 3.92% mean sheet accuracy** ⚠️ (figure read from the publisher PDF text). This is sheet-level, the strictest metric — notably lower than the per-bubble numbers in classical papers, illustrating error compounding.
- **Dataset:** Not released.
- **Why it matters for us:** Strong recent evidence that **template-free detection alone is not yet good enough for high-stakes grading**. With a known generated layout we can beat this by using exact geometry, and reserve learned models for ambiguous bubbles.
- **Limitations:** Heavy training requirement; no public data; sheet-level accuracy below what a school can tolerate without review.

### 8. Federated OMR for Automated Examination Paper Digitization and Assessment

- **Authors:** Duc Thuan Le, Huy Hoang Nguyen, et al.
- **Year / venue:** 2026, *Applied Sciences* 16(17), 8859
- **Open-access link:** [MDPI](https://www.mdpi.com/2076-3417/16/17/8859)
- **DOI:** [10.3390/app16178859](https://doi.org/10.3390/app16178859)
- **Methodology:** Federated learning across multiple institutions/clients so exam data never leaves each site; combines detection/classification for OMR and automated assessment.
- **Reported accuracy:** The authors' public code repository reports five-client FedAvg results of **98.41% question-level accuracy**, **84.62% sheet-level accuracy**, and **99.42% ROI classification accuracy**. Treat these as project-reported until you read the paper's tables.
- **Dataset:** Five heterogeneous dataset groups described; public download links were placeholders when checked, so no verified dataset link.
- **Why it matters for us:** The **98.41% → 84.62% gap** is the single most important number in this review: per-question accuracy hides how often a whole sheet is wrong. Our acceptance metric must be sheet-level exact match, plus a review queue for low-confidence questions.
- **Limitations:** Federated infrastructure is out of scope for a single-school desktop tool; results not independently reproduced here.

### 9. SurveyNet: A Unified Deep Learning Framework for OCR and OMR-Based Survey Digitization

- **Authors:** Rubi Quiñones et al.
- **Year / venue:** 2026, *Journal of Imaging* 12(4), 175
- **Open-access link:** [MDPI](https://www.mdpi.com/2313-433X/12/4/175)
- **Methodology:** A five-stage deep pipeline that combines OCR of handwritten/text responses with OMR of marked options in one model, aimed at complex survey forms rather than clean exam sheets.
- **Reported accuracy:** **50%–97% classification accuracy across tasks** ⚠️, with strong performance claimed even on small/imbalanced datasets.
- **Dataset:** Private survey data; no public release identified.
- **Why it matters for us:** Shows the ceiling and variability of learned end-to-end form digitisation. For our fixed MCQ layout, a deterministic geometry-first approach is simpler, faster and easier to audit.
- **Limitations:** Survey forms, not timed exams; wide accuracy range; not real-time.

### 10. Automated Paper-based Multiple Choice Scoring Framework using Fast Object Detection Algorithm

- **Authors:** Pham Doan Tinh, Ta Quang Minh
- **Year / venue:** 2024, *International Journal of Advanced Computer Science and Applications (IJACSA)* 15(1)
- **Open-access link:** [The SAI paper page](https://thesai.org/Publications/ViewPaper?Volume=15&Issue=1&Code=IJACSA&SerialNo=115)
- **DOI:** [10.14569/IJACSA.2024.01501115](https://doi.org/10.14569/IJACSA.2024.01501115)
- **Methodology:** Uses a fast object-detection model to locate answer markers/bubbles, then scores against the key, avoiding hand-crafted per-template rules.
- **Reported accuracy:** **96.5%** ⚠️ (from the paper's abstract/index snippet; metric definition — detection vs sheet — should be confirmed in the PDF).
- **Dataset:** Not released.
- **Why it matters for us:** A middle ground between classical CV and heavy deep pipelines; useful if we later add an unknown-layout fallback.
- **Limitations:** No public data, no angle/lighting stress evaluation in the abstract.

### 11. An Automated Entrance Examination Checker Using Optical Mark Recognition

- **Authors:** Sheryl May D. Lainez, Leo Joshua Q. Gresos, Via Karen A. Maganggo
- **Year / venue:** 2024, *Journal of Computer, Software, and Program* 1(1), 8–13
- **Open-access link:** [Journal article page (CC BY 4.0)](https://journals.stecab.com/jcsp/article/view/43)
- **DOI:** [10.69739/jcsp.v1i1.43](https://doi.org/10.69739/jcsp.v1i1.43)
- **Methodology:** Camera-based checker with a microcontroller and LCD for 250-item multiple-choice entrance sheets; stores results and displays scores. Developed with Rapid Application Development.
- **Reported accuracy:** No numerical accuracy reported; validation is qualitative/beta testing with admission staff and students.
- **Dataset:** Not released.
- **Why it matters for us:** Evidence of deployment constraints in a real admissions office (throughput, operator workflow, display of results) even though it is not a strong accuracy reference.
- **Limitations:** No quantitative benchmark; no public code/data.

---

## 3. What the evidence says (design implications)

1. **Geometry first, learning second.** The best real-world result in this set (J. Imaging 2025, 99.95% answers / 96.15% perfect sheets) uses homography + adaptive thresholding, no neural network. Our sheet layout is known exactly, so exact geometry should be the backbone.
2. **Sheet-level accuracy is the metric that matters.** Federated OMR's 98.41% question vs 84.62% sheet accuracy and OMR-Exam's 88.98% sheet accuracy show that per-bubble accuracy flatters a system. A 30-question sheet needs ~99.8% per-question accuracy to reach ~95% perfect sheets.
3. **Camera capture is the bottleneck, not the marks.** The smartphone papers identify illumination, skew and uncontrolled capture as the dominant errors. Hence: quality gating, photometric normalisation, glare masking, multi-frame evidence.
4. **Faint, erased and crossed-out marks are a separate class.** The MC Answer Boxes dataset explicitly labels *confirmed / crossed-out / empty*; Wild-OMR includes scripted erasures and faint marks. A binary "filled/empty" threshold will fail these; we need a third "ambiguous → review" state.
5. **Template-free deep models are improving but not yet sufficient alone.** YOLO-based results (88.98% sheet-level, 96.5% reported overall) are promising but below what a school can accept without human review. A hybrid — exact template + classical evidence + tiny learned arbiter — is the pragmatic 2026 choice.
6. **Report latency and review rate, not just accuracy.** None of the camera papers report a full latency/throughput budget under real conditions. Our plan must.

---

## 4. Supporting work: geometry, dewarping and fiducials

These are not OMR papers, but they define the geometry toolkit for "any angle". Metrics are for document rectification, not mark reading — do not compare them to OMR accuracy.

| Work | Year | Method | Reported result | Link |
|------|------|--------|-----------------|------|
| **DocTr++ / Deep Unrestricted Document Image Rectification** | 2023 | Hierarchical transformer predicts per-patch warping flows; handles full, partial or missing document boundaries | DocUNet: MS-SSIM **0.51**, LD **7.52**, CER **16.95%**, **11.29 FPS** | [arXiv](https://arxiv.org/abs/2304.08796) · [code](https://github.com/fh2019ustc/DocTr-Plus) |
| **DocScanner** | 2021 | Localises document, then recurrently refines a dense warping flow with line/circle consistency | DocUNet: MS-SSIM **0.5134/0.5178**, LD **7.62/7.45**, **10.03/9.52 FPS** (1080p, RTX 2080 Ti) | [arXiv](https://arxiv.org/abs/2110.14968) |
| **UVDoc** | 2023 | Predicts a 3D document grid and a 2D inverse unwarping grid; 20k pseudo-photorealistic training images | UVDoc benchmark: MS-SSIM **0.785**, AD **0.119** | [arXiv](https://arxiv.org/abs/2302.02887) · [code](https://github.com/tanguymagne/UVDoc) |
| **DocUNet benchmark** | 2018 | Real photographed documents with flatbed references; MS-SSIM + Local Distortion | Baseline for the above | [Project page](https://www3.cs.stonybrook.edu/~cvl/docunet.html) |
| **Document corner localisation dataset** | 2020 | 1,111 smartphone document images for corner algorithms | Dataset page reports failures remain; numeric corner error not verified | [Mendeley](https://data.mendeley.com/datasets/x3nm4cxr83/3) |

**Practical reading:** a single homography is correct for a **flat** sheet viewed obliquely. It cannot undo folds, curvature or wrinkles. Learned dewarpers are accurate but run at ~10 FPS on a desktop GPU — too heavy for a CPU-only real-time tool and unnecessary when the sheet carries a printed, regular bubble lattice. Our plan uses **lattice-constrained local refinement** instead, with a learned dewarper as an optional offline fallback for badly curved sheets.

**Fiducials (ArUco/AprilTag):** no verified OMR paper reports accuracy/FPS for fiducial-based exam sheets. This is an engineering opportunity, not a literature-backed claim: four printed markers give known control points even when the paper edge is cropped, and a QR code gives sheet identity. Both are trivial to add to a generator we control.

---

## 5. Public datasets worth using

| Dataset | Size | Labels | Conditions / value | License | Link |
|---------|------|--------|--------------------|---------|------|
| **Wild-OMR** (2026) | 769 released captures of 30 sheets; 40 five-option questions each (200 bubbles) | Exact per-bubble ground truth from a pre-assigned key; clean, erased and faint-mark subsets | 5 phones × 5 conditions (flat, dim, glare, tilted ~30°, hand-held/curved) + flatbed reference; QR-based sheet identity; includes baselines and evaluation harness | **CC BY 4.0** data, MIT code | [Zenodo](https://zenodo.org/records/21710005) |
| **Tamaulipas MCQ Exam Image Dataset** (2026, v2) | 5,721 scanned sheets; 535,020 items (90 or 100 items/sheet) | Human-observed answer per item (text files) | Real 2024 exams from 42 high schools; varied marking styles, noise, scan artefacts; anonymised | Check record before reuse | [Mendeley Data](https://data.mendeley.com/datasets/djmynjwjpy/2) |
| **MC Answer Boxes Dataset** (2019) | 735 answer sheets; 33,540 answer-box ROIs; 6 exams/templates | 3 classes: **confirmed, crossed-out, empty**; model answers, metadata, grades | Real scanned exams from Assiut University; different templates and languages; ideal for the "crossed-out" class | Free for reasonable academic fair use | [Dataset page](https://sites.google.com/view/mcq-dataset) |
| **Kaggle OMR Image Dataset** | 145 labelled sheet images | Labels described; schema not verified | Small; useful for a quick smoke test, not a benchmark | Not stated on page | [Kaggle](https://www.kaggle.com/datasets/thejasan/omr-optical-mark-recognition-image-dataset) |
| **Bubble Answer Sheet (Roboflow, v2)** | Listing reports 5,001 images | Object-detection boxes; schema not verified (page returns 403 to scrapers) | Sheet/bubble detection | Listing reports CC BY 4.0 | [Roboflow Universe](https://universe.roboflow.com/bayan-ewozi/bubble-answer-sheet-cy55z-p4cro/dataset/2) |
| **OMR Scanner (Roboflow, v3)** | Listing reports 360 images | Object detection | Small OMR detection set | Listing reports CC BY 4.0 | [Roboflow Universe](https://universe.roboflow.com/sust-9qaee/omr-scanner/dataset/3) |
| **DocUNet benchmark** | Real phone photos + flatbed references | Pixel/flow supervision for dewarping | Curved/folded document rectification | Research use (see project page) | [Project page](https://www3.cs.stonybrook.edu/~cvl/docunet.html) |
| **UVDoc dataset** | 20k synthetic + real evaluation sets | 3D grid / 2D flow | Curved document unwarping | See repository | [GitHub](https://github.com/tanguymagne/UVDoc) |

**Recommended evaluation stack for this project:**
1. **Wild-OMR** for camera robustness — it is the only public benchmark that isolates tilt, glare, dim light and hand-held curvature with exact ground truth. Its 30 sheets are few, so use leave-one-sheet-out and report per-condition results.
2. **Tamaulipas** for flat-scan reading accuracy at scale (5,721 sheets, 535k marks).
3. **MC Answer Boxes** for the crossed-out/faint/empty classification sub-problem.
4. **Your own captures** (the actual pink Bengali sheet, several phones, pens and lighting setups) as the final acceptance set — no public dataset matches your layout.

---

## 6. Open-source baselines worth reading

| Project | What it is | Reported result | Link |
|---------|------------|-----------------|------|
| **amazingOMR** | Template-based OMR, scoring, confidence and rescan flags; synthetic ground-truthed suite | 100% raw question accuracy on its own 178-sheet/10,720-question suite; 100% auto-graded at 96% coverage in an extreme test (project-reported) | [GitHub](https://github.com/dwin-gharibi/amazingOMR) |
| **OMRFlow** | Template design, scan reading, conflict review, attendance reconciliation, report export | Alpha; thousands of automated tests, no real-cohort benchmark | [GitHub](https://github.com/sajidbuet/OMRflow) |
| **LazyOMR** | Phone-camera/PWA, offline MCQ scoring | No numeric accuracy; documents lighting/crease limitations | [GitHub](https://github.com/Dilshan-H/LazyOMR) |
| **bubblesheet-ai** | YOLO bubble detection + key comparison + grade export | mAP@50 **0.536**, mAP@50–95 **0.283**, precision 0.47, recall 0.54 (small dataset) | [GitHub](https://github.com/hp-8/bubblesheet-ai) |
| **OMRNet** | MobileNetV2 classifier for Marked / Empty / Crossed-out; quantised to ~3 MB | SOTA claimed on MC Answer Boxes; no single number in README | [GitHub](https://github.com/sa-y-an/OMRNet) |
| **BubbleGrader** | OpenCV sheet detection, perspective correction, bubble detection, grading | No accuracy reported | [GitHub](https://github.com/ShreenidhiBodas/OMR) |

---

## 7. Papers found but excluded (paywalled or unverifiable)

These are relevant but did **not** meet the open-access/verified standard, so they are not counted above:

- **OMRNet: A lightweight deep learning model for optical mark recognition** — Mondal et al., *Multimedia Tools and Applications* (2023), DOI [10.1007/s11042-023-15408-8](https://doi.org/10.1007/s11042-023-15408-8). The paper is paywalled; the [GitHub repo](https://github.com/sa-y-an/OMRNet) and the [MC Answer Boxes dataset](https://sites.google.com/view/mcq-dataset) are open.
- **Personal mark density-based high-performance OMR using K-means clustering** — Sancar et al., *Multimedia Tools and Applications* (2024). Paywalled; a related open dataset/paper may exist, but the full text was not verified.
- **Detection of risk levels in optical mark recognition exams** — Sancar et al., *Scientific Reports* (2025). Open access in principle, but the publisher page blocked automated reading; accuracy not verified here.
- **A YOLO-based software framework for automated paper-based MCQ scoring** — *Software Impacts* (2026). Open access in principle, but the page blocked automated reading; not included to avoid unverified figures.
- **Real-time Bubble Sheet Detection and Evaluation Using Object Recognition** — Razzaq et al., *Asian Bulletin of Big Data Management* (2025). Journal page is bot-protected; accuracy claim could not be read from the primary source.
- **An Image Processing Oriented Optical Mark Reader** — Fathima & Bhat (2018). Older; not recent.

> If you want any of these added, open the PDF in a browser once and share the exact accuracy table; I can fold it into this review with the correct metric.

---

## 8. Direct implications for `realtime-plan.md`

| Evidence | Design decision |
|----------|-----------------|
| J. Imaging 2025: homography + adaptive threshold → 99.95% answers | Use exact template geometry + per-sheet photometric calibration as the backbone |
| Federated OMR: 98.41% question vs 84.62% sheet | Grade at sheet level; surface low-confidence questions for review |
| OMR-Exam: 88.98% sheet with YOLO26 | Do not rely on generic detection; use the generator's known bubble lattice |
| Wild-OMR: tilt, glare, dim, hand-held conditions | Add quality gating, glare masks, multi-frame fusion and per-condition evaluation |
| MC Answer Boxes: crossed-out/empty classes | Three-state reading (marked / empty / ambiguous) instead of a binary threshold |
| IEEE Access 2024: illumination is the key smartphone failure | Local background normalisation and chroma separation of pink print vs ink |
| DocTr++/UVDoc: ~10 FPS on GPU | Avoid heavy dewarping in the real-time loop; use lattice refinement, optional offline pass |

---

*Compiled from publisher pages, PubMed Central, J-STAGE, Zenodo and Mendeley records on 28 October 2026. Links were opened where the publisher allowed it; blocked pages are flagged. Accuracy values are reproduced as reported by the authors, not independently re-measured.*
