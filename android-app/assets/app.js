/*
 * OMR Checker app controller.
 *
 * Subjects and exams live in the phone database.  There is one printed sheet
 * design (print/v3.pdf, 60 slots on A4).  One exam = one answer key and one
 * mark per question; the exam's question count says how many of the printed
 * slots it uses.  The live camera fuses several frames per sheet so a moving
 * hand cannot change an answer, then stops after one result.
 */
(function () {
  'use strict';

  const OPTIONS = [
    { label: 'A', digit: '1' },
    { label: 'B', digit: '2' },
    { label: 'C', digit: '3' },
    { label: 'D', digit: '4' },
  ];

  const sheets = window.OMR_SHEETS;
  const factory = window.OMREngineFactory;
  const store = window.OMRStore;
  const design = window.OMR_SHEETDESIGN;

  const state = {
    subjects: [],
    exams: [],
    subjectId: null,
    examId: null,
    exam: null,
    template: null,
    engine: null,
    questionCount: 0,
    key: [],
    marks: [],
    dirty: false,
  };

  let editingExamId = 0;
  const pendingDelete = { subject: 0, exam: 0 };
  let pendingTimer = 0;

  const $ = (id) => document.getElementById(id);

  const headerSub = $('headerSub');
  const footerNote = $('footerNote');

  const subjectName = $('subjectName');
  const addSubjectBtn = $('addSubjectBtn');
  const subjectMessage = $('subjectMessage');
  const subjectList = $('subjectList');
  const subjectDetail = $('subjectDetail');
  const detailName = $('detailName');
  const renameSubjectBtn = $('renameSubjectBtn');
  const deleteSubjectBtn = $('deleteSubjectBtn');
  const subjectRename = $('subjectRename');
  const subjectRenameInput = $('subjectRenameInput');
  const subjectRenameSave = $('subjectRenameSave');
  const subjectRenameCancel = $('subjectRenameCancel');
  const examName = $('examName');
  const examCount = $('examCount');
  const addExamBtn = $('addExamBtn');
  const examMessage = $('examMessage');
  const examList = $('examList');

  const keyTitle = $('keyTitle');
  const keySubtitle = $('keySubtitle');
  const bulkInput = $('bulkInput');
  const bulkApply = $('bulkApply');
  const bulkMessage = $('bulkMessage');
  const keyGrid = $('keyGrid');
  const keyCountEl = $('keyCount');
  const keyProgressEl = $('keyProgress');
  const clearKeyEl = $('clearKey');
  const saveKeyEl = $('saveKey');
  const saveMessage = $('saveMessage');

  const scanKeyState = $('scanKeyState');
  const takePhoto = $('takePhoto');
  const choosePhoto = $('choosePhoto');
  const cameraInput = $('cameraInput');
  const galleryInput = $('galleryInput');
  const processing = $('processing');
  const processingTitle = $('processingTitle');
  const processingText = $('processingText');
  const errorBox = $('errorBox');
  const errorText = $('errorText');
  const retryButton = $('retryButton');
  const resultEl = $('result');
  const resultTotal = $('resultTotal');
  const resultMax = $('resultMax');
  const resultVerdict = $('resultVerdict');
  const statCorrect = $('statCorrect');
  const statWrong = $('statWrong');
  const statBlank = $('statBlank');
  const statMulti = $('statMulti');
  const statReview = $('statReview');
  const previewCanvas = $('previewCanvas');
  const photoCanvas = $('photoCanvas');
  const breakdownEl = $('breakdown');
  const scanModeEl = $('scanMode');
  const scanAgain = $('scanAgain');
  const backToKey = $('backToKey');

  const startLive = $('startLive');
  const stopLive = $('stopLive');
  const nextLive = $('nextLive');
  const toggleVoice = $('toggleVoice');
  const liveKeyState = $('liveKeyState');
  const arStage = $('arStage');
  const arVideo = $('arVideo');
  const arCanvas = $('arCanvas');
  const arStatus = $('arStatus');
  const arHudMark = $('arHudMark');
  const arHudTotal = $('arHudTotal');
  const arHudMax = $('arHudMax');
  const arPlaceholder = $('arPlaceholder');
  const liveError = $('liveError');
  const liveStats = $('liveStats');
  const liveCorrect = $('liveCorrect');
  const liveWrong = $('liveWrong');
  const liveBlank = $('liveBlank');
  const liveMulti = $('liveMulti');
  const liveReview = $('liveReview');

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const mmToPx = (mm) => mm / 25.4 * (state.engine ? state.engine.DPI : 150);

  function setMessage(element, text, ok) {
    if (!element) return;
    element.textContent = text || '';
    element.classList.toggle('ok', Boolean(ok));
  }

  /* ------------------------------------------------------------- navigation */

  function activateTab(name) {
    document.querySelectorAll('.tab').forEach(tab => {
      tab.classList.toggle('is-active', tab.dataset.tab === name);
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
      panel.classList.toggle('is-hidden', panel.id !== `tab-${name}`);
    });
    const guideLink = document.querySelector('.topbar-link');
    if (guideLink) guideLink.classList.toggle('is-active', name === 'guide');
    if (name !== 'live' && live.running) stopLiveCamera();
    if (name === 'scan' || name === 'live') updateKeyUI();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  /* ------------------------------------------------------- subjects / exams */

  function armDelete(kind, id, button) {
    if (pendingDelete[kind] === id) {
      pendingDelete[kind] = 0;
      return true;
    }
    pendingDelete[kind] = id;
    if (button) {
      button.textContent = 'Confirm delete';
      button.classList.add('is-armed');
    }
    clearTimeout(pendingTimer);
    pendingTimer = window.setTimeout(() => {
      pendingDelete[kind] = 0;
      renderSubjectList();
      renderExamList();
    }, 4000);
    return false;
  }

  function renderSubjectList() {
    subjectList.innerHTML = '';
    if (!state.subjects.length) {
      subjectList.innerHTML = '<p class="hint">No subjects yet. Add the first one above.</p>';
      return;
    }
    const frag = document.createDocumentFragment();
    state.subjects.forEach(subject => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'subject-card' + (subject.id === state.subjectId ? ' is-active' : '');
      card.dataset.subject = String(subject.id);

      const text = document.createElement('div');
      text.className = 'spacer';
      const name = document.createElement('strong');
      name.textContent = subject.name;
      const meta = document.createElement('span');
      meta.textContent = subject.examCount === 1 ? '1 exam' : `${subject.examCount} exams`;
      text.appendChild(name);
      text.appendChild(meta);

      card.appendChild(text);
      card.addEventListener('click', () => selectSubject(subject.id));
      frag.appendChild(card);
    });
    subjectList.appendChild(frag);
  }

  function refreshSubjects(selectId) {
    try {
      state.subjects = store.listSubjects() || [];
    } catch (error) {
      state.subjects = [];
      setMessage(subjectMessage, error.message, false);
    }
    renderSubjectList();

    const wanted = selectId || state.subjectId;
    if (wanted && state.subjects.some(s => s.id === wanted)) {
      selectSubject(wanted);
    } else if (!state.subjects.length) {
      state.subjectId = null;
      subjectDetail.classList.add('is-hidden');
    }
  }

  function selectSubject(id) {
    state.subjectId = id;
    const subject = state.subjects.find(s => s.id === id);
    if (!subject) {
      subjectDetail.classList.add('is-hidden');
      return;
    }
    detailName.textContent = subject.name;
    subjectRename.classList.add('is-hidden');
    subjectRenameInput.value = subject.name;
    subjectDetail.classList.remove('is-hidden');
    renderSubjectList();
    refreshExams();
  }

  function refreshExams() {
    if (!state.subjectId) {
      examList.innerHTML = '';
      return;
    }
    try {
      state.exams = store.listExams(state.subjectId) || [];
    } catch (error) {
      state.exams = [];
      setMessage(examMessage, error.message, false);
    }
    renderExamList();
    if (state.examId && !state.exams.some(e => e.id === state.examId)) {
      state.examId = null;
      state.exam = null;
      loadExamKey();
      buildKeyGrid();
      updateKeyUI();
    }
  }

  function renderExamList() {
    examList.innerHTML = '';
    if (!state.exams.length) {
      examList.innerHTML = '<p class="hint">No exams yet. Add one above and pick its question count.</p>';
      return;
    }

    state.exams.forEach(exam => {
      const card = document.createElement('div');
      card.className = 'exam-card' + (exam.id === state.examId ? ' is-active' : '');

      if (editingExamId === exam.id) {
        const form = document.createElement('div');
        form.className = 'inline-form';
        const input = document.createElement('input');
        input.type = 'text';
        input.value = exam.name;
        input.autocomplete = 'off';
        const save = document.createElement('button');
        save.type = 'button';
        save.className = 'btn small primary';
        save.textContent = 'Save';
        const cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.className = 'btn small secondary';
        cancel.textContent = 'Cancel';

        save.addEventListener('click', () => {
          const name = input.value.trim();
          if (!name) { setMessage(examMessage, 'Enter a name for the exam.', false); return; }
          try {
            store.updateExam(exam.id, name, null, null);
            editingExamId = 0;
            setMessage(examMessage, 'Exam renamed.', true);
            refreshExams();
            if (state.examId === exam.id && state.exam) {
              state.exam.name = name;
              updateKeyUI();
              updateHeader();
            }
          } catch (error) {
            setMessage(examMessage, error.message, false);
          }
        });
        cancel.addEventListener('click', () => { editingExamId = 0; renderExamList(); });

        form.appendChild(input);
        form.appendChild(save);
        form.appendChild(cancel);
        card.appendChild(form);
        examList.appendChild(card);
        return;
      }

      const title = document.createElement('strong');
      title.textContent = exam.name;
      const meta = document.createElement('span');
      meta.className = 'exam-meta';
      meta.textContent = `${exam.questionCount} questions · ${exam.keyCount} answers set`;

      const actions = document.createElement('div');
      actions.className = 'exam-row-actions';

      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'btn small primary';
      open.textContent = exam.id === state.examId ? 'Editing' : 'Open key';
      open.addEventListener('click', () => openExam(exam.id));

      const rename = document.createElement('button');
      rename.type = 'button';
      rename.className = 'link-button';
      rename.textContent = 'Rename';
      rename.addEventListener('click', () => { editingExamId = exam.id; renderExamList(); });

      const print = document.createElement('button');
      print.type = 'button';
      print.className = 'link-button';
      print.textContent = 'Print sheet';
      print.addEventListener('click', () => printBlankSheet(exam));

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'link-button danger' +
        (pendingDelete.exam === exam.id ? ' is-armed' : '');
      remove.textContent = pendingDelete.exam === exam.id ? 'Confirm delete' : 'Delete';
      remove.addEventListener('click', () => {
        if (!armDelete('exam', exam.id, remove)) return;
        try {
          store.deleteExam(exam.id);
          if (state.examId === exam.id) {
            state.examId = null;
            state.exam = null;
            loadExamKey();
            buildKeyGrid();
            updateKeyUI();
          }
          setMessage(examMessage, 'Exam deleted.', true);
          refreshExams();
        } catch (error) {
          setMessage(examMessage, error.message, false);
        }
      });

      actions.appendChild(open);
      actions.appendChild(rename);
      actions.appendChild(print);
      actions.appendChild(remove);

      card.appendChild(title);
      card.appendChild(meta);
      card.appendChild(actions);
      examList.appendChild(card);
    });
  }

  function printBlankSheet(exam) {
    const template = sheets.master;
    if (!template) {
      setMessage(examMessage, 'No sheet design for that question count.', false);
      return;
    }
    const started = design.printSheet(template);
    setMessage(examMessage, started
      ? `Sending the ${exam.questionCount}-question sheet to the printer. Choose “Save as PDF” or a printer, and print at 100% (actual size).`
      : 'Printing is not available in this browser.', started);
  }

  /* -------------------------------------------------------------- key editor */

  function countSet() {
    return state.key.reduce((n, value) => n + (Number.isInteger(value) ? 1 : 0), 0);
  }

  function keyReady() {
    return Boolean(state.examId) && countSet() > 0;
  }

  function totalMarks() {
    let sum = 0;
    for (let i = 0; i < state.questionCount; i++) {
      if (Number.isInteger(state.key[i])) {
        const mark = Number(state.marks[i]);
        sum += Number.isFinite(mark) && mark >= 0 ? mark : 1;
      }
    }
    return Math.round(sum * 100) / 100;
  }

  function buildKeyGrid() {
    keyGrid.innerHTML = '';
    if (!state.examId) {
      keyGrid.innerHTML = '<p class="hint">Open an exam on the Subjects tab to edit its key.</p>';
      return;
    }
    const frag = document.createDocumentFragment();
    for (let q = 1; q <= state.questionCount; q++) {
      const row = document.createElement('div');
      row.className = 'q-row';
      row.dataset.q = String(q);

      const number = document.createElement('div');
      number.className = 'q-num';
      number.textContent = String(q).padStart(2, '0');

      const options = document.createElement('div');
      options.className = 'options';
      OPTIONS.forEach((opt, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'opt';
        button.dataset.q = String(q);
        button.dataset.opt = String(index);
        button.setAttribute('aria-label', `Question ${q}, option ${opt.label}`);
        button.innerHTML = `<span class="bn">${opt.label}</span>`;
        options.appendChild(button);
      });

      const marks = document.createElement('input');
      marks.type = 'number';
      marks.className = 'mark-input';
      marks.min = '0';
      marks.step = '0.5';
      marks.dataset.q = String(q);
      marks.value = String(state.marks[q - 1] !== undefined ? state.marks[q - 1] : 1);
      marks.setAttribute('aria-label', `Marks for question ${q}`);

      row.appendChild(number);
      row.appendChild(options);
      row.appendChild(marks);
      frag.appendChild(row);
    }
    keyGrid.appendChild(frag);
  }

  function updateKeyUI() {
    const count = countSet();
    const ready = keyReady();

    keyCountEl.textContent = state.examId ? `${count} / ${state.questionCount}` : '0 / 0';
    keyProgressEl.style.width = state.questionCount
      ? `${(count / state.questionCount) * 100}%`
      : '0%';
    clearKeyEl.disabled = count === 0;
    saveKeyEl.disabled = !state.examId || !state.dirty;

    keyGrid.querySelectorAll('.q-row').forEach(row => {
      const q = Number(row.dataset.q) - 1;
      row.classList.toggle('is-set', Number.isInteger(state.key[q]));
      row.querySelectorAll('.opt').forEach(button => {
        const opt = Number(button.dataset.opt);
        const selected = state.key[q] === opt;
        button.classList.toggle('is-selected', selected);
        button.setAttribute('aria-pressed', selected ? 'true' : 'false');
      });
    });

    takePhoto.disabled = !ready;
    choosePhoto.disabled = !ready;
    if (!live.running) startLive.disabled = !ready;

    if (!state.examId) {
      keyTitle.textContent = 'Answer key';
      keySubtitle.textContent = 'Choose a subject and an exam on the Subjects tab first.';
      scanKeyState.textContent = 'Set an answer key first.';
      liveKeyState.textContent = 'Set an answer key first.';
      arHudMax.textContent = '/0';
      return;
    }

    const subject = state.subjects.find(s => s.id === state.subjectId);
    keyTitle.textContent = state.exam.name;
    keySubtitle.textContent = `${subject ? subject.name + ' · ' : ''}${state.questionCount} questions · `
      + `${totalMarks()} total marks · saved on this phone`;

    scanKeyState.textContent = count === state.questionCount
      ? `Key ready: ${state.questionCount} answers, ${totalMarks()} marks.`
      : `${count} of ${state.questionCount} answers set — unanswered questions show “no key”.`;

    liveKeyState.textContent = count === 0
      ? 'Set the answer key first.'
      : `Key ready: ${count} answers · hold a ${state.questionCount}-question sheet.`;

    arHudMax.textContent = `/${totalMarks()}`;
  }

  function setKey(qIndex, option) {
    if (!state.examId) return;
    state.key[qIndex] = state.key[qIndex] === option ? null : option;
    state.dirty = true;
    setMessage(saveMessage, '', false);
    updateKeyUI();
  }

  function setMark(qIndex, value) {
    const mark = Number(value);
    state.marks[qIndex] = Number.isFinite(mark) && mark >= 0 ? mark : 1;
    state.dirty = true;
    setMessage(saveMessage, '', false);
    updateKeyUI();
  }

  function parseBulk(text) {
    const raw = String(text || '').trim();
    if (!raw) return null;
    const count = state.questionCount;
    if (new RegExp(`^[1-4]{${count}}$`).test(raw)) {
      return raw.split('').map(v => Number(v) - 1);
    }
    const letters = raw.toLowerCase();
    if (new RegExp(`^[a-d]{${count}}$`).test(letters)) {
      return letters.split('').map(v => v.charCodeAt(0) - 97);
    }
    const tokens = raw.split(/[\s,;،]+/).filter(Boolean);
    if (tokens.length !== count) return null;
    const map = {
      '1': 0, '2': 1, '3': 2, '4': 3,
      '১': 0, '২': 1, '৩': 2, '৪': 3,
      'a': 0, 'b': 1, 'c': 2, 'd': 3,
      'ক': 0, 'খ': 1, 'গ': 2, 'ঘ': 3,
    };
    const out = tokens.map(token => map[String(token).toLowerCase()]);
    return out.some(v => v === undefined) ? null : out;
  }

  function loadExamKey() {
    const count = state.questionCount || 0;
    const answers = new Array(count).fill(null);
    const marks = new Array(count).fill(1);
    if (state.examId) {
      try {
        const stored = store.loadKey(state.examId);
        if (stored) {
          for (let i = 0; i < count; i++) {
            if (Number.isInteger(stored.answers[i])) answers[i] = stored.answers[i];
            const mark = Number(stored.marks[i]);
            if (Number.isFinite(mark) && mark >= 0) marks[i] = mark;
          }
        }
      } catch (error) {
        setMessage(saveMessage, error.message, false);
      }
    }
    state.key = answers;
    state.marks = marks;
    state.dirty = false;
  }

  function updateHeader() {
    if (state.exam) {
      const subject = state.subjects.find(s => s.id === state.subjectId);
      headerSub.textContent = `${subject ? subject.name + ' · ' : ''}${state.exam.name} · `
        + `${state.questionCount} questions`;
    } else {
      headerSub.textContent = 'Saved on this phone · offline';
    }
    footerNote.textContent = store.isPersistent()
      ? 'Subjects, keys and marks are stored in the phone database'
      : 'Browser preview: subjects and keys are stored in local storage';
  }

  function openExam(examId) {
    const exam = state.exams.find(e => e.id === examId);
    if (!exam) return;
    // One printed design: every exam reads the same 60-slot sheet.  The
    // exam's own question count decides how many slots are graded.
    const template = sheets.master;

    state.examId = exam.id;
    state.exam = exam;
    state.template = template;
    state.questionCount = Math.max(1, Math.min(template.question_count,
      Number(exam.questionCount) || 30));
    state.engine = factory.createEngine(template);

    loadExamKey();
    buildKeyGrid();
    renderExamList();
    updateKeyUI();
    updateHeader();
    activateTab('key');
  }

  /* ------------------------------------------------------------- photo scan */

  function showProcessing(title, text) {
    processingTitle.textContent = title;
    processingText.textContent = text;
    processing.classList.remove('is-hidden');
    errorBox.classList.add('is-hidden');
    resultEl.classList.add('is-hidden');
    processing.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function showError(message) {
    processing.classList.add('is-hidden');
    resultEl.classList.add('is-hidden');
    errorText.textContent = message;
    errorBox.classList.remove('is-hidden');
    errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  let lastPhoto = null;
  const PHOTO_MAX_SIDE = 1800;

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('This photo could not be opened. Take another photo and try again.'));
      };
      img.src = url;
    });
  }

  /** Draws the photo into a canvas, turned by `rotation` degrees clockwise. */
  function drawPhoto(image, rotation) {
    const scale = Math.min(1, PHOTO_MAX_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));
    const quarter = rotation === 90 || rotation === 270;
    const canvas = document.createElement('canvas');
    canvas.width = quarter ? height : width;
    canvas.height = quarter ? width : height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rotation * Math.PI / 180);
    ctx.drawImage(image, -width / 2, -height / 2, width, height);
    return canvas;
  }

  function scanCanvas(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return state.engine.scanRGBA(
      data.data,
      data.width,
      data.height,
      state.key.slice(),
      state.marks.slice()
    );
  }

  function markedBubbles(result) {
    const bubbles = (result && result.read && result.read.bubbles) || [];
    let marked = 0;
    for (const bubble of bubbles) if (bubble.state === 'MARKED') marked++;
    return marked;
  }

  function betterScan(candidate, current) {
    if (!current) return true;
    if (!candidate.ok) return false;
    if (!current.ok) return true;
    // How well the printed rings line up decides first: a read from the wrong
    // angle can find a mark or two by luck and must not win over a read where
    // the whole grid sits on the bubbles.
    const fit = candidate.ringFit || 0;
    const currentFit = current.ringFit || 0;
    if (fit > currentFit + 0.06) return true;
    if (currentFit > fit + 0.06) return false;
    // A turn that finds a mark or two more is not evidence: a wrong angle can
    // pick up table borders and handwriting.  Only a clear difference counts.
    const marks = markedBubbles(candidate);
    const currentMarks = markedBubbles(current);
    if (marks > currentMarks + 2) return true;
    if (currentMarks > marks + 2) return false;
    return candidate.confidence > current.confidence + 0.05;
  }

  /**
   * Phones disagree about which way up a photo is: some turn it according to the
   * EXIF tag, some hand it over sideways, and a sheet photographed upside down
   * is legitimate too.  So the photo is read as it arrives and, when that finds
   * no mark, again at 180°, 90° and 270°; the angle that finds the most marks
   * wins.  A rotated upload used to score zero because the bubbles were sampled
   * from the wrong part of the page.
   */
  async function scanPhoto(file) {
    const image = await loadImage(file);
    const tried = [];
    let chosen = null;
    for (const rotation of [0, 180, 90, 270]) {
      if (chosen && chosen.result && chosen.result.ok && rotation > 0
          && (chosen.result.ringFit || 0) >= 0.30) break;
      if (rotation) {
        showProcessing('Checking the photo angle…', `Trying the photo turned ${rotation}°.`);
        await sleep(30);
      }
      const canvas = drawPhoto(image, rotation);
      const result = scanCanvas(canvas);
      tried.push(rotation + '° ' + (result.ok
        ? Math.round(result.confidence * 100) + '% match, ' + markedBubbles(result) + ' marks'
        : 'no sheet'));
      if (betterScan(result, chosen && chosen.result)) {
        chosen = { result, canvas, rotation, tried };
      }
      // Stop early only on a read that is definitely lined up.  One or two marks
      // found by luck at a wrong angle is not proof, so the other angles are
      // still tried.
      if (result.ok && markedBubbles(result) > 0 && (result.ringFit || 0) >= 0.30) break;
    }
    return chosen;
  }

  async function handleFile(file, input) {
    if (!file) {
      if (input) input.value = '';
      return;
    }
    if (!keyReady()) {
      if (input) input.value = '';
      activateTab('subjects');
      setMessage(subjectMessage, 'Open an exam and set its answers first.', false);
      return;
    }

    try {
      showProcessing('Reading photo…', 'Preparing the image.');
      await sleep(40);
      const scan = await scanPhoto(file);
      if (!scan || !scan.result || !scan.result.ok) {
        const detail = scan && scan.tried && scan.tried.length
          ? ' Tried ' + scan.tried.join('; ') + '.'
          : '';
        throw new Error((scan && scan.result && scan.result.error
          || 'Could not find the sheet in this photo.')
          + detail + ' Lay the sheet on a plain surface, fill the picture with it and try again.');
      }

      showProcessing('Grading…', 'Measuring the marked bubbles.');
      await sleep(30);
      lastPhoto = scan;
      renderResult(scan.result);
    } catch (error) {
      showError(error && error.message ? error.message : 'Something went wrong while reading the sheet.');
    } finally {
      if (input) input.value = '';
    }
  }

  function answerLabel(index) {
    if (!Number.isInteger(index)) return '—';
    const opt = OPTIONS[index];
    return opt ? opt.label : '—';
  }

  function renderResult(result) {
    const grade = result.grade;
    processing.classList.add('is-hidden');
    errorBox.classList.add('is-hidden');
    resultEl.classList.remove('is-hidden');

    resultTotal.textContent = String(grade.total);
    resultMax.textContent = String(grade.maximum);
    const pct = grade.maximum ? (grade.total / grade.maximum) * 100 : 0;
    let verdict = 'Needs review';
    if (pct >= 90) verdict = 'Excellent';
    else if (pct >= 75) verdict = 'Good';
    else if (pct >= 50) verdict = 'Keep practising';
    resultVerdict.textContent = `${verdict} · ${grade.correct} of ${grade.questionCount} correct`;

    statCorrect.textContent = String(grade.correct);
    statWrong.textContent = String(grade.wrong);
    statBlank.textContent = String(grade.blank);
    statMulti.textContent = String(grade.multi);
    statReview.textContent = String(grade.review);

    const modeText = result.mode === 'frame'
      ? 'Pink alignment frame'
      : result.mode === 'page'
        ? 'Page outline'
        : 'Full-page scan';
    scanModeEl.textContent = `${modeText} · ${Math.round(result.confidence * 100)}% match`;

    drawPreview(result);
    drawPhotoOverlay(lastPhoto, result);
    renderBreakdown(grade);

    resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderBreakdown(grade) {
    breakdownEl.innerHTML = grade.details.map(detail => {
      const selected = detail.selected.length
        ? detail.selected.map(answerLabel).join(', ')
        : '—';
      const keyText = detail.answer === null ? '—' : answerLabel(detail.answer);
      const markText = detail.mark === 1 ? '' : ` · ${detail.mark} marks`;
      return `
        <div class="q-line">
          <span class="q">Q${String(detail.q).padStart(2, '0')}</span>
          <span class="answers">
            <span class="answer-pill">${selected}</span>
            <span class="answer-pill key">${keyText}</span>
          </span>
          <span class="badge ${detail.outcome}">${detail.outcome.replace('_', ' ')}${markText}</span>
        </div>`;
    }).join('');
  }

  function drawCircle(ctx, x, y, r, color, width) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function drawPreview(result) {
    const template = state.template;
    const frame = template.frame_mm;
    const cropTop = Math.max(0, Math.round(mmToPx(frame[1])));
    const cropBottom = Math.min(result.warp.height, Math.round(mmToPx(frame[3])));
    const cropHeight = Math.max(1, cropBottom - cropTop);
    const pageWidth = result.warp.width;

    previewCanvas.width = pageWidth;
    previewCanvas.height = cropHeight;
    const ctx = previewCanvas.getContext('2d');
    const crop = ctx.createImageData(pageWidth, cropHeight);
    const src = result.warp.data;
    for (let y = 0; y < cropHeight; y++) {
      const start = ((y + cropTop) * pageWidth) * 4;
      crop.data.set(src.subarray(start, start + pageWidth * 4), y * pageWidth * 4);
    }
    ctx.putImageData(crop, 0, 0);

    const byBubble = new Map();
    result.read.bubbles.forEach(b => byBubble.set(`${b.q}:${b.option}`, b));
    const color = {
      CORRECT: '#177a45',
      WRONG: '#c0392b',
      BLUE: '#2478c4',
      MULTI: '#7b3fa0',
      REVIEW: '#a96500',
      BLANK: '#a96500',
    };
    const radius = state.engine.BUBBLES[0].r;
    const line = Math.max(2, Math.round(radius * 0.18));

    result.grade.details.forEach(detail => {
      const selected = detail.selected;
      if (detail.outcome === 'CORRECT') {
        selected.forEach(option => {
          const b = byBubble.get(`${detail.q}:${option}`);
          if (b) drawCircle(ctx, b.x, b.y - cropTop, radius * 1.12, color.CORRECT, line);
        });
      } else if (detail.outcome === 'WRONG' || detail.outcome === 'MULTI') {
        selected.forEach(option => {
          const b = byBubble.get(`${detail.q}:${option}`);
          if (b) {
            drawCircle(ctx, b.x, b.y - cropTop, radius * 1.12,
              detail.outcome === 'MULTI' ? color.MULTI : color.WRONG, line);
          }
        });
        if (Number.isInteger(detail.answer) && !selected.includes(detail.answer)) {
          const b = byBubble.get(`${detail.q}:${detail.answer}`);
          if (b) drawCircle(ctx, b.x, b.y - cropTop, radius * 1.12, color.BLUE, line);
        }
      } else {
        const b = byBubble.get(`${detail.q}:0`);
        if (b) {
          ctx.beginPath();
          ctx.arc(b.x - radius * 1.7, b.y - cropTop, radius * 0.28, 0, Math.PI * 2);
          ctx.fillStyle = color[detail.outcome] || color.BLANK;
          ctx.fill();
        }
      }
    });

    const region = template.regions && template.regions.total_marks;
    if (region) {
      const x = mmToPx(region.x_mm + region.w_mm / 2);
      const y = mmToPx(region.y_mm + region.h_mm / 2) - cropTop;
      ctx.fillStyle = '#c0392b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = `800 ${Math.round(mmToPx(10))}px system-ui, sans-serif`;
      ctx.fillText(String(result.grade.total), x, y);
    }
  }

  /*
   * The same AR marks the live camera draws, on top of the photo that was
   * scanned: green rings on the answers that match the key, red on the ones
   * that do not, blue on the key of a wrong answer, a purple M on a double mark,
   * an amber ? on a blank, and the total written into the printed mark box.
   */
  function drawPhotoOverlay(scan, result) {
    if (!photoCanvas || !scan || !scan.canvas) return;
    const source = scan.canvas;
    const scale = Math.min(1, 1200 / Math.max(source.width, source.height));
    photoCanvas.width = Math.max(1, Math.round(source.width * scale));
    photoCanvas.height = Math.max(1, Math.round(source.height * scale));
    const ctx = photoCanvas.getContext('2d');
    ctx.clearRect(0, 0, photoCanvas.width, photoCanvas.height);
    ctx.drawImage(source, 0, 0, photoCanvas.width, photoCanvas.height);
    if (!result || !result.corners) return;

    const engine = state.engine;
    const dst = result.mode === 'frame' ? engine.FRAME_DST : engine.PAGE_DST;
    const H = engine.homography(dst, result.corners);
    if (!H) return;
    paintMarks(ctx, result, (x, y) => {
      const point = engine.mapPagePoint(H, x, y, result.align);
      return [point[0] * scale, point[1] * scale];
    });
  }

  function resetResultView() {
    processing.classList.add('is-hidden');
    errorBox.classList.add('is-hidden');
    resultEl.classList.add('is-hidden');
  }

  /* ----------------------------------------------------------------- live AR */

  const LIVE_COLORS = {
    green: '#35e08b',
    red: '#ff5546',
    blue: '#54aaff',
    purple: '#c98bff',
    amber: '#ffb23e',
    sheet: 'rgba(255,255,255,.72)',
    sheetOk: 'rgba(53,224,139,.95)',
    sheetWarn: 'rgba(255,178,62,.95)',
  };
  const LIVE_MAX_SIDE = 1280;
  const TAU = Math.PI * 2;

  const live = {
    running: false,
    busy: false,
    frozen: false,
    voice: true,
    armed: true,
    result: null,
    stream: null,
    timer: null,
    fusion: null,
    smoothCorners: null,
    lastQuad: null,
    lastResultQuad: null,
    stableSince: 0,
    absenceSince: 0,
    shiftSince: 0,
    lastScanAt: 0,
    retryAt: 0,
    procW: 0,
    procH: 0,
    startToken: 0,
    lastDetection: null,
  };

  const procCanvas = document.createElement('canvas');
  const procCtx = procCanvas.getContext('2d', { willReadFrequently: true });
  let cameraPermissionResolver = null;

  function normalizeCorners(corners, width, height) {
    return corners.map(([x, y]) => [x / width, y / height]);
  }

  function maxCornerDelta(a, b) {
    if (!a || !b) return Infinity;
    let delta = 0;
    for (let i = 0; i < 4; i++) {
      delta = Math.max(delta, Math.abs(a[i][0] - b[i][0]), Math.abs(a[i][1] - b[i][1]));
    }
    return delta;
  }

  function setLiveStatus(text, kind) {
    arStatus.textContent = text;
    arStatus.className = 'ar-status' + (kind ? ` ${kind}` : '');
  }

  function showLiveError(message) {
    liveError.textContent = message;
    liveError.classList.remove('is-hidden');
  }

  function clearLiveError() {
    liveError.textContent = '';
    liveError.classList.add('is-hidden');
  }

  function updateVoiceUI() {
    toggleVoice.textContent = live.voice ? 'Voice: on' : 'Voice: off';
    toggleVoice.classList.toggle('voice-on', live.voice);
    toggleVoice.setAttribute('aria-pressed', live.voice ? 'true' : 'false');
  }

  function speakPhrase(phrase) {
    if (!live.voice || !phrase) return;
    const bridge = window.AndroidBridge;
    if (bridge && typeof bridge.speak === 'function') {
      try {
        bridge.speak(phrase);
        return;
      } catch (error) {
        // Fall through to the Web Speech API below.
      }
    }
    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(phrase);
        utterance.lang = 'en-US';
        utterance.rate = 0.98;
        window.speechSynthesis.speak(utterance);
      } catch (error) {
        // Voice is an enhancement; the mark is still shown on screen.
      }
    }
  }

  function requestCameraPermission() {
    const bridge = window.AndroidBridge;
    if (!bridge || typeof bridge.requestCamera !== 'function') {
      return Promise.resolve(null);
    }
    return new Promise(resolve => {
      cameraPermissionResolver = resolve;
      try {
        bridge.requestCamera();
      } catch (error) {
        cameraPermissionResolver = null;
        resolve(false);
        return;
      }
      window.setTimeout(() => {
        if (cameraPermissionResolver === resolve) {
          cameraPermissionResolver = null;
          resolve(null);
        }
      }, 20000);
    });
  }

  window.__onCameraPermission = (granted) => {
    if (!cameraPermissionResolver) return;
    const resolve = cameraPermissionResolver;
    cameraPermissionResolver = null;
    resolve(Boolean(granted));
  };

  window.__onAndroidPause = () => {
    if (live.running) stopLiveCamera();
  };

  function fitLiveStage() {
    if (!arVideo.videoWidth || !arVideo.videoHeight) return;
    arStage.style.aspectRatio = `${arVideo.videoWidth} / ${arVideo.videoHeight}`;
  }

  function grabLiveFrame() {
    const vw = arVideo.videoWidth;
    const vh = arVideo.videoHeight;
    if (!vw || !vh || arVideo.readyState < 2) return null;
    const scale = Math.min(1, LIVE_MAX_SIDE / Math.max(vw, vh));
    const width = Math.max(1, Math.round(vw * scale));
    const height = Math.max(1, Math.round(vh * scale));
    if (procCanvas.width !== width || procCanvas.height !== height) {
      procCanvas.width = width;
      procCanvas.height = height;
    }
    procCtx.drawImage(arVideo, 0, 0, width, height);
    return procCtx.getImageData(0, 0, width, height);
  }

  function clearLiveOverlay() {
    const width = arCanvas.clientWidth;
    const height = arCanvas.clientHeight;
    if (!width || !height) return;
    const ctx = arCanvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, arCanvas.width, arCanvas.height);
  }

  function renderLiveStats(grade) {
    if (!grade) {
      liveStats.classList.add('is-hidden');
      arHudMark.classList.add('is-hidden');
      return;
    }
    liveCorrect.textContent = String(grade.correct);
    liveWrong.textContent = String(grade.wrong);
    liveBlank.textContent = String(grade.blank);
    liveMulti.textContent = String(grade.multi);
    liveReview.textContent = String(grade.review);
    liveStats.classList.remove('is-hidden');
    arHudTotal.textContent = String(grade.total);
    arHudMax.textContent = `/${grade.maximum}`;
    arHudMark.classList.remove('is-hidden');
  }

  function clearLiveResult() {
    live.result = null;
    live.lastResultQuad = null;
    live.shiftSince = 0;
    renderLiveStats(null);
  }

  /*
   * Where the aim box sits, in fractions of the camera frame.  Its shape is the
   * printed answer table plus the mark box beside it, so a sheet that fills the
   * box is exactly the part of the page the reader measures.
   */
  function aimGuideNorm() {
    const g = state.engine.GUIDE_MM;
    const aspect = (g[2] - g[0]) / (g[3] - g[1]);
    const videoW = arVideo.videoWidth || 1080;
    const videoH = arVideo.videoHeight || 1920;
    const shape = aspect / (videoW / videoH);
    let widthN = 0.90;
    let heightN = widthN / shape;
    if (heightN > 0.78) {
      heightN = 0.78;
      widthN = heightN * shape;
    }
    const x0 = (1 - widthN) / 2;
    const y0 = (1 - heightN) / 2;
    return [x0, y0, x0 + widthN, y0 + heightN];
  }

  function drawAimGuide(ctx, stageW, stageH, videoW, videoH) {
    const g = aimGuideNorm();
    const fit = Math.min(stageW / videoW, stageH / videoH);
    const offsetX = (stageW - videoW * fit) / 2;
    const offsetY = (stageH - videoH * fit) / 2;
    const x = g[0] * videoW * fit + offsetX;
    const y = g[1] * videoH * fit + offsetY;
    const w = (g[2] - g[0]) * videoW * fit;
    const h = (g[3] - g[1]) * videoH * fit;
    const r = Math.min(16, w * 0.04);

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, stageW, stageH);
    ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
    ctx.fillStyle = 'rgba(5, 12, 20, 0.42)';
    ctx.fill('evenodd');

    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
    ctx.setLineDash([9, 7]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,255,255,.85)';
    ctx.stroke();

    // Corner ticks: a plain dashed outline is hard to aim with on a busy desk.
    ctx.setLineDash([]);
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = 'rgba(255,255,255,.95)';
    const tick = Math.min(w, h) * 0.07;
    const corners = [[x, y, 1, 1], [x + w, y, -1, 1], [x + w, y + h, -1, -1], [x, y + h, 1, -1]];
    for (const [cx, cy, sx, sy] of corners) {
      ctx.beginPath();
      ctx.moveTo(cx + sx * tick, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + sy * tick);
      ctx.stroke();
    }

    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(255,255,255,.95)';
    ctx.fillText('Answer table + mark box', x + w / 2, y - 6);
    ctx.restore();
  }

  function drawLiveOverlay(result, detection, procW, procH) {
    const stageW = arStage.clientWidth;
    const stageH = arStage.clientHeight;
    if (!stageW || !stageH) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const pixelW = Math.max(1, Math.round(stageW * dpr));
    const pixelH = Math.max(1, Math.round(stageH * dpr));
    if (arCanvas.width !== pixelW || arCanvas.height !== pixelH) {
      arCanvas.width = pixelW;
      arCanvas.height = pixelH;
    }
    const ctx = arCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, stageW, stageH);

    const videoW = arVideo.videoWidth || procW;
    const videoH = arVideo.videoHeight || procH;
    const procScale = procW / videoW;
    const fit = Math.min(stageW / videoW, stageH / videoH);
    const offsetX = (stageW - videoW * fit) / 2;
    const offsetY = (stageH - videoH * fit) / 2;
    const stageScale = fit / procScale;

    const toStage = (point) => [
      point[0] * stageScale + offsetX,
      point[1] * stageScale + offsetY,
    ];

    // The aim box: the answer table plus the printed mark box, and nothing
    // else.  A teacher lines the sheet up in it, and the reader then trusts the
    // box's own corners instead of hunting for the page outline -- which is the
    // part a second sheet, a table edge or a notebook can imitate.
    if (!result) drawAimGuide(ctx, stageW, stageH, videoW, videoH);

    const shown = result && result.corners
      ? { mode: result.mode, corners: result.corners, align: result.align }
      : detection;
    if (!shown || shown.mode === 'none' || !shown.corners) return;

    const outline = shown.corners.map(toStage);
    ctx.beginPath();
    outline.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point[0], point[1]);
      else ctx.lineTo(point[0], point[1]);
    });
    ctx.closePath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = result
      ? (result.grade.review > 0 ? LIVE_COLORS.sheetWarn : LIVE_COLORS.sheetOk)
      : LIVE_COLORS.sheet;
    ctx.stroke();

    if (!result) return;

    const dst = shown.mode === 'frame' ? state.engine.FRAME_DST : state.engine.PAGE_DST;
    const H = state.engine.homography(dst, shown.corners);
    if (!H) return;
    paintMarks(ctx, result, (x, y) => {
      const point = state.engine.mapPagePoint(H, x, y, shown.align);
      return [point[0] * stageScale + offsetX, point[1] * stageScale + offsetY];
    });
  }

  /*
   * Draws the result on any surface, given a function that maps page pixels to
   * that surface: the live video and the scanned photo share it, so both look
   * the same.
   */
  function paintMarks(ctx, result, project) {
    if (!ctx || !result || !result.grade) return;
    const engine = state.engine;
    const template = state.template;

    const drawRing = (bubble, color) => {
      if (!bubble) return;
      const center = project(bubble.x, bubble.y);
      const edgeX = project(bubble.x + bubble.r, bubble.y);
      const edgeY = project(bubble.x, bubble.y + bubble.r);
      const radius = Math.max(
        4,
        0.5 * (
          Math.hypot(edgeX[0] - center[0], edgeX[1] - center[1]) +
          Math.hypot(edgeY[0] - center[0], edgeY[1] - center[1])
        )
      );
      const line = Math.max(2.2, Math.min(7, radius * 0.30));
      ctx.beginPath();
      ctx.arc(center[0], center[1], radius * 1.08, 0, TAU);
      ctx.lineWidth = line + 2.4;
      ctx.strokeStyle = 'rgba(0,0,0,.42)';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(center[0], center[1], radius * 1.08, 0, TAU);
      ctx.lineWidth = line;
      ctx.strokeStyle = color;
      ctx.stroke();
    };

    const drawMarker = (bubble, color, text) => {
      if (!bubble) return;
      const center = project(bubble.x - bubble.r * 2.0, bubble.y - bubble.r * 1.45);
      const edge = project(bubble.x - bubble.r * 1.0, bubble.y - bubble.r * 1.45);
      const radius = Math.max(5, Math.hypot(edge[0] - center[0], edge[1] - center[1]) * 0.55);
      ctx.beginPath();
      ctx.arc(center[0], center[1], radius, 0, TAU);
      ctx.fillStyle = color;
      ctx.fill();
      if (text) {
        ctx.fillStyle = '#0b1512';
        ctx.font = `800 ${Math.max(7, radius * 1.25)}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, center[0], center[1] + radius * 0.06);
      }
    };

    result.grade.details.forEach(detail => {
      const selected = detail.selected || [];
      if (detail.outcome === 'CORRECT') {
        selected.forEach(option => drawRing(engine.bubblePx(detail.q, option), LIVE_COLORS.green));
      } else if (detail.outcome === 'WRONG') {
        selected.forEach(option => drawRing(engine.bubblePx(detail.q, option), LIVE_COLORS.red));
        if (Number.isInteger(detail.answer) && !selected.includes(detail.answer)) {
          drawRing(engine.bubblePx(detail.q, detail.answer), LIVE_COLORS.blue);
        }
      } else if (detail.outcome === 'MULTI') {
        selected.forEach(option => drawRing(engine.bubblePx(detail.q, option), LIVE_COLORS.purple));
        if (Number.isInteger(detail.answer) && !selected.includes(detail.answer)) {
          drawRing(engine.bubblePx(detail.q, detail.answer), LIVE_COLORS.blue);
        }
        drawMarker(engine.bubblePx(detail.q, 0), LIVE_COLORS.purple, 'M');
      } else {
        drawMarker(
          engine.bubblePx(detail.q, 0),
          LIVE_COLORS.amber,
          detail.outcome === 'BLANK' ? '?' : '!'
        );
      }
    });

    const region = template && template.regions && template.regions.total_marks;
    if (!region) return;
    const x0 = mmToPx(region.x_mm);
    const y0 = mmToPx(region.y_mm);
    const x1 = x0 + mmToPx(region.w_mm);
    const y1 = y0 + mmToPx(region.h_mm);
    const p0 = project(x0, y0);
    const p1 = project(x1, y1);
    const boxHeight = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
    const size = Math.max(20, Math.min(130, boxHeight * 0.58));
    const cx = (p0[0] + p1[0]) / 2;
    const cy = (p0[1] + p1[1]) / 2;
    const text = String(result.grade.total);
    ctx.font = `800 ${size}px system-ui, -apple-system, Roboto, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(2.5, size * 0.11);
    ctx.strokeStyle = 'rgba(24, 26, 24, .88)';
    ctx.strokeText(text, cx, cy);
    ctx.fillStyle = '#d42318';
    ctx.fillText(text, cx, cy);
  }

  function scheduleLive(delay) {
    if (!live.running) return;
    if (live.timer) clearTimeout(live.timer);
    live.timer = window.setTimeout(liveLoop, delay);
  }

  function lockResult(grade, detection, normalized) {
    // One sheet, one result.  The scan stops here and the result stays on
    // screen until the teacher taps "Scan next sheet": a camera left running
    // on the same paper re-read it with slight drift and could flip answers.
    live.frozen = true;
    live.result = {
      ok: true,
      mode: detection.mode,
      confidence: detection.confidence,
      orientation: detection.orientation,
      corners: detection.corners,
      read: { bubbles: [], questions: {}, calibration: { threshold: 0, method: 'fusion' } },
      grade,
      fused: true,
    };
    live.armed = false;
    live.lastResultQuad = normalized;
    live.lastScanAt = performance.now();
    live.shiftSince = 0;
    renderLiveStats(grade);
    setLiveStatus(`Mark ${grade.total} / ${grade.maximum}`, 'ok');
    speakPhrase(`Mark ${grade.total} out of ${grade.maximum}.`);
    if (nextLive) nextLive.classList.remove('is-hidden');
  }

  /** Re-opens the live reader for the next sheet after a locked result. */
  function armNextSheet() {
    if (!live.running) return;
    live.frozen = false;
    live.result = null;
    live.armed = true;
    live.shiftSince = 0;
    live.absenceSince = 0;
    live.lastResultQuad = null;
    live.stableSince = 0;
    if (live.fusion) live.fusion.reset();
    if (nextLive) nextLive.classList.add('is-hidden');
    renderLiveStats(null);
    clearLiveResult();
    setLiveStatus('Show the next sheet');
    scheduleLive(60);
  }

  async function liveLoop() {
    live.timer = null;
    if (!live.running || live.busy || live.frozen) return;

    const now = performance.now();
    if (now < live.retryAt) {
      scheduleLive(live.retryAt - now + 40);
      return;
    }

    const image = grabLiveFrame();
    if (!image) {
      setLiveStatus('Waiting for the camera…');
      scheduleLive(220);
      return;
    }
    live.procW = image.width;
    live.procH = image.height;

    const engine = state.engine;
    let detection = null;
    try {
      detection = engine.detectFrame(image.data, image.width, image.height);
    } catch (error) {
      setLiveStatus('Reader error — try again', 'err');
      scheduleLive(500);
      return;
    }
    if (!live.running) return;

    // One blurred, tilted or half-shadowed frame should not throw the sheet
    // away: hold the last geometry for a moment so a shaking hand keeps reading
    // instead of flashing "show the sheet" every few frames.
    if ((!detection || detection.mode === 'none' || !detection.corners)
        && live.lastDetection && live.lastSeenAt
        && now - live.lastSeenAt < 1200) {
      detection = live.lastDetection;
    }

    if (!detection || detection.mode === 'none' || !detection.corners) {
      live.lastQuad = null;
      live.lastDetection = null;
      live.stableSince = 0;
      live.shiftSince = 0;
      live.smoothCorners = null;
      if (!live.absenceSince) live.absenceSince = now;
      if (now - live.absenceSince > 800) live.fusion.reset();
      if (live.result && now - live.absenceSince > 1100) {
        clearLiveResult();
        live.armed = true;
        setLiveStatus('Ready for the next sheet');
      } else if (!live.result) {
        setLiveStatus('Show the sheet');
      }
      drawLiveOverlay(null, null, image.width, image.height);
      scheduleLive(150);
      return;
    }

    live.absenceSince = 0;
    live.lastDetection = detection;
    live.lastSeenAt = now;

    // The ring-likelihood check could not tell which way the sheet is turned
    // (two interpretations scored the same).  Reading it would risk marking the
    // wrong grid, so show why nothing happens instead of a misleading result.
    if (detection.ambiguous) {
      live.fusion.reset();
      setLiveStatus('Turn the sheet upright — top of the page towards the top of the screen', 'warn');
      drawLiveOverlay(null, detection, image.width, image.height);
      scheduleLive(150);
      return;
    }

    // Smooth the four corners: small jitter must not re-solve the geometry
    // from scratch every frame, but a real move snaps immediately.
    const raw = detection.corners;
    const maxSide = Math.max(image.width, image.height);
    if (!live.smoothCorners) {
      live.smoothCorners = raw.map(p => [p[0], p[1]]);
    } else {
      const jump = maxCornerDelta(raw, live.smoothCorners);
      if (jump > maxSide * 0.20) {
        live.smoothCorners = raw.map(p => [p[0], p[1]]);
        live.fusion.reset();
      } else {
        live.smoothCorners = live.smoothCorners.map((p, i) => [
          p[0] + (raw[i][0] - p[0]) * 0.45,
          p[1] + (raw[i][1] - p[1]) * 0.45,
        ]);
      }
    }

    const normalized = normalizeCorners(raw, image.width, image.height);
    const moved = live.lastQuad ? maxCornerDelta(normalized, live.lastQuad) : 1;
    live.lastQuad = normalized;
    const moving = moved > 0.025;

    if (live.result && live.lastResultQuad &&
        maxCornerDelta(normalized, live.lastResultQuad) > 0.11) {
      if (!live.shiftSince) live.shiftSince = now;
      if (now - live.shiftSince > 500) {
        clearLiveResult();
        live.armed = true;
        live.shiftSince = 0;
        live.fusion.reset();
        setLiveStatus('New sheet — hold still');
      }
    } else if (live.result) {
      live.shiftSince = 0;
    }

    if (!live.result && live.armed) {
      if (moving) {
        setLiveStatus('Hold still…');
      } else {
        const dst = detection.mode === 'frame' ? engine.FRAME_DST : engine.PAGE_DST;
        const H = engine.homography(dst, live.smoothCorners);
        // Read the answer area only: the bubble block is cropped out of the
        // frame (and enlarged when the sheet is far), so the header, the
        // instructions and any handwriting never affect the measurement.
        const area = engine.readRegion(image.data, image.width, image.height, H);
        const readings = area
          ? area.readings
          : engine.readProjected(image.data, image.width, image.height, H);

        // The printed rings say whether the grid really sits on the sheet.  A
        // frame that does not line up is dropped instead of being averaged into
        // the answer: one bad frame must never become a mark.
        const lined = !area || area.ringFit === undefined || area.ringFit >= 0.06;

        if (!readings || !lined) {
          live.fusion.reset();
          setLiveStatus(readings
            ? 'Line up the whole answer area'
            : 'Move closer — the bubbles are too small', 'warn');
        } else {
          live.fusion.push(readings);
          if (live.fusion.isSettled()) {
            // The tracker above only decides *when* the sheet is worth reading.
            // The answer itself comes from the full photo engine, on this exact
            // frame: a live scan and an uploaded photo of the same paper must
            // never disagree, and a printed sheet is sensitive enough that a
            // quick per-frame projection is not good enough to grade with.
            setLiveStatus('Reading the sheet…');
            // Let the status paint before the full read takes the main thread:
            // half a second of a frozen screen looks like a crash.
            await sleep(30);
            if (!live.running || live.busy || live.frozen) return;
            const full = engine.scanRGBA(
              image.data, image.width, image.height,
              state.key.slice(), state.marks.slice(), detection
            );
            if (full && full.ok) {
              lockResult(full.grade, detection, normalized);
              live.result.read = full.read;
              live.result.fused = false;
              live.result.ringFit = full.ringFit;
              // The read may have picked the other destination (frame or page)
              // and its own alignment; the overlay must draw with exactly those,
              // otherwise the circles land beside the bubbles and a correct read
              // looks like a wrong one.
              live.result.mode = full.mode;
              live.result.corners = full.corners;
              live.result.align = full.align;
              live.result.orientation = full.orientation;
            } else {
              // Not a clean read yet: keep the camera on the paper and try the
              // next frame instead of showing a number that could be wrong.
              live.fusion.reset();
              setLiveStatus('Hold still — lining up the sheet', 'warn');
            }
          } else {
            setLiveStatus(`Reading… ${Math.round(live.fusion.progress() * 100)}%`);
          }
        }
      }
    } else if (live.result) {
      setLiveStatus(
        `Mark ${live.result.grade.total} / ${live.result.grade.maximum} · show next sheet`,
        'ok'
      );
    }

    drawLiveOverlay(live.result, detection, image.width, image.height);
    scheduleLive(live.result ? 200 : 80);
  }

  async function startLiveCamera() {
    if (!keyReady()) {
      activateTab('subjects');
      setMessage(subjectMessage, 'Open an exam and set its answers first.', false);
      return;
    }
    if (live.running) return;

    clearLiveError();
    const token = ++live.startToken;
    startLive.disabled = true;
    setLiveStatus('Starting camera…');

    try {
      const granted = await requestCameraPermission();
      if (token !== live.startToken) return;
      if (granted === false) {
        throw new Error('Camera permission was denied. Allow camera access for OMR Checker in Android Settings, then try again.');
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('This phone does not support a live camera inside the app.');
      }

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });
      } catch (firstError) {
        if (firstError && (firstError.name === 'NotAllowedError' || firstError.name === 'SecurityError')) {
          throw firstError;
        }
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      }
      if (token !== live.startToken) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      live.stream = stream;
      arVideo.srcObject = stream;
      await arVideo.play();
      if (token !== live.startToken) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      live.running = true;
      live.busy = false;
      live.frozen = false;
      live.armed = true;
      live.result = null;
      live.fusion = state.engine.createFusion({ need: 5, agree: 3, window: 9 });
      live.smoothCorners = null;
      live.lastQuad = null;
      live.lastResultQuad = null;
      live.lastDetection = null;
      live.stableSince = 0;
      live.absenceSince = 0;
      live.shiftSince = 0;
      live.lastScanAt = 0;
      live.retryAt = 0;

      renderLiveStats(null);
      arPlaceholder.classList.add('is-hidden');
      arStage.classList.remove('is-hidden');
      startLive.classList.add('is-hidden');
      stopLive.classList.remove('is-hidden');
      if (nextLive) nextLive.classList.add('is-hidden');
      fitLiveStage();
      clearLiveOverlay();
      setLiveStatus('Show the sheet');
      scheduleLive(300);
    } catch (error) {
      const name = error && error.name;
      let message = error && error.message ? error.message : 'Could not start the camera.';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        message = 'Camera permission was denied. Allow camera access for OMR Checker in Android Settings, then try again.';
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        message = 'No usable camera was found on this phone.';
      }
      stopLiveCamera();
      showLiveError(message);
      setLiveStatus('Camera not running', 'err');
    } finally {
      if (token === live.startToken) {
        startLive.disabled = !keyReady();
      }
    }
  }

  function stopLiveCamera() {
    live.startToken++;
    live.running = false;
    live.busy = false;
    live.frozen = false;
    if (live.timer) {
      clearTimeout(live.timer);
      live.timer = null;
    }
    if (live.stream) {
      live.stream.getTracks().forEach(track => track.stop());
      live.stream = null;
    }
    try {
      arVideo.pause();
    } catch (error) {
      // The element may already be detached or stopped.
    }
    arVideo.srcObject = null;

    live.result = null;
    live.fusion = null;
    live.smoothCorners = null;
    live.lastQuad = null;
    live.lastResultQuad = null;
    live.lastDetection = null;
    live.stableSince = 0;
    live.absenceSince = 0;
    live.shiftSince = 0;
    live.lastScanAt = 0;
    live.retryAt = 0;

    renderLiveStats(null);
    clearLiveOverlay();
    arStage.classList.add('is-hidden');
    arPlaceholder.classList.remove('is-hidden');
    startLive.classList.remove('is-hidden');
    stopLive.classList.add('is-hidden');
    if (nextLive) nextLive.classList.add('is-hidden');
    startLive.disabled = !keyReady();
    arStage.style.aspectRatio = '';
    setLiveStatus('Point the camera at a sheet');
  }

  /* --------------------------------------------------------------- wiring up */

  function buildExamCountOptions() {
    examCount.innerHTML = '';
    sheets.sizeChoices.forEach(count => {
      const option = document.createElement('option');
      option.value = String(count);
      option.textContent = `${count} questions`;
      examCount.appendChild(option);
    });
    examCount.value = '30';
  }

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => activateTab(tab.dataset.tab));
  });

  // The home cards, the guide banner and the header link jump to a page.
  document.querySelectorAll('[data-go]').forEach(button => {
    button.addEventListener('click', () => activateTab(button.dataset.go));
  });

  addSubjectBtn.addEventListener('click', () => {
    const name = subjectName.value.trim();
    if (!name) {
      setMessage(subjectMessage, 'Enter a subject name.', false);
      subjectName.focus();
      return;
    }
    try {
      const created = store.addSubject(name);
      subjectName.value = '';
      setMessage(subjectMessage, `Added ${name}. Add an exam for it below.`, true);
      refreshSubjects(created && created.id);
    } catch (error) {
      setMessage(subjectMessage, error.message, false);
    }
  });

  subjectName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') addSubjectBtn.click();
  });
  subjectName.addEventListener('input', () => setMessage(subjectMessage, '', false));

  renameSubjectBtn.addEventListener('click', () => {
    subjectRename.classList.remove('is-hidden');
    subjectRenameInput.value = detailName.textContent;
    subjectRenameInput.focus();
  });

  subjectRenameCancel.addEventListener('click', () => subjectRename.classList.add('is-hidden'));

  subjectRenameSave.addEventListener('click', () => {
    const name = subjectRenameInput.value.trim();
    if (!name) {
      setMessage(subjectMessage, 'Enter a subject name.', false);
      return;
    }
    try {
      store.renameSubject(state.subjectId, name);
      subjectRename.classList.add('is-hidden');
      setMessage(subjectMessage, 'Subject renamed.', true);
      refreshSubjects(state.subjectId);
      updateKeyUI();
      updateHeader();
    } catch (error) {
      setMessage(subjectMessage, error.message, false);
    }
  });

  deleteSubjectBtn.addEventListener('click', () => {
    if (!state.subjectId) return;
    if (!armDelete('subject', state.subjectId, deleteSubjectBtn)) return;
    try {
      store.deleteSubject(state.subjectId);
      state.subjectId = null;
      if (state.examId) {
        state.examId = null;
        state.exam = null;
        loadExamKey();
        buildKeyGrid();
        updateKeyUI();
      }
      subjectDetail.classList.add('is-hidden');
      setMessage(subjectMessage, 'Subject and its exams deleted.', true);
      refreshSubjects();
      updateHeader();
    } catch (error) {
      setMessage(subjectMessage, error.message, false);
    }
  });

  addExamBtn.addEventListener('click', () => {
    if (!state.subjectId) {
      setMessage(examMessage, 'Add a subject first.', false);
      return;
    }
    const name = examName.value.trim();
    const count = Number(examCount.value) || 30;
    if (!name) {
      setMessage(examMessage, 'Enter an exam name.', false);
      examName.focus();
      return;
    }
    try {
      const created = store.addExam(state.subjectId, name, count, 'omr_' + count);
      examName.value = '';
      setMessage(examMessage, `Added ${name} (${count} questions).`, true);
      refreshExams();
      if (created && created.id) openExam(created.id);
    } catch (error) {
      setMessage(examMessage, error.message, false);
    }
  });

  examName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') addExamBtn.click();
  });
  examName.addEventListener('input', () => setMessage(examMessage, '', false));

  keyGrid.addEventListener('click', (event) => {
    const button = event.target.closest('.opt');
    if (!button) return;
    setKey(Number(button.dataset.q) - 1, Number(button.dataset.opt));
  });

  keyGrid.addEventListener('change', (event) => {
    const input = event.target.closest('.mark-input');
    if (!input) return;
    setMark(Number(input.dataset.q) - 1, input.value);
  });

  bulkApply.addEventListener('click', () => {
    if (!state.examId) {
      setMessage(bulkMessage, 'Open an exam first.', false);
      return;
    }
    const parsed = parseBulk(bulkInput.value);
    if (!parsed) {
      setMessage(bulkMessage,
        `Enter exactly ${state.questionCount} answers, each A–D (or 1–4).`, false);
      bulkInput.focus();
      return;
    }
    for (let i = 0; i < state.questionCount; i++) state.key[i] = parsed[i];
    state.dirty = true;
    bulkInput.value = '';
    setMessage(bulkMessage, `${state.questionCount} answers applied. Save to keep them.`, true);
    updateKeyUI();
  });

  bulkInput.addEventListener('input', () => setMessage(bulkMessage, '', false));
  bulkInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') bulkApply.click();
  });

  clearKeyEl.addEventListener('click', () => {
    if (!state.examId) return;
    state.key = new Array(state.questionCount).fill(null);
    state.marks = new Array(state.questionCount).fill(1);
    state.dirty = true;
    setMessage(bulkMessage, 'Answers cleared. Save to keep the change.', true);
    buildKeyGrid();
    updateKeyUI();
  });

  saveKeyEl.addEventListener('click', () => {
    if (!state.examId) return;
    try {
      store.saveKey(state.examId, state.key, state.marks);
      state.dirty = false;
      setMessage(saveMessage,
        `Saved ${countSet()} answers and marks to the phone.`, true);
      updateKeyUI();
      refreshExams();
    } catch (error) {
      setMessage(saveMessage, error.message, false);
    }
  });

  takePhoto.addEventListener('click', () => {
    if (!keyReady()) return;
    cameraInput.click();
  });

  choosePhoto.addEventListener('click', () => {
    if (!keyReady()) return;
    galleryInput.click();
  });

  cameraInput.addEventListener('change', () => handleFile(cameraInput.files && cameraInput.files[0], cameraInput));
  galleryInput.addEventListener('change', () => handleFile(galleryInput.files && galleryInput.files[0], galleryInput));

  retryButton.addEventListener('click', () => {
    resetResultView();
    takePhoto.click();
  });

  scanAgain.addEventListener('click', () => {
    resetResultView();
    document.querySelector('.scan-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  backToKey.addEventListener('click', () => activateTab('key'));

  startLive.addEventListener('click', () => { void startLiveCamera(); });

  // After a locked result the reader waits; this starts it on the next sheet.
  if (nextLive) nextLive.addEventListener('click', armNextSheet);

  stopLive.addEventListener('click', () => {
    stopLiveCamera();
    setLiveStatus('Camera stopped');
  });

  toggleVoice.addEventListener('click', () => {
    live.voice = !live.voice;
    updateVoiceUI();
    if (!live.voice && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (error) {
        // Nothing to cancel.
      }
    }
  });

  arVideo.addEventListener('loadedmetadata', () => {
    fitLiveStage();
    if (live.running) {
      drawLiveOverlay(live.result, live.lastDetection, live.procW, live.procH);
    }
  });

  window.addEventListener('resize', () => {
    if (live.running) {
      drawLiveOverlay(live.result, live.lastDetection, live.procW, live.procH);
    }
  });

  /* ------------------------------------------------------------------- boot */

  // The print module needs the open exam, and a way to refresh the key screen
  // after it sends a whole answer key across.
  window.OMR_APP = {
    exam: () => (state.exam
      ? { id: state.examId, name: state.exam.name || '', count: state.questionCount }
      : null),
    reloadKey: () => { loadExamKey(); buildKeyGrid(); updateKeyUI(); },
  };

  buildExamCountOptions();
  buildKeyGrid();
  refreshSubjects();
  updateKeyUI();
  updateVoiceUI();
  updateHeader();
  setLiveStatus('Point the camera at a sheet');
})();
