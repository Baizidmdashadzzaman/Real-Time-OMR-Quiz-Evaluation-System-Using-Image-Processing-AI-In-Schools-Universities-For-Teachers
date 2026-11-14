/*
 * School paper module.
 *
 * A teacher fills in the school heading (logo, name, address, exam line), types
 * the questions with their four options and taps the correct option.  The module
 * then prints the very same A4 OMR sheet as print/v3 — same measured grid, same
 * bubble positions — with that heading at the top, and can push the answers into
 * the open exam as its answer key.
 *
 * Everything is kept in the phone's own storage; nothing is uploaded.
 */
(function (root) {
  'use strict';

  const DB_KEY = 'omr.paper.v1';
  const LABELS = ['A', 'B', 'C', 'D'];
  const $ = (id) => root.document.getElementById(id);

  /* ---------------------------------------------------------------- storage */

  let memory = null;
  let saveTimer = 0;

  function fresh() {
    return {
      profile: { name: '', address: '', line: '', note: '', logo: '' },
      papers: [],
      activeId: null,
    };
  }

  function readDb() {
    if (memory) return memory;
    try {
      const raw = root.localStorage && root.localStorage.getItem(DB_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      memory = parsed && parsed.profile && Array.isArray(parsed.papers) ? parsed : fresh();
    } catch (error) {
      memory = fresh();
    }
    return memory;
  }

  function flush() {
    try {
      if (root.localStorage) root.localStorage.setItem(DB_KEY, JSON.stringify(memory));
    } catch (error) {
      // Storage full or blocked: the session keeps working from memory.
    }
  }

  function saveSoon(message) {
    if (message) setStatus(message);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(flush, 350);
  }

  const db = readDb();

  /* ----------------------------------------------------------------- papers */

  function blankQuestion() {
    return { text: '', options: ['', '', '', ''], key: null };
  }

  function blankPaper(count, name) {
    return {
      id: 'p' + Date.now().toString(36) + Math.floor(Math.random() * 1000),
      name: name || 'Untitled paper',
      count: count || 30,
      questions: new Array(count || 30).fill(0).map(blankQuestion),
      updatedAt: Date.now(),
    };
  }

  function activePaper() {
    let paper = db.papers.find((p) => p.id === db.activeId);
    if (!paper) {
      paper = db.papers[0] || blankPaper(30, 'Paper 1');
      if (!db.papers.length) db.papers.push(paper);
      db.activeId = paper.id;
    }
    return paper;
  }

  function ensureQuestions(paper) {
    if (!Array.isArray(paper.questions)) paper.questions = [];
    while (paper.questions.length < paper.count) paper.questions.push(blankQuestion());
    paper.questions.length = paper.count;
    paper.questions.forEach((q) => {
      if (!Array.isArray(q.options) || q.options.length !== 4) {
        q.options = (q.options || []).slice(0, 4);
        while (q.options.length < 4) q.options.push('');
      }
      if (!Number.isInteger(q.key) || q.key < 0 || q.key > 3) q.key = null;
      if (typeof q.text !== 'string') q.text = '';
    });
    return paper;
  }

  /* --------------------------------------------------------------- elements */

  const el = {};
  let renderedId = '';

  function setStatus(text, bad) {
    if (!el.message) return;
    el.message.textContent = text || '';
    el.message.classList.toggle('ok', !bad && Boolean(text));
  }

  function setSaved(on) {
    if (el.saved) el.saved.textContent = on ? 'Saved on phone' : 'Draft';
  }

  /* ---------------------------------------------------------------- heading */

  function readHeaderForm() {
    db.profile.name = el.school.value.trim();
    db.profile.address = el.address.value.trim();
    db.profile.line = el.line.value.trim();
    db.profile.note = el.note.value.trim();
  }

  function fillHeaderForm() {
    const p = db.profile;
    el.school.value = p.name || '';
    el.address.value = p.address || '';
    el.line.value = p.line || '';
    el.note.value = p.note || '';
    paintLogo();
  }

  function paintLogo() {
    if (!el.logoBox) return;
    const src = db.profile.logo;
    el.logoBox.innerHTML = src
      ? '<img alt="School logo">'
      : '<span>Add<br>logo</span>';
    if (src) el.logoBox.querySelector('img').src = src;
    if (el.logoClear) el.logoClear.classList.toggle('is-hidden', !src);
  }

  function shrinkLogo(file, done) {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const max = 320;
        const scale = Math.min(1, max / Math.max(img.width, img.height));
        const canvas = root.document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        try {
          done(canvas.toDataURL('image/png'));
        } catch (error) {
          done('');
        }
      };
      img.onerror = () => done('');
      img.src = String(reader.result || '');
    };
    reader.onerror = () => done('');
    reader.readAsDataURL(file);
  }

  /* -------------------------------------------------------------- questions */

  function renderQuestions() {
    const paper = ensureQuestions(activePaper());
    if (!el.questions) return;
    const frag = root.document.createDocumentFragment();

    paper.questions.forEach((q, index) => {
      const row = root.document.createElement('div');
      row.className = 'pq' + (q.key === null ? '' : ' is-keyed');
      row.dataset.i = String(index);

      const head = root.document.createElement('div');
      head.className = 'pq-wrap';

      const num = root.document.createElement('span');
      num.className = 'pq-num';
      num.textContent = String(index + 1).padStart(2, '0');

      const text = root.document.createElement('input');
      text.type = 'text';
      text.className = 'pq-text';
      text.value = q.text;
      text.placeholder = 'Question ' + (index + 1) + ' (optional — kept with the key)';
      text.dataset.i = String(index);

      head.appendChild(num);
      head.appendChild(text);
      row.appendChild(head);

      const opts = root.document.createElement('div');
      opts.className = 'pq-opts';
      for (let option = 0; option < 4; option++) {
        const cell = root.document.createElement('div');
        cell.className = 'pq-opt';

        const key = root.document.createElement('button');
        key.type = 'button';
        key.className = 'pq-key' + (q.key === option ? ' is-on' : '');
        key.textContent = LABELS[option];
        key.dataset.i = String(index);
        key.dataset.o = String(option);
        key.setAttribute('aria-label', 'Question ' + (index + 1) + ' answer ' + LABELS[option]);
        key.title = 'Tap to mark this as the correct answer';

        const input = root.document.createElement('input');
        input.type = 'text';
        input.className = 'pq-opt-input';
        input.value = q.options[option] || '';
        input.placeholder = 'Option ' + LABELS[option];
        input.dataset.i = String(index);
        input.dataset.o = String(option);

        cell.appendChild(key);
        cell.appendChild(input);
        opts.appendChild(cell);
      }
      row.appendChild(opts);
      frag.appendChild(row);
    });

    el.questions.innerHTML = '';
    el.questions.appendChild(frag);
    paintStats();
  }

  function paintStats() {
    const paper = activePaper();
    const keyed = paper.questions.filter((q) => q.key !== null).length;
    const named = paper.questions.filter((q) => q.text.trim()).length;
    el.stats.innerHTML = '';
    const parts = [
      `<span><b>${paper.count}</b> questions</span>`,
      `<span><b>${keyed}</b> answers marked</span>`,
      `<span><b>${named}</b> written</span>`,
    ];
    el.stats.innerHTML = parts.join('');
    el.keysHint.textContent = keyed === paper.count
      ? 'Every question has an answer — this can become the exam key.'
      : `${paper.count - keyed} question(s) still need a correct option.`;
  }

  /* ---------------------------------------------------------------- preview */

  function headerOptions() {
    const p = db.profile;
    const paper = activePaper();
    const count = paper.count;
    const standard = `${count} questions · 4 options · fill one bubble per question`;
    return {
      title: p.name || 'OMR ANSWER SHEET',
      subtitle: p.address || (p.name ? standard : ''),
      line2: p.line || '',
      logo: p.logo || '',
      footer: p.note || '',
    };
  }

  function renderPreview() {
    const paper = activePaper();
    // One printed design: the count only changes the question list.
    const template = root.OMR_SHEETS.master;
    if (!template) return;
    el.stage.innerHTML = root.OMR_SHEETDESIGN.sheetHtml(template, headerOptions());
    fitPreview();
  }

  function fitPreview() {
    const page = el.stage.querySelector('.sheet-page');
    if (!page) return;
    const width = el.stage.clientWidth;
    if (!width) return;
    const scale = Math.min(1, width / page.offsetWidth);
    page.style.transform = 'scale(' + scale + ')';
    el.stage.style.height = Math.round(page.offsetHeight * scale) + 'px';
  }

  /* ------------------------------------------------------------------ print */

  function printPaper() {
    const paper = activePaper();
    const template = root.OMR_SHEETS.master;
    if (!template) {
      setStatus('The sheet design did not load.', true);
      return;
    }
    const done = root.OMR_SHEETDESIGN.printSheet(template, headerOptions());
    setStatus(done
      ? 'Opening the print dialog — choose “Save as PDF” to keep a copy on the phone.'
      : 'Printing is not available in this browser.', !done);
  }

  function pushKey() {
    const paper = activePaper();
    const app = root.OMR_APP;
    const exam = app && typeof app.exam === 'function' ? app.exam() : null;
    if (!exam) {
      setStatus('Open an exam on the Subjects tab first, then come back.', true);
      return;
    }
    if (exam.count !== paper.count) {
      setStatus(`That exam has ${exam.count} questions, this paper has ${paper.count}.`, true);
      return;
    }
    const answers = paper.questions.map((q) => q.key);
    if (answers.some((a) => a === null)) {
      setStatus('Mark the correct option for every question first.', true);
      return;
    }
    try {
      root.OMRStore.saveKey(exam.id, answers, new Array(paper.count).fill(1));
      if (typeof app.reloadKey === 'function') app.reloadKey();
      setStatus(`Answer key sent to “${exam.name}”.`, false);
    } catch (error) {
      setStatus((error && error.message) || 'The key could not be saved.', true);
    }
  }

  /* ------------------------------------------------------------- paper list */

  function paintPapers() {
    const paper = activePaper();
    el.pick.innerHTML = '';
    db.papers.forEach((p) => {
      const option = root.document.createElement('option');
      option.value = p.id;
      option.textContent = p.name + ' · ' + p.count + 'q';
      el.pick.appendChild(option);
    });
    el.pick.value = paper.id;
    el.name.value = paper.name;
    el.count.value = String(paper.count);
    renderQuestions();
    renderPreview();
    setSaved(true);
  }

  function switchCount(count) {
    const paper = activePaper();
    if (paper.count === count) return;
    paper.count = count;
    ensureQuestions(paper);
    paper.updatedAt = Date.now();
    renderQuestions();
    renderPreview();
    saveSoon('Sheet size set to ' + count + ' questions.');
  }

  /* ------------------------------------------------------------------ wiring */

  function init() {
    const panel = $('tab-paper');
    if (!panel) return;

    el.school = $('paperSchool');
    el.address = $('paperAddress');
    el.line = $('paperLine');
    el.note = $('paperNote');
    el.logoInput = $('paperLogo');
    el.logoBox = $('paperLogoBox');
    el.logoClear = $('paperLogoClear');
    el.name = $('paperName');
    el.count = $('paperCount');
    el.pick = $('paperPick');
    el.questions = $('paperQuestions');
    el.stage = $('paperStage');
    el.stats = $('paperStats');
    el.keysHint = $('paperKeysHint');
    el.message = $('paperMessage');
    el.saved = $('paperSaved');

    // The printed sheet is always the same 60-slot page; this list only picks
    // how many questions the paper asks.
    root.OMR_SHEETS.sizeChoices.forEach((count) => {
      const option = root.document.createElement('option');
      option.value = String(count);
      option.textContent = count + ' questions';
      el.count.appendChild(option);
    });

    fillHeaderForm();
    paintPapers();

    ['school', 'address', 'line', 'note'].forEach((field) => {
      el[field].addEventListener('input', () => {
        readHeaderForm();
        renderPreview();
        setSaved(false);
        saveSoon('Saved on this phone');
      });
    });

    el.logoInput.addEventListener('change', () => {
      const file = el.logoInput.files && el.logoInput.files[0];
      if (!file) return;
      shrinkLogo(file, (src) => {
        db.profile.logo = src;
        paintLogo();
        renderPreview();
        setSaved(false);
        saveSoon(src ? 'Logo saved on this phone.' : 'That image could not be used.');
      });
      el.logoInput.value = '';
    });

    el.logoClear.addEventListener('click', () => {
      db.profile.logo = '';
      paintLogo();
      renderPreview();
      setSaved(false);
      saveSoon('Logo removed.');
    });

    el.name.addEventListener('input', () => {
      const paper = activePaper();
      paper.name = el.name.value.trim() || 'Untitled paper';
      paper.updatedAt = Date.now();
      const option = el.pick.querySelector('option[value="' + paper.id + '"]');
      if (option) option.textContent = paper.name + ' · ' + paper.count + 'q';
      setSaved(false);
      saveSoon('Saved on this phone');
    });

    el.count.addEventListener('change', () => switchCount(Number(el.count.value) || 30));

    el.pick.addEventListener('change', () => {
      db.activeId = el.pick.value;
      saveSoon();
      paintPapers();
      setStatus('');
    });

    $('paperNew').addEventListener('click', () => {
      const paper = blankPaper(Number(el.count.value) || 30, 'Paper ' + (db.papers.length + 1));
      db.papers.push(paper);
      db.activeId = paper.id;
      saveSoon();
      paintPapers();
      setStatus('New paper started.');
    });

    $('paperDelete').addEventListener('click', () => {
      const paper = activePaper();
      if (db.papers.length === 1) {
        db.papers[0] = blankPaper(paper.count, paper.name);
        db.activeId = db.papers[0].id;
      } else {
        db.papers = db.papers.filter((p) => p.id !== paper.id);
        db.activeId = db.papers[0].id;
      }
      saveSoon();
      paintPapers();
      setStatus('Paper cleared.');
    });

    el.questions.addEventListener('input', (event) => {
      const target = event.target;
      const index = Number(target.dataset.i);
      const paper = activePaper();
      const question = paper.questions[index];
      if (!question) return;
      if (target.classList.contains('pq-text')) question.text = target.value;
      else if (target.classList.contains('pq-opt-input')) {
        question.options[Number(target.dataset.o)] = target.value;
      } else return;
      paper.updatedAt = Date.now();
      renderPreview();
      setSaved(false);
      saveSoon('Saved on this phone');
    });

    el.questions.addEventListener('click', (event) => {
      const button = event.target.closest('.pq-key');
      if (!button) return;
      const index = Number(button.dataset.i);
      const option = Number(button.dataset.o);
      const paper = activePaper();
      const question = paper.questions[index];
      if (!question) return;
      question.key = question.key === option ? null : option;
      const row = button.closest('.pq');
      row.classList.toggle('is-keyed', question.key !== null);
      row.querySelectorAll('.pq-key').forEach((key, i) => {
        key.classList.toggle('is-on', i === question.key);
      });
      paper.updatedAt = Date.now();
      paintStats();
      setSaved(false);
      saveSoon('Saved on this phone');
    });

    $('paperPrint').addEventListener('click', printPaper);
    $('paperPushKey').addEventListener('click', pushKey);
    $('paperFit').addEventListener('click', fitPreview);

    // The panel starts hidden, so the preview can only be measured once the tab
    // is actually on screen.
    root.document.addEventListener('click', (event) => {
      if (!event.target.closest('[data-tab="paper"], [data-go="paper"]')) return;
      setTimeout(fitPreview, 40);
    });

    root.addEventListener('resize', fitPreview);
  }

  if (root.document.readyState === 'loading') {
    root.document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  root.OMR_PAPER = {
    profile: () => db.profile,
    papers: () => db.papers,
    active: activePaper,
    render: () => { paintPapers(); },
  };
})(typeof window !== 'undefined' ? window : globalThis);
