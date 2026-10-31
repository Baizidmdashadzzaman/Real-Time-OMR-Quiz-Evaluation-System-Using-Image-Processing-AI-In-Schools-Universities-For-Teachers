(function () {
  'use strict';

  const OPTIONS = [
    { label: 'ক', digit: '1' },
    { label: 'খ', digit: '2' },
    { label: 'গ', digit: '3' },
    { label: 'ঘ', digit: '4' },
  ];
  const QUESTION_COUNT = 30;
  const key = new Array(QUESTION_COUNT).fill(null);
  const engine = window.OMREngine;
  const template = window.OMR_TEMPLATE;
  const mmToPx = (mm) => mm / 25.4 * engine.DPI;
  let lastResult = null;

  const $ = (id) => document.getElementById(id);
  const keyGrid = $('keyGrid');
  const keyCountEl = $('keyCount');
  const keyProgressEl = $('keyProgress');
  const clearKeyEl = $('clearKey');
  const bulkInput = $('bulkInput');
  const bulkApply = $('bulkApply');
  const bulkMessage = $('bulkMessage');
  const goScan = $('goScan');
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
  const breakdownEl = $('breakdown');
  const scanModeEl = $('scanMode');
  const scanAgain = $('scanAgain');
  const backToKey = $('backToKey');
  const startLive = $('startLive');
  const stopLive = $('stopLive');
  const toggleVoice = $('toggleVoice');
  const liveKeyState = $('liveKeyState');
  const arStage = $('arStage');
  const arVideo = $('arVideo');
  const arCanvas = $('arCanvas');
  const arStatus = $('arStatus');
  const arHudMark = $('arHudMark');
  const arHudTotal = $('arHudTotal');
  const arPlaceholder = $('arPlaceholder');
  const liveError = $('liveError');
  const liveStats = $('liveStats');
  const liveCorrect = $('liveCorrect');
  const liveWrong = $('liveWrong');
  const liveBlank = $('liveBlank');
  const liveMulti = $('liveMulti');
  const liveReview = $('liveReview');

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  /* ------------------------------------------------------------ live AR state */

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
    voice: true,
    armed: true,
    result: null,
    stream: null,
    timer: null,
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

  function buildKeyGrid() {
    const frag = document.createDocumentFragment();
    for (let q = 1; q <= QUESTION_COUNT; q++) {
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
        button.innerHTML = `<span class="bn">${opt.label}</span><span class="digit">${opt.digit}</span>`;
        options.appendChild(button);
      });

      row.appendChild(number);
      row.appendChild(options);
      frag.appendChild(row);
    }
    keyGrid.appendChild(frag);
  }

  function countSet() {
    return key.reduce((n, value) => n + (Number.isInteger(value) ? 1 : 0), 0);
  }

  function updateKeyUI() {
    const count = countSet();
    keyCountEl.textContent = `${count} / ${QUESTION_COUNT}`;
    keyProgressEl.style.width = `${(count / QUESTION_COUNT) * 100}%`;
    clearKeyEl.disabled = count === 0;

    keyGrid.querySelectorAll('.q-row').forEach(row => {
      const q = Number(row.dataset.q) - 1;
      row.classList.toggle('is-set', Number.isInteger(key[q]));
      row.querySelectorAll('.opt').forEach(button => {
        const opt = Number(button.dataset.opt);
        const selected = key[q] === opt;
        button.classList.toggle('is-selected', selected);
        button.setAttribute('aria-pressed', selected ? 'true' : 'false');
      });
    });

    const ready = count === QUESTION_COUNT;
    takePhoto.disabled = !ready;
    choosePhoto.disabled = !ready;
    startLive.disabled = !ready;
    if (ready) {
      scanKeyState.textContent = 'Key ready: 30 answers · 1 mark each.';
      liveKeyState.textContent = 'Key ready: start the camera and hold up a sheet.';
    } else {
      scanKeyState.textContent = `Set all 30 answers first (${count}/30).`;
      liveKeyState.textContent = `Set all 30 answers first (${count}/30).`;
    }
  }

  function setKey(qIndex, option) {
    key[qIndex] = key[qIndex] === option ? null : option;
    updateKeyUI();
  }

  function activateTab(name) {
    document.querySelectorAll('.tab').forEach(tab => {
      tab.classList.toggle('is-active', tab.dataset.tab === name);
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
      panel.classList.toggle('is-hidden', panel.id !== `tab-${name}`);
    });
    if (name !== 'live' && live.running) stopLiveCamera();
    if (name === 'scan' || name === 'live') updateKeyUI();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function parseBulk(text) {
    let raw = text.trim();
    if (!raw) return null;
    if (/^[1-4]{30}$/.test(raw)) return raw.split('').map(v => Number(v) - 1);
    const tokens = raw.split(/[\s,;،]+/).filter(Boolean);
    if (tokens.length !== QUESTION_COUNT) return null;
    const map = {
      '1': 0, '2': 1, '3': 2, '4': 3,
      '১': 0, '২': 1, '৩': 2, '৪': 3,
      'ক': 0, 'খ': 1, 'গ': 2, 'ঘ': 3,
    };
    const out = tokens.map(token => map[token]);
    return out.some(v => v === undefined) ? null : out;
  }

  function setBulkMessage(text, ok) {
    bulkMessage.textContent = text;
    bulkMessage.classList.toggle('ok', Boolean(ok));
  }

  keyGrid.addEventListener('click', (event) => {
    const button = event.target.closest('.opt');
    if (!button) return;
    setKey(Number(button.dataset.q) - 1, Number(button.dataset.opt));
  });

  bulkApply.addEventListener('click', () => {
    const parsed = parseBulk(bulkInput.value);
    if (!parsed) {
      setBulkMessage('Enter exactly 30 answers, each 1–4 (or ক–ঘ).');
      bulkInput.focus();
      return;
    }
    for (let i = 0; i < QUESTION_COUNT; i++) key[i] = parsed[i];
    bulkInput.value = '';
    setBulkMessage('30 answers applied.', true);
    updateKeyUI();
  });

  bulkInput.addEventListener('input', () => setBulkMessage('', false));
  bulkInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') bulkApply.click();
  });

  clearKeyEl.addEventListener('click', () => {
    if (!window.confirm('Clear all 30 answers?')) return;
    key.fill(null);
    setBulkMessage('', false);
    updateKeyUI();
  });

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => activateTab(tab.dataset.tab));
  });

  goScan.addEventListener('click', () => {
    activateTab('scan');
    document.querySelector('.scan-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  takePhoto.addEventListener('click', () => {
    if (countSet() !== QUESTION_COUNT) {
      activateTab('key');
      setBulkMessage('Set all 30 answers first.', false);
      return;
    }
    cameraInput.click();
  });

  choosePhoto.addEventListener('click', () => {
    if (countSet() !== QUESTION_COUNT) {
      activateTab('key');
      setBulkMessage('Set all 30 answers first.', false);
      return;
    }
    galleryInput.click();
  });

  cameraInput.addEventListener('change', () => handleFile(cameraInput.files && cameraInput.files[0], cameraInput));
  galleryInput.addEventListener('change', () => handleFile(galleryInput.files && galleryInput.files[0], galleryInput));

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

  function imageToData(img) {
    const MAX_SIDE = 1800;
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const width = Math.max(1, Math.round(img.naturalWidth * scale));
    const height = Math.max(1, Math.round(img.naturalHeight * scale));
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
  }

  async function handleFile(file, input) {
    if (!file) {
      if (input) input.value = '';
      return;
    }
    if (countSet() !== QUESTION_COUNT) {
      if (input) input.value = '';
      activateTab('key');
      setBulkMessage('Set all 30 answers first.', false);
      return;
    }

    try {
      showProcessing('Reading photo…', 'Preparing the image.');
      await sleep(40);
      const img = await loadImage(file);
      const imageData = imageToData(img);

      showProcessing('Finding the sheet…', 'Looking for the pink alignment frame.');
      await sleep(40);
      const result = window.OMREngine.scanRGBA(
        imageData.data,
        imageData.width,
        imageData.height,
        key.slice(),
        1
      );
      if (!result.ok) throw new Error(result.error);

      showProcessing('Grading…', 'Measuring the marked bubbles.');
      await sleep(30);
      renderResult(result);
    } catch (error) {
      showError(error && error.message ? error.message : 'Something went wrong while reading the sheet.');
    } finally {
      if (input) input.value = '';
    }
  }

  function answerLabel(index) {
    if (!Number.isInteger(index)) return '—';
    const opt = OPTIONS[index];
    return `${opt.label} (${opt.digit})`;
  }

  function renderResult(result) {
    lastResult = result;
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
    resultVerdict.textContent = `${verdict} · ${grade.correct} of ${QUESTION_COUNT} correct`;

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
    renderBreakdown(grade);

    resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderBreakdown(grade) {
    breakdownEl.innerHTML = grade.details.map(detail => {
      const selected = detail.selected.length
        ? detail.selected.map(answerLabel).join(', ')
        : '—';
      const keyText = detail.answer === null ? '—' : answerLabel(detail.answer);
      return `
        <div class="q-line">
          <span class="q">Q${String(detail.q).padStart(2, '0')}</span>
          <span class="answers">
            <span class="answer-pill">${selected}</span>
            <span class="answer-pill key">${keyText}</span>
          </span>
          <span class="badge ${detail.outcome}">${detail.outcome.replace('_', ' ')}</span>
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
    const engine = window.OMREngine;
    const template = window.OMR_TEMPLATE;
    const mmToPx = (mm) => mm / 25.4 * engine.DPI;
    const cropTop = Math.max(0, Math.round(mmToPx(44)));
    const cropBottom = Math.min(result.warp.height, Math.round(mmToPx(140)));
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
    const radius = mmToPx(2.979);
    const line = Math.max(2, Math.round(radius * 0.18));

    result.grade.details.forEach(detail => {
      const selected = detail.selected;
      if (detail.outcome === 'CORRECT') {
        selected.forEach(opt => {
          const b = byBubble.get(`${detail.q}:${opt}`);
          if (b) drawCircle(ctx, b.x, b.y - cropTop, radius * 1.12, color.CORRECT, line);
        });
      } else if (detail.outcome === 'WRONG' || detail.outcome === 'MULTI') {
        selected.forEach(opt => {
          const b = byBubble.get(`${detail.q}:${opt}`);
          if (b) drawCircle(ctx, b.x, b.y - cropTop, radius * 1.12,
            detail.outcome === 'MULTI' ? color.MULTI : color.WRONG, line);
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

    if (template.regions && template.regions.total_marks) {
      const r = template.regions.total_marks;
      const x = mmToPx(r.x_mm + r.w_mm / 2);
      const y = mmToPx(r.y_mm + r.h_mm / 2) - cropTop;
      ctx.fillStyle = '#c0392b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = `800 ${Math.round(mmToPx(10))}px system-ui, sans-serif`;
      ctx.fillText(String(result.grade.total), x, y);
    }
  }

  function resetResultView() {
    lastResult = null;
    processing.classList.add('is-hidden');
    errorBox.classList.add('is-hidden');
    resultEl.classList.add('is-hidden');
  }

  retryButton.addEventListener('click', () => {
    resetResultView();
    takePhoto.click();
  });

  scanAgain.addEventListener('click', () => {
    resetResultView();
    document.querySelector('.scan-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  backToKey.addEventListener('click', () => activateTab('key'));

  /* -------------------------------------------------------------- live AR */

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
      // Browser fallback (used during development): getUserMedia asks directly.
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
    arHudMark.querySelector('em').textContent = `/${grade.maximum}`;
    arHudMark.classList.remove('is-hidden');
  }

  function clearLiveResult() {
    live.result = null;
    live.lastResultQuad = null;
    live.shiftSince = 0;
    renderLiveStats(null);
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
    if (!detection || detection.mode === 'none') return;

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

    // Sheet outline, so the teacher can see what the engine is tracking.
    const outline = detection.corners.map(toStage);
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

    const dst = detection.mode === 'frame' ? engine.FRAME_DST : engine.PAGE_DST;
    const H = engine.homography(dst, detection.corners);
    if (!H) return;
    const project = (x, y) => {
      const point = engine.applyH(H, x, y);
      return [point[0] * stageScale + offsetX, point[1] * stageScale + offsetY];
    };

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

    const region = template.regions && template.regions.total_marks;
    if (region) {
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
  }

  function scheduleLive(delay) {
    if (!live.running) return;
    if (live.timer) clearTimeout(live.timer);
    live.timer = window.setTimeout(liveLoop, delay);
  }

  async function liveLoop() {
    live.timer = null;
    if (!live.running || live.busy) return;

    const now = performance.now();
    if (now < live.retryAt) {
      scheduleLive(live.retryAt - now + 40);
      return;
    }

    const image = grabLiveFrame();
    if (!image) {
      setLiveStatus('Waiting for the camera…');
      scheduleLive(250);
      return;
    }
    live.procW = image.width;
    live.procH = image.height;

    let detection = null;
    try {
      detection = engine.detectFrame(image.data, image.width, image.height);
    } catch (error) {
      setLiveStatus('Reader error — try again', 'err');
      scheduleLive(500);
      return;
    }
    if (!live.running) return;

    if (!detection || detection.mode === 'none') {
      live.lastQuad = null;
      live.lastDetection = null;
      live.stableSince = 0;
      live.shiftSince = 0;
      if (!live.absenceSince) live.absenceSince = now;
      if (live.result && now - live.absenceSince > 650) {
        clearLiveResult();
        live.armed = true;
        setLiveStatus('Ready for the next sheet');
      } else if (!live.result) {
        setLiveStatus('Show the sheet');
      }
      drawLiveOverlay(null, null, image.width, image.height);
      scheduleLive(240);
      return;
    }

    live.absenceSince = 0;
    live.lastDetection = detection;
    const normalized = normalizeCorners(detection.corners, image.width, image.height);
    const stable = live.lastQuad && maxCornerDelta(normalized, live.lastQuad) < 0.035;
    if (!stable) live.stableSince = now;
    const stableFor = now - live.stableSince;

    if (live.result && live.lastResultQuad &&
        maxCornerDelta(normalized, live.lastResultQuad) > 0.07) {
      if (!live.shiftSince) live.shiftSince = now;
      if (now - live.shiftSince > 280) {
        clearLiveResult();
        live.armed = true;
        live.shiftSince = 0;
        setLiveStatus('New sheet — hold still');
      }
    } else if (live.result) {
      live.shiftSince = 0;
    }

    if (!live.result && live.armed) {
      if (stableFor < 320) {
        setLiveStatus('Hold still…');
      } else if (now - live.lastScanAt > 650) {
        setLiveStatus('Reading sheet…');
        live.busy = true;
        await sleep(30);
        if (!live.running) {
          live.busy = false;
          return;
        }

        let result = null;
        try {
          result = engine.scanRGBA(
            image.data,
            image.width,
            image.height,
            key.slice(),
            1,
            detection
          );
        } catch (error) {
          result = null;
        }
        if (!live.running) {
          live.busy = false;
          return;
        }
        live.busy = false;

        if (result && result.ok) {
          live.result = result;
          live.armed = false;
          live.lastResultQuad = normalized;
          live.lastScanAt = performance.now();
          live.shiftSince = 0;
          renderLiveStats(result.grade);
          drawLiveOverlay(result, detection, image.width, image.height);
          setLiveStatus(`Mark ${result.grade.total} / ${result.grade.maximum}`, 'ok');
          speakPhrase(`Mark ${result.grade.total} out of ${result.grade.maximum}.`);
        } else {
          setLiveStatus('Could not read — hold the whole pink frame', 'warn');
          live.retryAt = performance.now() + 1200;
        }
      } else {
        setLiveStatus('Hold still…');
      }
    } else if (live.result) {
      setLiveStatus(
        `Mark ${live.result.grade.total} / ${live.result.grade.maximum} · show next sheet`,
        'ok'
      );
    }

    live.lastQuad = normalized;
    drawLiveOverlay(live.result, detection, image.width, image.height);
    scheduleLive(240);
  }

  async function startLiveCamera() {
    if (countSet() !== QUESTION_COUNT) {
      activateTab('key');
      setBulkMessage('Set all 30 answers first.', false);
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
            width: { ideal: 1280 },
            height: { ideal: 720 },
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
      live.armed = true;
      live.result = null;
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
        startLive.disabled = countSet() !== QUESTION_COUNT;
      }
    }
  }

  function stopLiveCamera() {
    live.startToken++;
    live.running = false;
    live.busy = false;
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
    startLive.disabled = countSet() !== QUESTION_COUNT;
    arStage.style.aspectRatio = '';
    setLiveStatus('Point the camera at a sheet');
  }

  startLive.addEventListener('click', () => {
    void startLiveCamera();
  });

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

  buildKeyGrid();
  updateKeyUI();
  updateVoiceUI();
  setLiveStatus('Point the camera at a sheet');
  void lastResult;
})();
