/*
 * Headless test for the OMR pipeline.
 *
 * Renders a synthetic copy of each sheet design (paper, pink frame, bubbles),
 * fills a known answer key, then runs the real engine over it.  Also checks the
 * live fusion path and the projected (unwarped) reader.
 *
 * Run: node test/pipeline-test.js
 */
'use strict';

const path = require('path');

global.window = global;
require(path.join(__dirname, '..', 'assets', 'sheet-designs.js'));
require(path.join(__dirname, '..', 'assets', 'omr-engine.js'));

const DPI = 150;
const PINK = [200, 111, 136];
const BLACK = [25, 25, 28];
const PENCIL = [150, 152, 160];

let failures = 0;

function check(label, condition, detail) {
  if (condition) {
    console.log('  ok   ' + label);
  } else {
    failures++;
    console.log('  FAIL ' + label + (detail ? ' — ' + detail : ''));
  }
}

/* --------------------------------------------------------------- drawing */

function blankPage(template) {
  const width = Math.round(template.page_mm[0] / 25.4 * DPI);
  const height = Math.round(template.page_mm[1] / 25.4 * DPI);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = 255;
  }
  return { data, width, height, mm: (v) => v / 25.4 * DPI };
}

function setPixel(canvas, x, y, colour) {
  const px = Math.round(x);
  const py = Math.round(y);
  if (px < 0 || py < 0 || px >= canvas.width || py >= canvas.height) return;
  const i = (py * canvas.width + px) * 4;
  canvas.data[i] = colour[0];
  canvas.data[i + 1] = colour[1];
  canvas.data[i + 2] = colour[2];
  canvas.data[i + 3] = 255;
}

function strokeCircle(canvas, cx, cy, radius, colour) {
  const steps = Math.max(60, Math.round(radius * 10));
  for (let i = 0; i < steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    setPixel(canvas, cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius, colour);
  }
}

function fillCircle(canvas, cx, cy, radius, colour) {
  for (let y = Math.floor(cy - radius); y <= Math.ceil(cy + radius); y++) {
    for (let x = Math.floor(cx - radius); x <= Math.ceil(cx + radius); x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= radius * radius) setPixel(canvas, x, y, colour);
    }
  }
}

function strokeLine(canvas, x0, y0, x1, y1, colour, thickness) {
  const steps = Math.max(2, Math.round(Math.hypot(x1 - x0, y1 - y0)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = x0 + (x1 - x0) * t;
    const y = y0 + (y1 - y0) * t;
    for (let dy = -thickness; dy <= thickness; dy++) {
      for (let dx = -thickness; dx <= thickness; dx++) {
        setPixel(canvas, x + dx * 0.5, y + dy * 0.5, colour);
      }
    }
  }
}

function fillRectPx(canvas, x0, y0, x1, y1, colour) {
  for (let y = Math.round(y0); y <= Math.round(y1); y++) {
    for (let x = Math.round(x0); x <= Math.round(x1); x++) setPixel(canvas, x, y, colour);
  }
}

function renderSheet(template, answers, ink, ringColour) {
  const canvas = blankPage(template);
  const mm = canvas.mm;
  const frame = template.frame_mm;
  const [x0, y0, x1, y1] = frame.map(mm);
  const ring = ringColour || PINK;

  strokeLine(canvas, x0, y0, x1, y0, ring, 1);
  strokeLine(canvas, x1, y0, x1, y1, ring, 1);
  strokeLine(canvas, x1, y1, x0, y1, ring, 1);
  strokeLine(canvas, x0, y1, x0, y0, ring, 1);

  for (const [q, option, x, y, r] of template.bubbles) {
    const cx = mm(x);
    const cy = mm(y);
    const radius = mm(r);
    strokeCircle(canvas, cx, cy, radius, ring);
    if (answers[q - 1] === option) {
      fillCircle(canvas, cx, cy, radius * 0.82, ink || BLACK);
    }
  }

  // Numbered band above every question row and the pink info boxes beside the
  // table: the printed asymmetry the engine's orientation check reads.
  const rowTopsPx = [];
  let lastTop = null;
  for (const [q, option, x, y] of template.bubbles) {
    if (option !== 0) continue;
    if (lastTop === null || Math.abs(y - lastTop) > 1) { rowTopsPx.push(y); lastTop = y; }
  }
  for (const y of rowTopsPx) {
    // A thin numbered band above every question row: the synthetic test sheet
    // keeps its original style (the reader's band witness covers this area,
    // and the real production print was validated separately against the
    // actual scans in samples/q-paper_*.png).
    const bandY = mm(y - 7.3);
    if (bandY > y0 + 1.5 && bandY < y1 - 1.5) {
      strokeLine(canvas, x0 + 1.5, bandY, x1 - 1.5, bandY, ring, 1);
    }
  }
  const regions = template.regions || {};
  for (const key of Object.keys(regions)) {
    const box = regions[key];
    fillRectPx(canvas, mm(box.x_mm), mm(box.y_mm),
      mm(box.x_mm + box.w_mm), mm(box.y_mm + box.h_mm), ring);
  }
  return canvas;
}

function downscale(canvas, maxSide) {
  const scale = Math.min(1, maxSide / Math.max(canvas.width, canvas.height));
  if (scale === 1) return canvas;
  const width = Math.round(canvas.width * scale);
  const height = Math.round(canvas.height * scale);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(canvas.height - 1, Math.floor(y / scale));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(canvas.width - 1, Math.floor(x / scale));
      const from = (sy * canvas.width + sx) * 4;
      const to = (y * width + x) * 4;
      data[to] = canvas.data[from];
      data[to + 1] = canvas.data[from + 1];
      data[to + 2] = canvas.data[from + 2];
      data[to + 3] = 255;
    }
  }
  return { data, width, height };
}

/* ------------------------------------------------------------------ tests */

function keyFor(count) {
  // A deterministic pattern that uses all four options.
  const key = [];
  for (let i = 0; i < count; i++) key.push((i * 3 + 1) % 4);
  return key;
}

function testSheetSize(count) {
  const template = window.OMR_SHEETS.build(count);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(count);
  const page = renderSheet(template, key);

  console.log(`\n${count}-question sheet (${template.id})`);

  const result = engine.scanRGBA(page.data, page.width, page.height, key, 1);
  check('sheet located', result.ok, result.error || '');
  if (!result.ok) return;

  check('question count', result.grade.questionCount === count,
    String(result.grade.questionCount));
  check('every answer correct', result.grade.correct === count,
    `${result.grade.correct}/${count}`);
  check('no wrong/blank/review', result.grade.wrong === 0 && result.grade.blank === 0
    && result.grade.review === 0,
    `wrong=${result.grade.wrong} blank=${result.grade.blank} review=${result.grade.review}`);
  check('total equals maximum', result.grade.total === result.grade.maximum,
    `${result.grade.total}/${result.grade.maximum}`);

  // Wrong answers must be reported as wrong, not silently correct.
  const wrongKey = key.map(value => (value + 1) % 4);
  const wrong = engine.scanRGBA(page.data, page.width, page.height, wrongKey, 1);
  check('a different key scores zero', wrong.ok && wrong.grade.total === 0,
    wrong.ok ? String(wrong.grade.total) : wrong.error);

  // Blank sheet: nothing marked.
  const empty = renderSheet(template, new Array(count).fill(null));
  const blank = engine.scanRGBA(empty.data, empty.width, empty.height, key, 1);
  check('blank sheet scores zero', blank.ok && blank.grade.total === 0,
    blank.ok ? String(blank.grade.total) : blank.error);

  // Live path: no warp, sampling projected from the camera frame.
  const camera = downscale(page, 1280);
  const detection = engine.detectFrame(camera.data, camera.width, camera.height);
  check('camera frame detected', detection.mode !== 'none', detection.mode);
  if (detection.mode !== 'none') {
    const H = engine.homography(
      detection.mode === 'frame' ? engine.FRAME_DST : engine.PAGE_DST,
      detection.corners
    );
    const readings = engine.readProjected(camera.data, camera.width, camera.height, H);
    check('projected read returns every bubble',
      Boolean(readings) && readings.length === engine.BUBBLES.length,
      readings ? String(readings.length) : 'null');
    if (readings) {
      const marked = readings.filter(r => r.score >= 0.5).length;
      check('projected read finds every mark', marked === count, `${marked}/${count}`);
    }
  }
}

function testMarks() {
  console.log('\nper-question marks');
  const template = window.OMR_SHEETS.build(15);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(15);
  const marks = key.map((value, index) => (index < 5 ? 2 : 1));
  const page = renderSheet(template, key);
  const result = engine.scanRGBA(page.data, page.width, page.height, key, marks);
  check('maximum includes per-question marks', result.grade.maximum === 20,
    String(result.grade.maximum));
  check('total includes per-question marks', result.grade.total === 20,
    String(result.grade.total));

  const partial = engine.scanRGBA(page.data, page.width, page.height, key, marks);
  check('detail carries the mark', partial.grade.details[0].mark === 2,
    String(partial.grade.details[0].mark));
}

function testFusion() {
  console.log('\nlive fusion');
  const template = window.OMR_SHEETS.build(40);
  const engine = window.OMREngineFactory.createEngine(template);
  const fusion = engine.createFusion({ need: 5, agree: 3 });

  const clean = engine.BUBBLES.map(b => ({
    q: b.q,
    option: b.opt,
    score: b.opt === 0 ? 0.95 : 0.03,
    fill: b.opt === 0 ? 0.9 : 0.02,
    ring: 0,
    paper: 0,
    darkFrac: 1,
  }));

  for (let frame = 0; frame < 8; frame++) fusion.push(clean);
  check('settles after enough good frames', fusion.isSettled());
  const before = fusion.readings().filter(r => r.state === 'MARKED').length;
  check('every question marked once', before === 40, String(before));

  // Two blurred frames must not change any decision.
  const bad = engine.BUBBLES.map(b => ({
    q: b.q, option: b.opt, score: 0.02, fill: 0.02, ring: 0, paper: 0, darkFrac: 0,
  }));
  fusion.push(bad);
  fusion.push(bad);
  const after = fusion.readings().filter(r => r.state === 'MARKED').length;
  check('two bad frames change nothing', after === before, `${after} vs ${before}`);

  // A frame with the wrong bubble is also absorbed by hysteresis.
  const swapped = engine.BUBBLES.map(b => ({
    q: b.q, option: b.opt, score: b.opt === 3 ? 0.9 : 0.02, fill: 0.9, ring: 0, paper: 0, darkFrac: 1,
  }));
  fusion.push(swapped);
  fusion.push(swapped);
  const stillMarked = fusion.readings();
  const q1 = stillMarked.filter(r => r.q === 1 && r.state === 'MARKED').map(r => r.option);
  check('a swapped frame does not flip the answer', q1.length === 1 && q1[0] === 0,
    JSON.stringify(q1));

  fusion.reset();
  check('reset clears the buffer', fusion.frames === 0 && !fusion.isSettled());
}

/*
 * The live path end to end: a sheet that only fills part of the frame, marked
 * with light pencil, read from the answer area and fused over several frames.
 * This is the case that used to report zero marks: fusion compared every
 * bubble against a fixed threshold that a phone-sized mark never reaches.
 */
function testLiveCamera() {
  console.log('\nlive camera (answer area + fusion)');
  const template = window.OMR_SHEETS.build(25);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(25);
  const page = renderSheet(template, key, PENCIL);
  const camera = downscale(page, 640);

  const detection = engine.detectFrame(camera.data, camera.width, camera.height);
  check('small sheet detected', detection.mode !== 'none', detection.mode);
  if (detection.mode === 'none') return;

  const H = engine.homography(
    detection.mode === 'frame' ? engine.FRAME_DST : engine.PAGE_DST,
    detection.corners
  );
  const area = engine.readRegion(camera.data, camera.width, camera.height, H);
  check('answer-area read returns every bubble',
    Boolean(area) && area.readings.length === engine.BUBBLES.length,
    area ? String(area.readings.length) : 'null');
  if (!area) return;

  // A sheet at reading distance is measured straight from the frame (no crop);
  // a far one is cropped and enlarged, so compare that at camera scale.
  const cropShare = area.crop
    ? (area.crop.width * area.crop.height) / (area.crop.scale * area.crop.scale)
      / (camera.width * camera.height)
    : 0;
  check('only the answer area is read', cropShare < 0.7,
    area.crop
      ? area.crop.width + 'x' + area.crop.height + ' of ' + camera.width + 'x' + camera.height
      : 'read straight from the frame');

  const fusion = engine.createFusion({ need: 5, agree: 3 });
  for (let frame = 0; frame < 8; frame++) fusion.push(area.readings);
  check('fusion settles', fusion.isSettled());
  const fused = fusion.readings();
  const marked = fused.filter(r => r.state === 'MARKED').length;
  check('every mark survives the live path', marked === 25, String(marked));

  const read = {
    questions: engine.questionsFromReadings(fused),
    bubbles: fused,
    calibration: { threshold: 0, method: 'fusion' },
    diagnostics: {},
  };
  const grade = engine.gradeRead(read, key, 1);
  check('live scan is not zero', grade.total === 25 && grade.maximum === 25,
    grade.total + '/' + grade.maximum);

  const blankSheet = renderSheet(template, new Array(25).fill(null), PENCIL);
  const blankCam = downscale(blankSheet, 640);
  const blankDetection = engine.detectFrame(blankCam.data, blankCam.width, blankCam.height);
  const blankH = blankDetection.mode === 'none' ? null : engine.homography(
    blankDetection.mode === 'frame' ? engine.FRAME_DST : engine.PAGE_DST,
    blankDetection.corners
  );
  const blankArea = blankH
    ? engine.readRegion(blankCam.data, blankCam.width, blankCam.height, blankH)
    : null;
  check('blank sheet read through the answer area', Boolean(blankArea));
  if (!blankArea) return;

  const blankFusion = engine.createFusion({ need: 5, agree: 3 });
  for (let frame = 0; frame < 8; frame++) blankFusion.push(blankArea.readings);
  const blankFused = blankFusion.readings();
  const blankRead = {
    questions: engine.questionsFromReadings(blankFused),
    bubbles: blankFused,
    calibration: { threshold: 0, method: 'fusion' },
    diagnostics: {},
  };
  const blankGrade = engine.gradeRead(blankRead, key, 1);
  check('blank sheet through the live path scores zero', blankGrade.total === 0,
    String(blankGrade.total));
}

/*
 * Turns an image by a quarter turn (90°) or half a turn (180°) in plain pixels,
 * the way a phone's EXIF rotation would hand the photo over.
 */
function rotateQuarter(source) {
  const { data, width, height } = source;
  const outW = height;
  const outH = width;
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const nx = height - 1 - y;
      const ny = x;
      const from = (y * width + x) * 4;
      const to = (ny * outW + nx) * 4;
      out[to] = data[from];
      out[to + 1] = data[from + 1];
      out[to + 2] = data[from + 2];
      out[to + 3] = 255;
    }
  }
  return { data: out, width: outW, height: outH };
}

function rotatePixels(source, degrees) {
  const turns = ((degrees % 360) + 360) % 360;
  let image = source;
  for (let i = 0; i < turns / 90; i++) image = rotateQuarter(image);
  return image;
}

/*
 * Photo mode has to cope with a phone that hands the picture over rotated.  The
 * app reads the photo as it arrives and, when that finds no mark, again at 180,
 * 90 and 270 degrees.  Here every starting angle must end up at 25/25.
 */
function testRotatedPhoto() {
  console.log('\nrotated photo');
  const template = window.OMR_SHEETS.build(25);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(25);
  const page = downscale(renderSheet(template, key, PENCIL), 1400);

  const cases = [
    ['upright', page],
    ['upside down', rotatePixels(page, 180)],
    ['sideways', rotatePixels(page, 90)],
  ];

  for (const [label, image] of cases) {
    let bestTotal = 0;
    for (const degrees of [0, 180, 90, 270]) {
      const turned = degrees ? rotatePixels(image, degrees) : image;
      const result = engine.scanRGBA(turned.data, turned.width, turned.height, key, 1);
      const total = result.ok ? result.grade.total : 0;
      if (total > bestTotal) bestTotal = total;
    }
    check(label + ' photo still scores 25/25', bestTotal === 25, String(bestTotal));
  }
}

/*
 * A hand is never perfectly still: every third frame is a shaken one in which
 * the marks smear away and blank bubbles pick up ink.  The median over a window
 * of frames has to ignore them and keep every answer.
 */
function testShakyHand() {
  console.log('\nshaky hand');
  const template = window.OMR_SHEETS.build(40);
  const engine = window.OMREngineFactory.createEngine(template);
  const fusion = engine.createFusion({ need: 7, agree: 3, window: 9 });

  const clean = engine.BUBBLES.map(b => ({
    q: b.q,
    option: b.opt,
    score: b.opt === 0 ? 0.86 : 0.05,
    fill: b.opt === 0 ? 0.80 : 0.03,
    ring: 0,
    paper: 0,
    darkFrac: 1,
  }));
  const shaken = engine.BUBBLES.map(b => ({
    q: b.q,
    option: b.opt,
    score: 0.12,
    fill: 0.10,
    ring: 0,
    paper: 0,
    darkFrac: 0,
  }));

  for (let frame = 0; frame < 12; frame++) {
    fusion.push(frame % 3 === 2 ? shaken : clean);
  }
  const marked = fusion.readings().filter(r => r.state === 'MARKED').length;
  check('shaken frames are outvoted', marked === 40, String(marked));
  check('the sheet still settles', fusion.isSettled());
}

function eraseFrame(page, template) {
  const data = new Uint8ClampedArray(page.data);
  const width = page.width;
  const height = page.height;
  const mm = (v) => v / 25.4 * 150;
  const f = template.frame_mm.map(mm);
  const band = 6;
  const white = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const i = (y * width + x) * 4;
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = 255;
  };
  for (let x = Math.floor(f[0]) - band; x <= Math.ceil(f[2]) + band; x++) {
    for (let dy = -band; dy <= band; dy++) {
      white(x, Math.round(f[1]) + dy);
      white(x, Math.round(f[3]) + dy);
    }
  }
  for (let y = Math.floor(f[1]) - band; y <= Math.ceil(f[3]) + band; y++) {
    for (let dx = -band; dx <= band; dx++) {
      white(Math.round(f[0]) + dx, y);
      white(Math.round(f[2]) + dx, y);
    }
  }
  return { data, width, height };
}

function pasteUnder(background, image, offsetX, offsetY) {
  for (let y = 0; y < image.height; y++) {
    const dy = offsetY + y;
    if (dy < 0 || dy >= background.height) continue;
    for (let x = 0; x < image.width; x++) {
      const dx = offsetX + x;
      if (dx < 0 || dx >= background.width) continue;
      const from = (y * image.width + x) * 4;
      const to = (dy * background.width + dx) * 4;
      background.data[to] = image.data[from];
      background.data[to + 1] = image.data[from + 1];
      background.data[to + 2] = image.data[from + 2];
      background.data[to + 3] = 255;
    }
  }
  return background;
}

/*
 * A photo from the phone camera: the sheet lies on a dark desk, the pink border
 * is washed out by the light, and the sheet fills only part of the picture.  The
 * reader has to fall back to the paper outline and still find every mark.
 */
function testPhotoOnDesk() {
  console.log('\nphoto on a desk (no usable pink border)');
  const template = window.OMR_SHEETS.build(25);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(25);
  const page = eraseFrame(renderSheet(template, key, PENCIL), template);
  const sheet = downscale(page, 900);

  const desk = { width: 1400, height: 1050, data: new Uint8ClampedArray(1400 * 1050 * 4) };
  for (let i = 0; i < desk.data.length; i += 4) {
    desk.data[i] = 74;
    desk.data[i + 1] = 80;
    desk.data[i + 2] = 88;
    desk.data[i + 3] = 255;
  }
  pasteUnder(desk, sheet, 420, 62);

  const detection = engine.detectFrame(desk.data, desk.width, desk.height);
  check('paper outline is found', detection.mode !== 'none', detection.mode);
  if (detection.mode === 'none') return;

  const result = engine.scanRGBA(desk.data, desk.width, desk.height, key, 1);
  check('sheet on a desk still scores 25/25',
    result.ok && result.grade.total === 25,
    result.ok ? result.grade.total + '/' + result.grade.maximum : result.error);
}

/*
 * A frame whose geometry is off by one bubble row, as happens while the hand is
 * still moving.  Every bubble disagrees, so fusion has to throw the frame away
 * instead of letting it into the median.
 */
function testShakenGeometry() {
  console.log('\nshaken geometry');
  const template = window.OMR_SHEETS.build(40);
  const engine = window.OMREngineFactory.createEngine(template);
  const fusion = engine.createFusion({ need: 7, agree: 3, window: 9 });

  const make = (shift) => engine.BUBBLES.map(b => {
    const marked = (b.q + shift) % 4 === b.opt;
    return {
      q: b.q,
      option: b.opt,
      score: marked ? 0.88 : 0.04,
      fill: marked ? 0.82 : 0.03,
      ring: 0,
      paper: 0,
      darkFrac: 1,
    };
  });

  const clean = make(0);
  const shifted = make(1);

  for (let frame = 0; frame < 7; frame++) fusion.push(clean);
  check('clean frames settle', fusion.isSettled());
  for (let frame = 0; frame < 5; frame++) fusion.push(shifted);
  const marked = fusion.readings().filter(r => r.state === 'MARKED' && r.option === r.q % 4).length;
  check('misaligned frames are thrown away', marked === 40, String(marked));
}

function testAlignment() {
  console.log('\na crooked detection');
  const template = window.OMR_SHEETS.build(25);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = Array.from({ length: 25 }, (_, i) => i % 4);
  const page = renderSheet(template, key, PENCIL);
  const camera = downscale(page, 640);

  // The page outline is found about 1 mm off, which is a third of a bubble.
  const bias = Math.round(camera.width * 0.005);
  const corners = [
    [bias, bias],
    [camera.width - 1 + bias, bias],
    [camera.width - 1 + bias, camera.height - 1 + bias],
    [bias, camera.height - 1 + bias],
  ];
  const result = engine.scanRGBA(
    camera.data, camera.width, camera.height, key, new Array(25).fill(1),
    { mode: 'page', corners, confidence: 0.5, orientation: 0 }
  );
  check('the sheet is still read', result.ok === true, result.ok ? '' : result.error);
  if (!result.ok) return;
  const diag = result.read.diagnostics || {};
  check('every answer survives the offset', result.grade.correct === 25,
    result.grade.correct + '/25 align=' + result.align + ' marked=' + diag.markedBubbles
    + ' ' + result.read.calibration.method + ' thr='
    + Math.round((result.read.calibration.threshold || 0) * 100) / 100
    + ' max=' + Math.round((diag.scoreMax || 0) * 100) / 100);
  check('the printed rings line up after the correction', result.ringFit > 0.25,
    'fit=' + Math.round((result.ringFit || 0) * 100) / 100 + ' align=' + result.align);
}

/*
 * A sheet that came out of the printer a few percent small — "fit to page", a
 * photocopier, or a school that reprinted it.  The grid is in the right place
 * but the wrong size, so the offset alone cannot save it; the size has to be
 * searched too.
 */
function scaleOntoPage(page, factor) {
  const width = page.width;
  const height = page.height;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = 255;
    out[i + 1] = 255;
    out[i + 2] = 255;
    out[i + 3] = 255;
  }
  const cx = width / 2;
  const cy = height / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = Math.round(cx + (x - cx) / factor);
      const sy = Math.round(cy + (y - cy) / factor);
      if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
      const from = (sy * width + sx) * 4;
      const to = (y * width + x) * 4;
      out[to] = page.data[from];
      out[to + 1] = page.data[from + 1];
      out[to + 2] = page.data[from + 2];
    }
  }
  return { data: out, width, height };
}

function testScaledPrint() {
  console.log('\nsheet printed a few percent small');
  const template = window.OMR_SHEETS.build(30);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(30);
  const page = renderSheet(template, key, BLACK);

  for (const factor of [0.96, 1.04]) {
    const scaled = scaleOntoPage(page, factor);
    const result = engine.scanRGBA(scaled.data, scaled.width, scaled.height, key, 1);
    check('printed at ' + Math.round(factor * 100) + '% reads 30/30',
      result.ok && result.grade.total === 30,
      result.ok
        ? result.grade.total + '/30 fit=' + Math.round((result.ringFit || 0) * 100) / 100
          + ' scale=' + (result.align[2] || 1)
        : result.error);
  }
}

/*
 * A sheet printed on a plain laser printer: no pink anywhere, rings in grey.
 * The reader must fall back to the paper outline and still line up.
 */
function testGreyPrint() {
  console.log('\nblack-and-white printed sheet');
  const template = window.OMR_SHEETS.build(30);
  const engine = window.OMREngineFactory.createEngine(template);
  const key = keyFor(30);
  const sheet = downscale(renderSheet(template, key, BLACK, [72, 72, 76]), 900);

  const desk = { width: 1400, height: 1050, data: new Uint8ClampedArray(1400 * 1050 * 4) };
  for (let i = 0; i < desk.data.length; i += 4) {
    desk.data[i] = 74;
    desk.data[i + 1] = 80;
    desk.data[i + 2] = 88;
    desk.data[i + 3] = 255;
  }
  pasteUnder(desk, sheet, 420, 62);

  const result = engine.scanRGBA(desk.data, desk.width, desk.height, key, 1);
  check('a grey sheet on a desk is found', result.ok, result.ok ? '' : result.error);
  if (!result.ok) return;
  check('a grey sheet scores 30/30', result.grade.total === 30,
    result.grade.total + '/30 fit=' + Math.round((result.ringFit || 0) * 100) / 100);
}

function testSharedDesign() {
  console.log('\none sheet design for every question count');
  const design = window.OMR_SHEETDESIGN;
  const counts = window.OMR_SHEETS.sizeChoices;
  const base = window.OMR_SHEETS.build(15);
  const baseLayout = design.layoutOf(base);
  const baseFrame = base.frame_mm;
  const near = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 0.05 : tol);

  check('the eight printed formats are offered', counts.length === 8, counts.join(' '));
  counts.forEach((c) => check(c + ': has a sheet', Boolean(window.OMR_SHEETS.build(c))));

  check('15 questions per row', baseLayout.perRow === 15, String(baseLayout.perRow));
  check('four options 9.0 mm apart', near(baseLayout.optionPitch, 9.0, 0.06),
    String(baseLayout.optionPitch));
  check('bubble size matches the paper sheet', near(baseLayout.radius, 3.56),
    String(baseLayout.radius));

  for (const count of counts) {
    const template = window.OMR_SHEETS.build(count);
    const L = design.layoutOf(template);
    const frame = template.frame_mm;
    const rows = Math.ceil(count / 15);

    check(count + ': ' + rows + ' row(s) of at most 15', L.rows === rows, String(L.rows));
    check(count + ': four bubbles per question', template.bubbles.length === count * 4,
      String(template.bubbles.length));

    let sameColumns = L.colX.length === baseLayout.colX.length;
    for (let i = 0; i < L.colX.length && sameColumns; i++) {
      sameColumns = near(L.colX[i], baseLayout.colX[i]);
    }
    check(count + ': same 15 columns', sameColumns);
    check(count + ': same frame width', near(frame[2] - frame[0], baseFrame[2] - baseFrame[0]),
      String(Math.round((frame[2] - frame[0]) * 100) / 100));
    check(count + ': same frame top', near(frame[1], baseFrame[1]), String(frame[1]));

    if (rows > 1) {
      const pitch = (L.rowY[rows - 1] - L.rowY[0]) / (rows - 1);
      // The printed v3 page spaces its four rows between 43.3 and 43.6 mm.
      check(count + ': a new row every ~43.45 mm', pitch > 43.25 && pitch < 43.65,
        String(pitch));
    }
  }

  const tallest = window.OMR_SHEETS.master.frame_mm;
  check('only the answer area grows', tallest[3] > baseFrame[3]
    && near(tallest[0], baseFrame[0]) && near(tallest[2], baseFrame[2]),
    '15:' + baseFrame[3] + ' 60:' + tallest[3]);
  check('the tallest sheet still fits A4', tallest[3] < 287, String(tallest[3]));
}

function testDesignGeometry() {
  console.log('\nsheet design geometry');
  for (const count of window.OMR_SHEETS.sizeChoices) {
    const template = window.OMR_SHEETS.build(count);
    const total = template.question_count;
    const bubbles = template.bubbles.length;
    check(`${count}: ${total * 4} bubbles`, bubbles === total * 4, String(bubbles));

    const [fx0, fy0, fx1, fy1] = template.frame_mm;
    const outside = template.bubbles.filter(([, , x, y, r]) =>
      x - r < fx0 || x + r > fx1 || y - r < fy0 || y + r > fy1);
    check(`${count}: all bubbles inside the frame`, outside.length === 0,
      String(outside.length) + ' outside');

    const inPage = template.bubbles.filter(([, , x, y, r]) =>
      x - r < 0 || y - r < 0 || x + r > template.page_mm[0] || y + r > template.page_mm[1]);
    check(`${count}: all bubbles on the page`, inPage.length === 0,
      String(inPage.length) + ' off page');
  }
}

console.log('OMR pipeline test');
testDesignGeometry();
testSharedDesign();
testAlignment();
for (const count of window.OMR_SHEETS.sizeChoices) testSheetSize(count);
testMarks();
testFusion();
testLiveCamera();
testRotatedPhoto();
testShakyHand();
testShakenGeometry();
testPhotoOnDesk();
testScaledPrint();
testGreyPrint();

console.log('\n' + (failures === 0 ? 'all checks passed' : failures + ' check(s) failed'));
process.exit(failures === 0 ? 0 : 1);
