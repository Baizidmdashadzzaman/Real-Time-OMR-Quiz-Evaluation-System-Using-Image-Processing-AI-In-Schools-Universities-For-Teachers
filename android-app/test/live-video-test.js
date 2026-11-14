/*
 * Virtual video test for live camera scanning.
 *
 * Instead of one still picture per case, this renders a short video: the sheet
 * is photographed frame after frame through a moving quad (hand sway, shake, a
 * bump), with the degradations a phone camera produces along the way (motion
 * blur, defocus, noise, uneven light, a sheet drifting in and out of frame).
 *
 * The frames are pushed through the same path the app's live screen uses:
 * detect the sheet, crop the answer area, fused over several frames.  The
 * harness then reports how long the sheet took to settle, whether the settled
 * answer is right, how many frames had to be dropped, and how expensive a frame
 * is -- the three things a teacher actually feels in the classroom.
 *
 * Run: node test/live-video-test.js
 */
'use strict';

const path = require('path');

global.window = global;
require(path.join(__dirname, '..', 'assets', 'sheet-designs.js'));
require(path.join(__dirname, '..', 'assets', 'omr-engine.js'));

const sim = require('./sim.js');

const W = 960;
const H = 720;
const FPS = 30;
const TEMPLATE = window.OMR_SHEETS.build(30);
const ENGINE = window.OMREngineFactory.createEngine(TEMPLATE);
const KEY = sim.keyFor(30);
const TOTAL = 30;
const PAGE = sim.renderSheet(TEMPLATE, KEY);

/* --------------------------------------------------------------- the video */

function makeRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Corner path of a hand-held sheet over `frames` frames.  `sway` is the slow
 * drift of a hand, `shake` the nervous tremble on top of it, and `bumps` adds
 * the occasional knock that makes a paper jump.
 */
function handPath(base, frames, options) {
  const o = options || {};
  const sway = o.sway === undefined ? 14 : o.sway;
  const shake = o.shake === undefined ? 3 : o.shake;
  const bumps = o.bumps === undefined ? 1 : o.bumps;
  const rnd = makeRandom(o.seed || 5);
  const phase = [rnd() * 6.28, rnd() * 6.28, rnd() * 6.28, rnd() * 6.28];
  const bumpAt = new Map();
  for (let i = 0; i < bumps; i++) {
    bumpAt.set(6 + Math.floor(rnd() * (frames - 10)), (rnd() - 0.5) * 40);
  }
  const quads = [];
  for (let f = 0; f < frames; f++) {
    const t = f / Math.max(1, frames - 1);
    const wobbleX = Math.sin(t * 6.0 + phase[0]) * sway + Math.sin(t * 14.0 + phase[1]) * sway * 0.3;
    const wobbleY = Math.sin(t * 5.0 + phase[2]) * sway * 0.5 + Math.sin(t * 11.0 + phase[3]) * sway * 0.2;
    const bump = bumpAt.has(f) ? bumpAt.get(f) : 0;
    const grow = o.grow ? 1 + o.grow * (t - 0.5) : 1;
    const cx = W / 2, cy = H / 2;
    quads.push(base.map(([x, y]) => [
      cx + (x - cx) * grow + wobbleX + bump + (rnd() - 0.5) * shake * 2,
      cy + (y - cy) * grow + wobbleY + (rnd() - 0.5) * shake * 2,
    ]));
  }
  return quads;
}

/** Photograph one frame and degrade it the way a phone camera would. */
function frame(engine, quad, index, options) {
  const o = options || {};
  const rnd = makeRandom(1000 + index * 7);
  const image = sim.photograph(engine, PAGE, quad, W, H, o.background || sim.DESK);
  if (o.shadow !== false) {
    // Light falls off across the page and drifts a little frame to frame.
    const drift = 0.04 * Math.sin(index * 0.5);
    sim.shadowAcross(image, 0.94 - drift, 0.98 + drift);
  }
  if (o.expose) sim.brighten(image, o.expose, 0);
  const blurPasses = o.blur === undefined ? (rnd() < 0.35 ? 1 : 0) : o.blur;
  if (blurPasses) sim.blur(image, blurPasses);
  sim.addNoise(image, o.noise === undefined ? 5 : o.noise, 11 + index);
  return image;
}

/* ------------------------------------------------------------- live reader */

/**
 * A faithful copy of the live screen's loop, minus the DOM: follow the sheet,
 * require it to be steady for a few frames, then read the settled frame with
 * the full photo engine (the same one the upload screen uses).
 */
function maxCornerDelta(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]));
  }
  return worst;
}

function runVideo(quads, options) {
  const o = options || {};
  const frameMs = 1000 / FPS;
  const steadyNeed = o.steadyNeed === undefined ? 3 : o.steadyNeed;
  const steadyLimit = o.steadyLimit === undefined ? 0.015 : o.steadyLimit;
  let detection = null;
  let lastDetection = null;
  let lastSeenAt = -Infinity;
  let previous = null;
  let steady = 0;
  let detected = 0;
  let missed = 0;
  let readFrame = -1;
  let readMs = 0;
  let correct = 0;
  let wrong = 0;
  let blank = 0;
  let ok = false;

  for (let i = 0; i < quads.length; i++) {
    const now = i * frameMs;
    const image = frame(ENGINE, quads[i], i, o.degrade);

    detection = ENGINE.detectFrame(image.data, image.width, image.height);
    if (!detection || detection.mode === 'none' || !detection.corners) {
      // Same grace as the app: a single unreadable frame (blur, glare) must not
      // throw the sheet away, and must not reset the steady counter either.
      if (lastDetection && now - lastSeenAt < 1200) detection = lastDetection;
      else { missed++; steady = 0; previous = null; continue; }
    }
    lastDetection = detection;
    lastSeenAt = now;
    detected++;

    const normalized = detection.corners.map(p => [p[0] / image.width, p[1] / image.height]);
    if (previous) {
      if (maxCornerDelta(normalized, previous) <= steadyLimit) steady++;
      else steady = 0;
    }
    previous = normalized;

    if (steady < steadyNeed) continue;

    const t0 = process.hrtime.bigint();
    const result = ENGINE.scanRGBA(image.data, image.width, image.height, KEY, 1);
    readMs += Number(process.hrtime.bigint() - t0) / 1e6;
    if (result && result.ok) {
      ok = true;
      readFrame = i;
      correct = result.grade.correct;
      wrong = result.grade.wrong;
      blank = result.grade.blank;
      break;
    }
    // Not readable yet: wait for the hand to settle again and retry.
    steady = 0;
  }

  return {
    settled: ok,
    settleFrame: readFrame,
    correct, wrong, blank,
    detected, missed,
    readMs: readFrame >= 0 ? readMs : 0,
  };
}

/* ------------------------------------------------------------------- cases */

const base = () => [[246, 30], [713, 30], [713, 690], [246, 690]];
const FRAMES = 45;

const cases = [];

function addCase(name, quads, options) {
  const result = runVideo(quads, options);
  cases.push({ name, result });
}

addCase('steady hand', handPath(base(), FRAMES, { sway: 4, shake: 2, bumps: 0 }), {});
addCase('gentle sway', handPath(base(), FRAMES, { sway: 12, shake: 3 }), {});
addCase('shaky hand', handPath(base(), FRAMES, { sway: 22, shake: 7 }), {});
addCase('shaky + blurred frames', handPath(base(), FRAMES, { sway: 20, shake: 6 }), {
  degrade: { blur: undefined },
});
addCase('dim room, drifting shadow', handPath(base(), FRAMES, { sway: 10, shake: 4 }), {
  degrade: { expose: 0.6, blur: 1 },
});
addCase('tilted 9 degrees', handPath(sim.rotateQuad(base(), 9), FRAMES, { sway: 10, shake: 4 }), {});
addCase('hand coming closer', handPath(base(), FRAMES, { sway: 8, shake: 3, grow: -0.35 }), {});
addCase('sheet turned 180', handPath(sim.rotateQuad(base(), 180), FRAMES, { sway: 10, shake: 4 }), {});

/* --------------------------------------------------------------------- run */

console.log('case                        read   frame  correct   seen  miss   ms');
let perfect = 0;
for (const item of cases) {
  const r = item.result;
  if (r.settled && r.correct === TOTAL) perfect++;
  console.log(
    item.name.padEnd(28)
    + (r.settled ? 'yes' : 'NO ').padEnd(7)
    + String(r.settleFrame).padEnd(7)
    + (r.correct + '/' + TOTAL + (r.wrong ? ` (${r.wrong} wrong)` : '')).padEnd(10)
    + String(r.detected).padStart(5)
    + String(r.missed).padStart(6)
    + ('  ' + Math.round(r.readMs))
  );
  if (!r.settled) console.log('   never read: sheet seen in ' + r.detected + ' frames, missed ' + r.missed);
}
console.log('---------------------------------------------------------------');
console.log('read and perfect: ' + perfect + '/' + cases.length + ' videos');
process.exitCode = perfect === cases.length ? 0 : 1;
