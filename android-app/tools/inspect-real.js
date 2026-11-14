/*
 * Runs the reader over a real photo or a folder of real video frames and
 * prints what it decided, frame by frame.  Used to diagnose classroom reports
 * and to check a fix against the real picture instead of a rendered one.
 *
 *   node tools/inspect-real.js photo tools/photo-stack.png
 *   node tools/inspect-real.js video tools/frames
 */
'use strict';

const path = require('path');
const fs = require('fs');

global.window = global;
require(path.join(__dirname, '..', 'assets', 'sheet-designs.js'));
require(path.join(__dirname, '..', 'assets', 'omr-engine.js'));
const { readPNG } = require(path.join(__dirname, '..', 'test', 'png.js'));

const ENGINE = window.OMREngineFactory.createEngine(window.OMR_SHEETS.master);
const KEY = new Array(60).fill(null);

function answersOf(read) {
  const out = [];
  for (let q = 1; q <= 60; q++) {
    const st = read.questions[q];
    out.push(st && st.selected.length ? 'ABCD'[st.selected[0]] : '.');
  }
  return out.join('');
}

function inspectPhoto(file) {
  const image = readPNG(file);
  const t0 = Date.now();
  const detection = ENGINE.detectFrame(image.data, image.width, image.height);
  console.log('detect:', detection.mode,
    'conf', detection.confidence && detection.confidence.toFixed(2),
    'corners', detection.corners && JSON.stringify(detection.corners.map(p => p.map(Math.round))));
  if (detection.corners && detection.mode !== 'flat') {
    const H = ENGINE.homography(
      detection.mode === 'frame' ? ENGINE.FRAME_DST : ENGINE.PAGE_DST,
      detection.corners);
    const area = ENGINE.readRegion(image.data, image.width, image.height, H);
    if (area) {
      console.log('region: fit', area.ringFit.toFixed(2), 'nudge', area.nudgeFit.toFixed(2),
        'onRings', area.onRings, 'radiusPx', area.radiusPx.toFixed(1), 'align', JSON.stringify(area.align));
      console.log('region answers:', answersOf({ questions: ENGINE.questionsFromReadings(area.readings) }));
    }
  }
  const result = ENGINE.scanRGBA(image.data, image.width, image.height, KEY, 1);
  console.log('scan:', result.ok ? 'ok' : 'FAILED', result.ok ? '' : result.error,
    'mode', result.mode, 'fit', result.ringFit && result.ringFit.toFixed(2),
    'orientation', result.orientation, 'align', JSON.stringify(result.align),
    'ms', Date.now() - t0);
  if (result.ok) {
    console.log('marked', result.read.bubbles.filter(b => b.state === 'MARKED').length,
      'review', result.read.bubbles.filter(b => b.state === 'UNCERTAIN').length);
    console.log('answers:', answersOf(result.read));
  }
}

function inspectVideo(dir) {
  const files = fs.readdirSync(dir).filter(f => /\.png$/i.test(f)).sort();
  let readFrames = 0;
  let best = null;
  const lines = [];
  for (const file of files) {
    const image = readPNG(path.join(dir, file));
    const t0 = Date.now();
    const detection = ENGINE.detectFrame(image.data, image.width, image.height);
    const ms = Date.now() - t0;
    let line = file + ' ' + (detection.mode || 'none').padEnd(5) + ' ' + String(ms).padStart(4) + 'ms';
    if (detection.corners && detection.mode !== 'none') {
      const result = ENGINE.scanRGBA(image.data, image.width, image.height, KEY, 1);
      if (result.ok) {
        readFrames++;
        const answers = answersOf(result.read);
        const marked = answers.replace(/\./g, '').length;
        line += ' READ marked ' + marked + ' fit ' + (result.ringFit || 0).toFixed(2)
          + ' orient ' + result.orientation + ' ' + answers;
        if (!best || marked > best.marked) best = { marked, answers, file };
      } else {
        line += ' refused (' + (result.error || '').slice(0, 60) + ')';
      }
    }
    lines.push(line);
  }
  console.log(lines.join('\n'));
  console.log('---');
  console.log('frames with a readable sheet:', readFrames + '/' + files.length);
  if (best) console.log('best frame', best.file, 'marked', best.marked, best.answers);
}

const mode = process.argv[2];
const target = process.argv[3];
if (mode === 'video') inspectVideo(target);
else inspectPhoto(target);
