/*
 * Shared simulation helpers for the reader tests: a renderer for the printed
 * sheet design, a perspective "photograph" of it through a quad, and the ways a
 * classroom camera degrades a picture (noise, blur, uneven light).
 *
 * Extracted so the video test and the conditions test measure the same sheet.
 */
'use strict';

const DPI = 150;
const PINK = [200, 111, 136];
const BLACK = [25, 25, 28];
const GREY = [72, 72, 76];
const DESK = [74, 80, 88];

/* --------------------------------------------------------------- rendering */

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
  const [x0, y0, x1, y1] = template.frame_mm.map(mm);
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
    if (answers[q - 1] === option) fillCircle(canvas, cx, cy, radius * 0.82, ink || BLACK);
  }
  
  // Numbered band above every question row, plus the pink info boxes the real
  // sheet prints next to the table — the printed asymmetry the orientation
  // check reads.
  const rowTopsPx = [];
  let lastTop = null;
  for (const [q, option, x, y] of template.bubbles) {
    if (option !== 0) continue;
    if (lastTop === null || Math.abs(y - lastTop) > 1) { rowTopsPx.push(y); lastTop = y; }
  }
  for (const y of rowTopsPx) {
    // The printed numbers band: a pink strip ~11.4 .. ~5.0 mm above the row-A
    // line, as measured on the production sheet (the reader samples this band
    // as the front-of-the-page witness for the 180-degree turn).
    const bandTop = mm(y - 11.4);
    const bandBottom = mm(y - 5.0);
    if (bandTop > y0 + 1.5 && bandBottom < y1 - 1.5) {
      // Light pink band background, like the real print (#f7c4df), so the
      // rings stay the strongest structure the alignment can lock onto.
      fillRectPx(canvas, x0 + 1.5, bandTop, x1 - 1.5, bandBottom, [247, 196, 223]);
      // Dark question-numbers ink, as printed.

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

function keyFor(count) {
  const key = [];
  for (let i = 0; i < count; i++) key.push((i * 3 + 1) % 4);
  return key;
}

/* ------------------------------------------------------------- photography */

/**
 * Photographs the page through a quad (page corners in photo coordinates) onto
 * a desk-coloured background.  This is the piece the other tests do not have:
 * real perspective, not a straight scan.
 */
function photograph(engine, page, quad, outW, outH, background) {
  const bg = background || DESK;
  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = bg[0];
    out[i + 1] = bg[1];
    out[i + 2] = bg[2];
    out[i + 3] = 255;
  }
  const pageCorners = [
    [0, 0],
    [page.width - 1, 0],
    [page.width - 1, page.height - 1],
    [0, page.height - 1],
  ];
  const H = engine.homography(quad, pageCorners);
  if (!H) return { data: out, width: outW, height: outH };

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of quad) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  minX = Math.max(0, Math.floor(minX) - 2);
  minY = Math.max(0, Math.floor(minY) - 2);
  maxX = Math.min(outW - 1, Math.ceil(maxX) + 2);
  maxY = Math.min(outH - 1, Math.ceil(maxY) + 2);

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const d = H[6] * x + H[7] * y + 1;
      const sx = (H[0] * x + H[1] * y + H[2]) / d;
      const sy = (H[3] * x + H[4] * y + H[5]) / d;
      if (sx < 0 || sy < 0 || sx > page.width - 1.001 || sy > page.height - 1.001) continue;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = Math.min(page.width - 1, x0 + 1), y1 = Math.min(page.height - 1, y0 + 1);
      const fx = sx - x0, fy = sy - y0;
      const i00 = (y0 * page.width + x0) * 4, i10 = (y0 * page.width + x1) * 4;
      const i01 = (y1 * page.width + x0) * 4, i11 = (y1 * page.width + x1) * 4;
      const o = (y * outW + x) * 4;
      for (let c = 0; c < 3; c++) {
        const top = page.data[i00 + c] * (1 - fx) + page.data[i10 + c] * fx;
        const bot = page.data[i01 + c] * (1 - fx) + page.data[i11 + c] * fx;
        out[o + c] = top * (1 - fy) + bot * fy;
      }
    }
  }
  return { data: out, width: outW, height: outH };
}

/* Rendered before the degradation helpers, which all mutate in place. */

function makeRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function brighten(image, factor, offset) {
  const add = offset || 0;
  for (let i = 0; i < image.data.length; i += 4) {
    image.data[i] = image.data[i] * factor + add;
    image.data[i + 1] = image.data[i + 1] * factor + add;
    image.data[i + 2] = image.data[i + 2] * factor + add;
  }
  return image;
}

/** A light gradient across the page, as a hand or the head shadows it. */
function shadowAcross(image, from, to) {
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const k = from + (to - from) * (x / image.width);
      const i = (y * image.width + x) * 4;
      image.data[i] *= k;
      image.data[i + 1] *= k;
      image.data[i + 2] *= k;
    }
  }
  return image;
}

function blur(image, passes) {
  const w = image.width, h = image.height;
  let src = image.data;
  for (let pass = 0; pass < (passes || 1); pass++) {
    const out = new Uint8ClampedArray(src.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0, n = 0;
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const sx = Math.min(w - 1, Math.max(0, x + ox));
            const sy = Math.min(h - 1, Math.max(0, y + oy));
            const i = (sy * w + sx) * 4;
            r += src[i]; g += src[i + 1]; b += src[i + 2]; n++;
          }
        }
        const o = (y * w + x) * 4;
        out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
      }
    }
    src = out;
  }
  image.data = src;
  return image;
}

function addNoise(image, amount, seed) {
  const rnd = makeRandom(seed || 7);
  for (let i = 0; i < image.data.length; i += 4) {
    const n = (rnd() - 0.5) * 2 * amount;
    image.data[i] += n;
    image.data[i + 1] += n;
    image.data[i + 2] += n;
  }
  return image;
}

function scaleContent(page, factor) {
  const width = page.width;
  const height = page.height;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = 255; out[i + 1] = 255; out[i + 2] = 255; out[i + 3] = 255;
  }
  const cx = width / 2, cy = height / 2;
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

function rotateQuarter(source) {
  const { data, width, height } = source;
  const outW = height;
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
  return { data: out, width: outW, height };
}

/* ------------------------------------------------------------------ quads */

const PHOTO_W = 1400;
const PHOTO_H = 1050;

/** Page corners for a phone held at an angle, page filling most of the frame. */
function quadUpright() {
  return [[392, 48], [1048, 48], [1048, 1002], [392, 1002]];
}

function rotateQuad(quad, degrees) {
  let cx = 0, cy = 0;
  for (const [x, y] of quad) { cx += x; cy += y; }
  cx /= quad.length; cy /= quad.length;
  const a = degrees * Math.PI / 180;
  return quad.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    return [cx + dx * Math.cos(a) - dy * Math.sin(a), cy + dx * Math.sin(a) + dy * Math.cos(a)];
  });
}

/** Phone held up and to the left: the far edge is smaller. */
function quadSteep() {
  return [[500, 70], [960, 60], [1058, 1000], [382, 1000]];
}

/** The sheet is on the desk but the phone is far away. */
function quadSmall() {
  return [[560, 330], [842, 330], [842, 720], [560, 720]];
}

module.exports = {
  DPI, PINK, BLACK, GREY, DESK, PHOTO_W, PHOTO_H,
  blankPage, setPixel, strokeCircle, fillCircle, strokeLine, fillRectPx,
  renderSheet, keyFor, photograph, makeRandom, brighten, shadowAcross, blur,
  addNoise, scaleContent, rotateQuarter,
  quadUpright, rotateQuad, quadSteep, quadSmall,
};
