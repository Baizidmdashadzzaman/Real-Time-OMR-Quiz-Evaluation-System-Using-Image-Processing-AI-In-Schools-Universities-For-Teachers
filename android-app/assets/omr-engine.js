/*
 * Offline OMR engine for the 30-question / 4-option answer sheet.
 *
 * The sheet is located by its printed pink alignment frame, mapped to the
 * canonical page with a homography, then every bubble is measured.  No image
 * or result is written anywhere; the caller owns the returned data.
 */
(function (root) {
  'use strict';

  const T = root.OMR_TEMPLATE;
  if (!T) throw new Error('OMR_TEMPLATE is missing');

  const MM_PER_INCH = 25.4;
  const DPI = 150;
  const PAGE_W = Math.round(T.page_mm[0] / MM_PER_INCH * DPI);
  const PAGE_H = Math.round(T.page_mm[1] / MM_PER_INCH * DPI);
  const FRAME = T.frame_mm;
  const FRAME_W = FRAME[2] - FRAME[0];
  const FRAME_H = FRAME[3] - FRAME[1];
  const FRAME_ASPECT = Math.min(FRAME_W, FRAME_H) / Math.max(FRAME_W, FRAME_H);
  const FRAME_DST = [
    [FRAME[0] / MM_PER_INCH * DPI, FRAME[1] / MM_PER_INCH * DPI],
    [FRAME[2] / MM_PER_INCH * DPI, FRAME[1] / MM_PER_INCH * DPI],
    [FRAME[2] / MM_PER_INCH * DPI, FRAME[3] / MM_PER_INCH * DPI],
    [FRAME[0] / MM_PER_INCH * DPI, FRAME[3] / MM_PER_INCH * DPI],
  ];
  const PAGE_DST = [[0, 0], [PAGE_W - 1, 0], [PAGE_W - 1, PAGE_H - 1], [0, PAGE_H - 1]];
  const BUBBLES = T.bubbles.map(([q, opt, x, y, r]) => ({
    q, opt,
    x: x / MM_PER_INCH * DPI,
    y: y / MM_PER_INCH * DPI,
    r: r / MM_PER_INCH * DPI,
  }));
  const BUBBLE_BY_KEY = new Map(BUBBLES.map(b => [`${b.q}:${b.opt}`, b]));
  const QUESTION_COUNT = 30;

  function bubblePx(q, option) {
    return BUBBLE_BY_KEY.get(`${q}:${option}`) || null;
  }

  const OPTION_LABELS = ['ক', 'খ', 'গ', 'ঘ'];

  /* ------------------------------------------------------------------ maths */

  function polygonArea(points) {
    let s = 0;
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      s += a[0] * b[1] - b[0] * a[1];
    }
    return s / 2;
  }

  function cross(o, a, b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  }

  function quadSides(q) {
    return q.map((p, i) => {
      const n = q[(i + 1) % 4];
      return Math.hypot(n[0] - p[0], n[1] - p[1]);
    });
  }

  function orderQuad(points) {
    let pts = points.map(p => [p[0], p[1]]);
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    pts.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    let start = 0;
    let best = Infinity;
    for (let i = 0; i < 4; i++) {
      const s = pts[i][0] + pts[i][1];
      if (s < best) { best = s; start = i; }
    }
    pts = pts.slice(start).concat(pts.slice(0, start));
    if (cross(pts[0], pts[1], pts[2]) < 0) pts.reverse();
    return pts;
  }

  function solveLinear(A, b) {
    const n = b.length;
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(A[row][col]) > Math.abs(A[pivot][col])) pivot = row;
      }
      if (Math.abs(A[pivot][col]) < 1e-10) return null;
      if (pivot !== col) {
        const tmp = A[pivot]; A[pivot] = A[col]; A[col] = tmp;
        const tv = b[pivot]; b[pivot] = b[col]; b[col] = tv;
      }
      const div = A[col][col];
      for (let j = col; j < n; j++) A[col][j] /= div;
      b[col] /= div;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const f = A[row][col];
        if (Math.abs(f) < 1e-14) continue;
        for (let j = col; j < n; j++) A[row][j] -= f * A[col][j];
        b[row] -= f * b[col];
      }
    }
    return b;
  }

  function homography(src, dst) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
      b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
      b.push(v);
    }
    const h = solveLinear(A, b);
    if (!h) return null;
    return h.concat(1);
  }

  function applyH(H, x, y) {
    const d = H[6] * x + H[7] * y + 1;
    return [
      (H[0] * x + H[1] * y + H[2]) / d,
      (H[3] * x + H[4] * y + H[5]) / d,
    ];
  }

  function roll(points, k) {
    const out = [];
    for (let i = 0; i < 4; i++) out.push(points[(i + k) % 4]);
    return out;
  }

  /* --------------------------------------------------------------- imaging */

  function downscaleRGBA(src, sw, sh, dw, dh) {
    if (dw === sw && dh === sh) return new Uint8ClampedArray(src);
    const out = new Uint8ClampedArray(dw * dh * 4);
    const xr = sw / dw, yr = sh / dh;
    for (let y = 0; y < dh; y++) {
      const sy0 = Math.floor(y * yr);
      const sy1 = Math.max(sy0 + 1, Math.min(sh, Math.ceil((y + 1) * yr)));
      for (let x = 0; x < dw; x++) {
        const sx0 = Math.floor(x * xr);
        const sx1 = Math.max(sx0 + 1, Math.min(sw, Math.ceil((x + 1) * xr)));
        let r = 0, g = 0, b = 0, a = 0, n = 0;
        for (let yy = sy0; yy < sy1; yy++) {
          let idx = (yy * sw + sx0) * 4;
          for (let xx = sx0; xx < sx1; xx++, idx += 4) {
            r += src[idx]; g += src[idx + 1]; b += src[idx + 2]; a += src[idx + 3]; n++;
          }
        }
        const o = (y * dw + x) * 4;
        out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = a / n;
      }
    }
    return out;
  }

  function pinkMask(data, w, h) {
    const mask = new Uint8Array(w * h);
    for (let i = 0, p = 0; p < w * h; i += 4, p++) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      if (r - g > 18 && r - b > 6 && b - g > -2 && r > 130) mask[p] = 1;
    }
    return mask;
  }

  function dilate3(mask, w, h) {
    const tmp = new Uint8Array(w * h);
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        if (mask[row + x] || (x > 0 && mask[row + x - 1]) || (x + 1 < w && mask[row + x + 1])) {
          tmp[row + x] = 1;
        }
      }
    }
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        if (tmp[row + x] || (y > 0 && tmp[row - w + x]) || (y + 1 < h && tmp[row + w + x])) {
          out[row + x] = 1;
        }
      }
    }
    return out;
  }

  function components(mask, w, h, minCount) {
    const seen = new Uint8Array(w * h);
    const stack = new Int32Array(w * h);
    const out = [];
    for (let start = 0; start < w * h; start++) {
      if (!mask[start] || seen[start]) continue;
      let sp = 0;
      stack[sp++] = start;
      seen[start] = 1;
      let count = 0, minX = w, minY = h, maxX = 0, maxY = 0;
      const points = [];
      while (sp > 0) {
        const idx = stack[--sp];
        const x = idx % w;
        const y = (idx / w) | 0;
        count++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        // Only the outline is needed for the hull.  A filled page has ~1M
        // pixels but only a few thousand boundary pixels; storing all of them
        // would make hull construction needlessly slow on a phone.
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
            !mask[idx - 1] || !mask[idx + 1] || !mask[idx - w] || !mask[idx + w]) {
          points.push(x, y);
        }
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx;
            if (nx < 0 || nx >= w) continue;
            const ni = ny * w + nx;
            if (mask[ni] && !seen[ni]) {
              seen[ni] = 1;
              stack[sp++] = ni;
            }
          }
        }
      }
      if (count >= minCount) {
        out.push({ count, minX, minY, maxX, maxY, points });
      }
    }
    return out;
  }

  function convexHull(points) {
    const pts = [];
    for (let i = 0; i < points.length; i += 2) pts.push([points[i], points[i + 1]]);
    pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const unique = [];
    for (const p of pts) {
      if (!unique.length || unique[unique.length - 1][0] !== p[0] || unique[unique.length - 1][1] !== p[1]) {
        unique.push(p);
      }
    }
    if (unique.length <= 3) return unique;
    const lower = [];
    for (const p of unique) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = unique.length - 1; i >= 0; i--) {
      const p = unique[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper);
  }

  function reduceToQuad(hull) {
    let pts = hull.map(p => [p[0], p[1]]);
    if (pts.length < 4) return null;
    while (pts.length > 4) {
      let bestLoss = Infinity, bestIndex = 0;
      for (let i = 0; i < pts.length; i++) {
        const prev = pts[(i - 1 + pts.length) % pts.length];
        const next = pts[(i + 1) % pts.length];
        const loss = Math.abs(cross(prev, pts[i], next)) / 2;
        if (loss < bestLoss) { bestLoss = loss; bestIndex = i; }
      }
      pts.splice(bestIndex, 1);
    }
    const q = orderQuad(pts);
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
      if (cross(a, b, c) < 0) return null;
    }
    return q;
  }

  function median(values) {
    if (!values.length) return 0;
    const v = values.slice().sort((a, b) => a - b);
    const m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }

  function percentile(values, p) {
    if (!values.length) return 0;
    const v = values.slice().sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.floor(p * (v.length - 1)))];
  }

  function borderMedian(data, w, h) {
    const band = Math.max(2, Math.round(Math.min(w, h) * 0.02));
    const vals = [];
    const step = Math.max(1, Math.round(Math.max(w, h) / 500));
    const add = (x, y) => {
      const i = (y * w + x) * 4;
      vals.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    };
    for (let x = 0; x < w; x += step) {
      for (let y = 0; y < band; y++) { add(x, y); add(x, h - 1 - y); }
    }
    for (let y = 0; y < h; y += step) {
      for (let x = 0; x < band; x++) { add(x, y); add(w - 1 - x, y); }
    }
    return median(vals);
  }

  function isFlatScan(data, w, h) {
    const pageAspect = T.page_mm[0] / T.page_mm[1];
    const aspect = Math.min(w, h) / Math.max(w, h);
    if (Math.abs(Math.log(aspect / pageAspect)) > 0.015) return false;
    return borderMedian(data, w, h) >= 245;
  }

  function findPanel(comps, frameComp, frameArea, w, h) {
    const candidates = [];
    for (const c of comps) {
      if (frameComp && c === frameComp) continue;
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      if (bw < 0.03 * w || bw > 0.30 * w) continue;
      if (bh < 0.02 * h || bh > 0.30 * h) continue;
      if (c.count < Math.max(150, 0.003 * frameArea)) continue;
      if (c.count / (bw * bh) < 0.10) continue;
      candidates.push(c);
    }
    if (!candidates.length) return null;
    candidates.sort((a, b) => b.count - a.count);
    const use = candidates.slice(0, 3);
    return [
      use.reduce((s, c) => s + (c.minX + c.maxX) / 2, 0) / use.length,
      use.reduce((s, c) => s + (c.minY + c.maxY) / 2, 0) / use.length,
    ];
  }

  function expectedPanelMm() {
    const r = T.regions;
    const keys = ['student_id', 'subject_code', 'total_marks'].filter(k => r[k]);
    return [
      keys.reduce((s, k) => s + r[k].x_mm + r[k].w_mm / 2, 0) / keys.length,
      keys.reduce((s, k) => s + r[k].y_mm + r[k].h_mm / 2, 0) / keys.length,
    ];
  }

  function chooseOrientation(corners, panelFull, dstQuad) {
    if (!panelFull) return 0;
    const exp = expectedPanelMm();
    let orientation = 0, bestDist = Infinity;
    for (let k = 0; k < 4; k++) {
      const H = homography(roll(corners, k), dstQuad);
      if (!H) continue;
      const p = applyH(H, panelFull[0], panelFull[1]);
      const mmx = p[0] * MM_PER_INCH / DPI;
      const mmy = p[1] * MM_PER_INCH / DPI;
      const dist = Math.hypot((mmx - exp[0]) / 20, (mmy - exp[1]) / 20);
      if (dist < bestDist) { bestDist = dist; orientation = k; }
    }
    return orientation;
  }

  function detectPage(data, small, dw, dh, pinkComps, scale) {
    const pageAspect = Math.min(T.page_mm[0], T.page_mm[1]) / Math.max(T.page_mm[0], T.page_mm[1]);
    const bright = new Uint8Array(dw * dh);
    for (let i = 0, p = 0; p < dw * dh; i += 4, p++) {
      const gray = 0.299 * small[i] + 0.587 * small[i + 1] + 0.114 * small[i + 2];
      if (gray > 110) bright[p] = 1;
    }
    const pageComps = components(bright, dw, dh, 500);
    let best = null;
    for (const c of pageComps) {
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      if (bw < 0.35 * dw || bh < 0.35 * dh) continue;
      if (c.count / (bw * bh) < 0.35) continue;
      const hull = convexHull(c.points);
      const quad = reduceToQuad(hull);
      if (!quad) continue;
      const sides = quadSides(quad);
      const aspect = Math.min(...sides) / Math.max(...sides);
      const aspectErr = Math.abs(aspect - pageAspect);
      if (aspectErr > 0.12) continue;
      const area = Math.abs(polygonArea(quad));
      if (area < 0.15 * dw * dh) continue;
      const score = area * (1 - aspectErr / 0.12);
      if (!best || score > best.score) best = { score, quad, area };
    }
    if (!best) return null;

    const corners = orderQuad(best.quad).map(p => [p[0] / scale, p[1] / scale]);
    const panel = findPanel(pinkComps, null, best.area, dw, dh);
    const panelFull = panel ? [panel[0] / scale, panel[1] / scale] : null;
    const orientation = chooseOrientation(corners, panelFull, PAGE_DST);
    return {
      mode: 'page',
      corners: roll(corners, orientation),
      orientation,
      confidence: Math.min(0.9, 0.5 + 0.4 * Math.min(1, best.area / (0.35 * dw * dh))),
    };
  }

  function detectFrame(data, width, height) {
    if (isFlatScan(data, width, height)) {
      return {
        mode: 'flat',
        corners: [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        orientation: 0,
        confidence: 0.9,
      };
    }

    const maxDet = 1200;
    const scale = Math.min(1, maxDet / Math.max(width, height));
    const dw = Math.max(1, Math.round(width * scale));
    const dh = Math.max(1, Math.round(height * scale));
    const small = downscaleRGBA(data, width, height, dw, dh);
    const mask = dilate3(pinkMask(small, dw, dh), dw, dh);
    const comps = components(mask, dw, dh, 180);

    let best = null;
    for (const c of comps) {
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      if (bw < 0.35 * dw || bh < 0.12 * dh) continue;
      const fill = c.count / (bw * bh);
      if (fill > 0.25) continue;
      const hull = convexHull(c.points);
      const quad = reduceToQuad(hull);
      if (!quad) continue;
      const sides = quadSides(quad);
      const aspect = Math.min(...sides) / Math.max(...sides);
      const aspectErr = Math.abs(aspect - FRAME_ASPECT);
      if (aspectErr > 0.10) continue;
      const area = Math.abs(polygonArea(quad));
      if (area < 0.03 * dw * dh) continue;
      const score = area * (1 - aspectErr / 0.10);
      if (!best || score > best.score) best = { score, quad, area, comp: c, aspect, fill };
    }

    if (!best) {
      const page = detectPage(data, small, dw, dh, comps, scale);
      if (page) return page;
      return { mode: 'none', corners: null, orientation: 0, confidence: 0 };
    }

    const panel = findPanel(comps, best.comp, best.area, dw, dh);
    const corners = orderQuad(best.quad).map(p => [p[0] / scale, p[1] / scale]);
    const panelFull = panel ? [panel[0] / scale, panel[1] / scale] : null;
    const orientation = chooseOrientation(corners, panelFull, FRAME_DST);
    const conf = 0.6 + 0.3 * Math.min(1, best.area / (0.10 * dw * dh)) + 0.1;
    return {
      mode: 'frame',
      corners: roll(corners, orientation),
      orientation,
      confidence: Math.min(0.99, conf),
      frameAspect: best.aspect,
    };
  }

  function warpPage(src, sw, sh, srcQuad, dstQuad, outW, outH) {
    const H = homography(dstQuad, srcQuad);
    const out = new Uint8ClampedArray(outW * outH * 4);
    if (!H) return out;
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const d = H[6] * x + H[7] * y + 1;
        const sx = (H[0] * x + H[1] * y + H[2]) / d;
        const sy = (H[3] * x + H[4] * y + H[5]) / d;
        const o = (y * outW + x) * 4;
        if (sx < 0 || sy < 0 || sx > sw - 1 || sy > sh - 1) {
          out[o] = 255; out[o + 1] = 255; out[o + 2] = 255; out[o + 3] = 255;
          continue;
        }
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1);
        const fx = sx - x0, fy = sy - y0;
        const i00 = (y0 * sw + x0) * 4, i10 = (y0 * sw + x1) * 4;
        const i01 = (y1 * sw + x0) * 4, i11 = (y1 * sw + x1) * 4;
        for (let c = 0; c < 4; c++) {
          const top = src[i00 + c] * (1 - fx) + src[i10 + c] * fx;
          const bot = src[i01 + c] * (1 - fx) + src[i11 + c] * fx;
          out[o + c] = top * (1 - fy) + bot * fy;
        }
      }
    }
    return out;
  }

  /* --------------------------------------------------------------- reading */

  function calibrate(scores, ringBaseline) {
    const fallback = Math.max(0.10, 0.40 * ringBaseline);
    const lo = Math.min(...scores), hi = Math.max(...scores);
    if (hi - lo < 0.03 || scores.length < 8) {
      return { threshold: fallback, method: 'fallback' };
    }
    let c0 = lo, c1 = hi, v0 = 0, v1 = 0, n0 = 0, n1 = 0;
    for (let iter = 0; iter < 25; iter++) {
      n0 = 0; n1 = 0;
      let s0 = 0, s1 = 0;
      for (const s of scores) {
        if (Math.abs(s - c0) <= Math.abs(s - c1)) { n0++; s0 += s; }
        else { n1++; s1 += s; }
      }
      if (!n0 || !n1) break;
      const m0 = s0 / n0, m1 = s1 / n1;
      v0 = 0; v1 = 0;
      for (const s of scores) {
        if (Math.abs(s - m0) <= Math.abs(s - m1)) v0 += (s - m0) ** 2;
        else v1 += (s - m1) ** 2;
      }
      v0 /= n0; v1 /= n1;
      if (Math.abs(m0 - c0) < 1e-6 && Math.abs(m1 - c1) < 1e-6) {
        c0 = m0; c1 = m1;
        break;
      }
      c0 = m0; c1 = m1;
    }
    const low = Math.min(c0, c1), high = Math.max(c0, c1);
    const lowVar = c0 <= c1 ? v0 : v1;
    const highCount = c0 <= c1 ? n1 : n0;
    const separation = (high - low) / Math.sqrt(Math.max(1e-9, (v0 + v1) / 2));
    if (high - low > 0.08 && highCount >= 3 && separation >= 2.0) {
      return {
        threshold: Math.max(low + 3 * Math.sqrt(Math.max(0, lowVar)), 0.35 * ringBaseline, 0.10),
        method: 'kmeans',
      };
    }
    return { threshold: fallback, method: 'fallback' };
  }

  function readBubbles(data, width, height) {
    const scores = [];
    const rings = [];
    const readings = [];

    for (const b of BUBBLES) {
      const inner = [], ann = [], outer = [];
      const x0 = Math.max(0, Math.floor(b.x - b.r * 1.6));
      const x1 = Math.min(width - 1, Math.ceil(b.x + b.r * 1.6));
      const y0 = Math.max(0, Math.floor(b.y - b.r * 1.6));
      const y1 = Math.min(height - 1, Math.ceil(b.y + b.r * 1.6));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x + 0.5 - b.x, dy = y + 0.5 - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 > (1.60 * b.r) ** 2) continue;
          const i = (y * width + x) * 4;
          const ink = 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          if (d2 <= (0.52 * b.r) ** 2) inner.push(ink);
          else if (d2 >= (0.85 * b.r) ** 2 && d2 <= (1.15 * b.r) ** 2) ann.push(ink);
          else if (d2 >= (1.35 * b.r) ** 2) outer.push(ink);
        }
      }
      // A phone photo is rarely evenly lit.  Use the paper immediately around
      // this bubble as its zero-ink reference instead of assuming white.
      const paper = outer.length ? median(outer) : 0;
      const fill = inner.length ? median(inner) / 255 : 0;
      const ring = ann.length ? percentile(ann, 0.75) / 255 : 0;
      const fillRel = Math.max(0, (median(inner) - paper) / 255);
      const darkFrac = inner.length
        ? inner.filter(v => v > paper + 70).length / inner.length
        : 0;
      const score = 0.70 * fillRel + 0.30 * darkFrac;
      scores.push(score);
      rings.push(ring);
      readings.push({ q: b.q, option: b.opt, score, fill, ring, paper: paper / 255, darkFrac });
    }

    const ringBaseline = median(rings);
    const cal = calibrate(scores, ringBaseline);
    const threshold = cal.threshold;

    for (const r of readings) {
      // `fill` is the absolute darkness of the bubble interior; a score can be
      // high on a shadowed sheet, so a real mark also needs actual ink.
      if (r.score >= threshold && r.score >= 0.15 && r.fill >= 0.40) {
        r.state = 'MARKED';
      } else if (r.score > threshold * 0.90 && r.fill >= 0.15) {
        r.state = 'UNCERTAIN';
      } else {
        r.state = 'EMPTY';
      }
    }

    const questions = {};
    for (let q = 1; q <= QUESTION_COUNT; q++) {
      const rs = readings.filter(r => r.q === q);
      const marked = rs.filter(r => r.state === 'MARKED');
      const uncertain = rs.filter(r => r.state === 'UNCERTAIN');
      let state = 'BLANK', selected = [];
      if (marked.length > 1) {
        state = 'MULTI';
        selected = marked.map(r => r.option);
      } else if (marked.length === 1 && uncertain.length === 0) {
        state = 'ANSWERED';
        selected = [marked[0].option];
      } else if (marked.length === 1) {
        state = 'REVIEW';
        selected = [marked[0].option];
      } else if (uncertain.length) {
        state = 'REVIEW';
      }
      questions[q] = { q, state, selected };
    }

    return {
      questions,
      bubbles: readings,
      calibration: cal,
      diagnostics: {
        ringBaseline,
        scoreMin: Math.min(...scores),
        scoreMax: Math.max(...scores),
        markedBubbles: readings.filter(r => r.state === 'MARKED').length,
      },
    };
  }

  function gradeRead(read, key, marksPerQuestion) {
    const marks = Number(marksPerQuestion) || 1;
    const details = [];
    let correct = 0, wrong = 0, blank = 0, multi = 0, review = 0, noKey = 0;
    for (let q = 1; q <= QUESTION_COUNT; q++) {
      const r = read.questions[q];
      const answer = key && Number.isInteger(key[q - 1]) ? key[q - 1] : null;
      let outcome;
      if (answer === null) outcome = 'NO_KEY';
      else if (r.state === 'ANSWERED') outcome = r.selected[0] === answer ? 'CORRECT' : 'WRONG';
      else if (r.state === 'MULTI') outcome = 'MULTI';
      else if (r.state === 'BLANK') outcome = 'BLANK';
      else outcome = 'REVIEW';
      if (outcome === 'CORRECT') correct++;
      else if (outcome === 'WRONG') wrong++;
      else if (outcome === 'BLANK') blank++;
      else if (outcome === 'MULTI') multi++;
      else if (outcome === 'REVIEW') review++;
      else noKey++;
      details.push({ q, selected: r.selected, answer, state: r.state, outcome });
    }
    return {
      correct, wrong, blank, multi, review, noKey,
      total: correct * marks,
      maximum: QUESTION_COUNT * marks,
      marksPerQuestion: marks,
      details,
    };
  }

  function scanRGBA(data, width, height, key, marksPerQuestion, precomputedFrame) {
    const frame = precomputedFrame || detectFrame(data, width, height);
    if (frame.mode === 'none') {
      return {
        ok: false,
        error: 'Could not find the pink alignment frame. Fill the frame with the answer area and retake.',
      };
    }
    const warped = warpPage(
      data, width, height, frame.corners,
      frame.mode === 'frame' ? FRAME_DST : PAGE_DST,
      PAGE_W, PAGE_H
    );
    const read = readBubbles(warped, PAGE_W, PAGE_H);
    const grade = gradeRead(read, key, marksPerQuestion);
    return {
      ok: true,
      mode: frame.mode,
      confidence: frame.confidence,
      orientation: frame.orientation,
      corners: frame.corners,
      warp: { data: warped, width: PAGE_W, height: PAGE_H },
      read,
      grade,
    };
  }

  root.OMREngine = {
    scanRGBA,
    detectFrame,
    readBubbles,
    gradeRead,
    homography,
    applyH,
    warpPage,
    bubblePx,
    BUBBLES,
    FRAME_DST,
    PAGE_DST,
    PAGE_W,
    PAGE_H,
    DPI,
    OPTION_LABELS,
  };
})(typeof window !== 'undefined' ? window : globalThis);
