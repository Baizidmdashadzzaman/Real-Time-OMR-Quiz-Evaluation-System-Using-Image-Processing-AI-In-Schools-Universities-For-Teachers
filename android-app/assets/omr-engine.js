/*
 * Offline OMR engine for the printed OMR sheet (print/v3.pdf): four rows of
 * 15 questions, four options A-D, 60 slots on one A4 page.
 *
 * The sheet is located by its printed pink alignment frame, mapped to the
 * canonical page with a homography, then every bubble is measured.  No image
 * or result is written anywhere; the caller owns the returned data.
 */
(function (root) {
  'use strict';

  const MM_PER_INCH = 25.4;
  const DPI = 150;
  const OPTION_LABELS = ['A', 'B', 'C', 'D'];

  /*
   * Everything below is derived from the template handed in, so a 15-, 25-,
   * 30-, 40- and 50-question sheet all run through exactly the same code.
   */
  function createEngine(T) {
  if (!T) throw new Error('OMR template is missing');

  const QUESTION_COUNT = T.question_count ||
    T.bubbles.reduce((max, b) => Math.max(max, b[0]), 0);
  const OPTION_COUNT = T.option_count ||
    T.bubbles.reduce((max, b) => Math.max(max, b[1] + 1), 0);
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

  // Centroid of the whole bubble lattice: every alignment or scale correction
  // is applied about this point, so the grid grows or shrinks in place.
  const GRID_CX = BUBBLES.reduce((sum, b) => sum + b.x, 0) / BUBBLES.length;
  const GRID_CY = BUBBLES.reduce((sum, b) => sum + b.y, 0) / BUBBLES.length;

  /* Geometry of the numbered band above every question row, in page pixels.
   * On the printed sheet there is ink above each row (the question-numbers
   * band) and blank paper between a row's option D and the next band.  Turned
   * 180 degrees, that pattern turns over, so it tells the reader which side of
   * the lattice is at the top of the page even when every row is full. */
  const PX_PER_MM = DPI / MM_PER_INCH;

  // The live viewfinder is aimed at the answer table plus the printed mark box
  // beside it, not at the sheet.  A page outline can be imitated by a stack of
  // papers, a table edge or a notebook, and a shadowed page edge pulled the grid
  // off by a whole option row; this rectangle is unique on the desk.
  const GUIDE_MM = (function () {
    const [x0, y0, x1, y1] = T.frame_mm;
    const box = T.regions && T.regions.total_marks;
    return [x0, y0, box ? Math.max(x1, box.x_mm + box.w_mm) : x1, y1];
  })();
  const GUIDE_DST = [
    [GUIDE_MM[0] * PX_PER_MM, GUIDE_MM[1] * PX_PER_MM],
    [GUIDE_MM[2] * PX_PER_MM, GUIDE_MM[1] * PX_PER_MM],
    [GUIDE_MM[2] * PX_PER_MM, GUIDE_MM[3] * PX_PER_MM],
    [GUIDE_MM[0] * PX_PER_MM, GUIDE_MM[3] * PX_PER_MM],
  ];
  const ORI_MARGIN = 0.03;
  const BIAS_K = typeof window.BIAS_K === 'number'
    ? window.BIAS_K : 0.03;      // the winner must score this much above runner-up
  /* Witness strip of the printed numbers band, in mm above every row-A
   * centre.  Measured on the printed school sheet: the pink band spans
   * ~11.4 .. ~4.9 mm above line A, with the dark question digits
   * ~9.1..7.1 mm above it.  The strip sits safely inside that background
   * (and well clear of option A's own ring) on both this sheet and the
   * one the app prints, so a turned page shows ink here while paper
   * below option D does not.
   */
  const BAND_TOP_MM = 11.0;   // strip top, above option-A centre
  const BAND_LEN_MM = 4.9;    // strip height
  const ROW_TOPS = (function () {
    const tops = [];
    let last = null;
    for (const b of BUBBLES) {
      if (b.opt !== 0) continue;
      if (last === null || Math.abs(b.y - last) > 1) { tops.push(b.y); last = b.y; }
    }
    return tops;
  })();
  const FIRST_A = BUBBLES.find(b => b.opt === 0);
  const FIRST_D_OF_A = BUBBLES.find(b => b.opt === 3 && (FIRST_A && b.x === FIRST_A.x));
  const OPTION_PITCH_PX = (FIRST_A && FIRST_D_OF_A) ? (FIRST_D_OF_A.y - FIRST_A.y) / 3 : 0;
  const RADIUS_PX = BUBBLES.length ? BUBBLES[0].r : 0;
  const GRID_MIN_X = Math.min(...BUBBLES.map(b => b.x)) - RADIUS_PX;
  const GRID_MAX_X = Math.max(...BUBBLES.map(b => b.x)) + RADIUS_PX;

  /** An evenly spread subset of the bubbles, for alignment and ring scoring. */
  function sampleBubbles(want) {
    const stride = Math.max(1, Math.ceil(BUBBLES.length / want));
    return BUBBLES.filter((b, i) => i % stride === 0);
  }

  /**
   * The answer area of the sheet: the box that holds every bubble, in page
   * pixels.  The live camera reads only this part, so the printed header, the
   * instructions and any handwriting never enter the measurement.
   */
  function omrRegion(padMm) {
    const pad = (padMm === undefined ? 3 : padMm) / MM_PER_INCH * DPI;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const b of BUBBLES) {
      if (b.x - b.r < minX) minX = b.x - b.r;
      if (b.y - b.r < minY) minY = b.y - b.r;
      if (b.x + b.r > maxX) maxX = b.x + b.r;
      if (b.y + b.r > maxY) maxY = b.y + b.r;
    }
    return {
      x: minX - pad,
      y: minY - pad,
      w: (maxX - minX) + pad * 2,
      h: (maxY - minY) + pad * 2,
    };
  }

  function regionCorners(region) {
    return [
      [region.x, region.y],
      [region.x + region.w, region.y],
      [region.x + region.w, region.y + region.h],
      [region.x, region.y + region.h],
    ];
  }

  function bubblePx(q, option) {
    return BUBBLE_BY_KEY.get(`${q}:${option}`) || null;
  }

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

  /** Otsu cut for a greyscale image: paper and background split where they differ. */
  function otsuThreshold(data, w, h) {
    const hist = new Uint32Array(256);
    const total = w * h;
    for (let i = 0, p = 0; p < total; i += 4, p++) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      hist[Math.max(0, Math.min(255, Math.round(gray)))]++;
    }
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];
    let sumB = 0, wB = 0, best = -1, cut = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; cut = t; }
    }
    return cut;
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

  /*
   * Band-number asymmetry: in template space the strip above every question
   * row (except the last) holds printed ink — the row's question numbers — and
   * the equally wide strip between that row's option D and the next band holds
   * blank paper.  This is measured through the candidate orientation's own
   * homography, so it scores +1 when the hypothesis puts the numbers above the
   * rows, drifts towards 0 when a hypothesis lands both strips on similar
   * content, and goes negative when the hypothesis puts the numbers below the
   * rows.
   */
  function bandAsymmetry(data, width, height, map) {
    if (ROW_TOPS.length < 2 || OPTION_PITCH_PX <= 0) return 0;
    // The numbered band above each row is darker than the paper *and* pinker
    // than it.  A dim classroom photo keeps the colour difference long after
    // the brightness difference has gone, while a phone screenshot of a bright
    // sheet has the brightness difference and little colour, so both are
    // measured and added: whichever survives this print and this light says
    // which way up the sheet is.
    const strip = (y0, y1) => {
      const yStart = Math.max(0, Math.round(y0));
      const yEnd = Math.min(height - 1, Math.round(y1));
      if (yEnd <= yStart) return null;
      const xStart = Math.max(0, Math.round(GRID_MIN_X));
      const xEnd = Math.min(width - 1, Math.round(GRID_MAX_X));
      let dark = 0, pink = 0, n = 0;
      for (let y = yStart; y <= yEnd; y += 2) {
        for (let x = xStart; x <= xEnd; x += 2) {
          const p = map(x, y);
          const px = Math.round(p[0]), py = Math.round(p[1]);
          if (px <= 0 || py <= 0 || px >= width - 1 || py >= height - 1) continue;
          const i = (py * width + px) * 4;
          dark += 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          pink += data[i] - data[i + 1];
          n++;
        }
      }
      return n ? { dark: dark / n / 255, pink: pink / n / 255 } : null;
    };
    const votes = [];
    const gapBelow = (rowTop) => {
      const dBottom = rowTop + 3 * OPTION_PITCH_PX + RADIUS_PX;
      const centre = dBottom + 0.85 * PX_PER_MM;
      return [centre - 0.35 * PX_PER_MM, centre + 0.35 * PX_PER_MM];
    };
    for (let r = 0; r + 1 < ROW_TOPS.length; r++) {
      const bandTop = ROW_TOPS[r] - (BAND_TOP_MM - 0.25) * PX_PER_MM;
      const bandBottom = bandTop + BAND_LEN_MM * PX_PER_MM;
      const above = strip(bandTop, bandBottom);
      if (!above) continue;
      const below = strip(...gapBelow(ROW_TOPS[r]));
      if (!below) continue;
      votes.push((above.dark - below.dark) + (above.pink - below.pink));
    }
    return votes.length ? median(votes) : 0;
  }



  /*
   * Ring-Likelihood Orientation Disambiguation (RLOD).
   *
   * A found quad can be interpreted four ways — the sheet may sit upright,
   * turned around, or on either side.  Earlier steps guess with the pink panel
   * boxes; RLOD verifies: every interpretation is a hypothesis, scored by how
   * well the template's bubble rings land on the photo's printed rings (ring
   * fit) and by the numbered-band asymmetry that separates the front of the
   * page from its 180-degree turn.  The winner must beat the runner-up by a
   * margin, or the reading refuses to grade rather than mark the wrong grid.
   */
  function orientationAnalysis(data, width, height, corners, dstQuad, panelGuess) {
    const scores = [];
    for (let k = 0; k < 4; k++) {
      const H = homography(dstQuad, roll(corners, k));
      if (!H) continue;
      const fit = ringFit(data, width, height, H, [0, 0, 1]);
      const band = bandAsymmetry(data, width, height, (x, y) => applyH(H, x, y));
      scores.push({ k, fit, band, score: fit + 0.9 * band });
    }
    scores.sort((a, b) => b.score - a.score);
    const best = scores[0] || { k: panelGuess || 0, fit: 0, band: 0, score: 0 };
    const second = scores[1] || null;
    const margin = second ? best.score - second.score : 1;
    // A spotted pink panel is a direct geometric anchor; when the ring score is
    // too soft to separate the two best turns (blur does this), the panel has
    // to decide — never the noise on the losing turn.  With no panel anchor
    // and no decisive margin, the turn is unresolved and must not be graded.
    if (!second) return { k: panelGuess || 0, fit: 0, band: 0, margin, ambiguous: true };
    if (margin >= ORI_MARGIN) {
      return { k: best.k, fit: best.fit, band: best.band, margin, ambiguous: false };
    }
    if (panelGuess !== null) {
      return { k: panelGuess, fit: best.fit, band: best.band, margin, ambiguous: false };
    }
    return { k: best.k, fit: best.fit, band: best.band, margin, ambiguous: true };
  }

  /**
   * Sheet outline without a usable pink border: the paper is the biggest bright
   * quad.  The cut between paper and desk comes from the picture's own histogram
   * (Otsu), not a fixed brightness, because a phone photo in a classroom is much
   * darker than a scan.  Both the page and the printed frame are accepted, and
   * the one that fits better decides which geometry the reader uses afterwards.
   */






  function detectPage(data, small, dw, dh, pinkComps, scale, width, height) {
    const pageAspect = Math.min(T.page_mm[0], T.page_mm[1]) / Math.max(T.page_mm[0], T.page_mm[1]);
    const cut = Math.max(90, Math.min(205, otsuThreshold(small, dw, dh)));
    const bright = new Uint8Array(dw * dh);
    for (let i = 0, p = 0; p < dw * dh; i += 4, p++) {
      const gray = 0.299 * small[i] + 0.587 * small[i + 1] + 0.114 * small[i + 2];
      if (gray > cut) bright[p] = 1;
    }
    const pageComps = components(bright, dw, dh, Math.max(200, 0.02 * dw * dh));
    let best = null;
    for (const c of pageComps) {
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      if (bw < 0.14 * dw || bh < 0.14 * dh) continue;
      if (c.count / (bw * bh) < 0.30) continue;
      const hull = convexHull(c.points);
      const quad = reduceToQuad(hull);
      if (!quad) continue;
      const sides = quadSides(quad);
      const aspect = Math.min(...sides) / Math.max(...sides);
      const errPage = Math.abs(aspect - pageAspect);
      const errFrame = Math.abs(aspect - FRAME_ASPECT);
      const err = Math.min(errPage, errFrame);
      // A shadow, a finger or a blurred edge can split the bright paper into
      // pieces, and a piece of a page still looks like "a" rectangle: one half
      // of the answer table has exactly the table's proportions.  Warping such
      // a piece to the sheet rescales the grid by half and reads nonsense, so
      // the shape has to fit the page (or the printed frame) closely now.
      // Perspective makes a page look squarer than it is (a sheet seen from
      // the side measured 0.49 here against a true 0.71), so this cannot be
      // tight.  A piece of the page is caught later instead: the live screen
      // only grades a frame the full reader accepts, and the reader rejects a
      // geometry whose printed rings do not line up.
      if (err > 0.15) continue;
      const area = Math.abs(polygonArea(quad));
      if (area < 0.05 * dw * dh) continue;
      const score = area * (1 - err / 0.15);
      if (!best || score > best.score) {
        best = { score, quad, area, mode: errFrame < errPage ? 'frame' : 'page' };
      }
    }
    if (!best) return null;

    const dstQuad = best.mode === 'frame' ? FRAME_DST : PAGE_DST;
    const corners = orderQuad(best.quad).map(p => [p[0] / scale, p[1] / scale]);
    const panel = findPanel(pinkComps, null, best.area, dw, dh);
    const panelFull = panel ? [panel[0] / scale, panel[1] / scale] : null;
    const analysis = orientationAnalysis(data, width, height, corners, dstQuad,
      panelFull ? chooseOrientation(corners, panelFull, dstQuad) : null);
    return {
      mode: best.mode,
      corners: roll(corners, analysis.k),
      orientation: analysis.k,
      ambiguous: analysis.ambiguous,
      orientationFit: analysis.fit,
      oriMargin: analysis.margin,
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

    // The answer table is drawn as separate thin pink strips -- four borders,
    // four numbered bands, fifteen column rules -- so on a real photo no single
    // component is ever the table.  Closing the pink mask joins them into one
    // blob whose outline *is* the table.  That matters for a stack of papers
    // (the table is unique, the paper edges are not) and for accuracy: the
    // bubbles all live inside this rectangle, so anchoring on it removes the
    // several-millimetre error that comes from using the page outline.
    const closed = dilate3(dilate3(mask, dw, dh), dw, dh);
    const tableComps = components(closed, dw, dh, Math.max(400, Math.round(0.02 * dw * dh)));

    let best = null;
    for (const c of comps) {
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      if (bw < 0.18 * dw || bh < 0.045 * dh) continue;
      const fill = c.count / (bw * bh);
      if (fill > 0.25) continue;
      const hull = convexHull(c.points);
      const quad = reduceToQuad(hull);
      if (!quad) continue;
      const sides = quadSides(quad);
      const aspect = Math.min(...sides) / Math.max(...sides);
      const aspectErr = Math.abs(aspect - FRAME_ASPECT);
      if (aspectErr > 0.15) continue;
      const area = Math.abs(polygonArea(quad));
      if (area < 0.03 * dw * dh) continue;
      const score = area * (1 - aspectErr / 0.10);
      if (!best || score > best.score) best = { score, quad, area, comp: c, aspect, fill };
    }

    const page = detectPage(data, small, dw, dh, comps, scale, width, height);

    // Pink-band anchor: the four numbered strips pin the table even when the
    // paper outline is lost in a shadow or the sheet lies on a stack of papers.


    const fromQuad = (quad, comp, extra) => {
      const corners = orderQuad(quad).map(p => [p[0] / scale, p[1] / scale]);
      const panel = comp ? findPanel(closed, comp, Math.abs(polygonArea(quad)), dw, dh) : null;
      const panelFull = panel ? [panel[0] / scale, panel[1] / scale] : null;
      const analysis = orientationAnalysis(data, width, height, corners, FRAME_DST,
        panelFull ? chooseOrientation(corners, panelFull, FRAME_DST) : null);
      return Object.assign({
        mode: 'frame',
        corners: roll(corners, analysis.k),
        orientation: analysis.k,
        ambiguous: analysis.ambiguous,
        orientationFit: analysis.fit,
        oriMargin: analysis.margin,
      }, extra || {});
    };

    let table = null;
    for (const c of tableComps) {
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      if (bw < 0.35 * dw || bh < 0.35 * dh) continue;
      if (c.count / (bw * bh) < 0.18) continue;
      const hull = convexHull(c.points);
      const quad = reduceToQuad(hull);
      if (!quad) continue;
      const sides = quadSides(quad);
      const aspect = Math.min(...sides) / Math.max(...sides);
      if (Math.abs(aspect - FRAME_ASPECT) > 0.12) continue;
      if (Math.abs(polygonArea(quad)) < 0.12 * dw * dh) continue;
      table = fromQuad(quad, c, { confidence: 0.9, tableAnchor: true });
      break;
    }

    if (table) {
      const H = homography(FRAME_DST, table.corners);
      const fit = H ? ringFit(data, width, height, H, [0, 0, 1]) : -1;
      if (fit >= 0.12) {
        table.ringFit = fit;
        return table;
      }
    }

    if (!best) {
      if (page) return page;
      return { mode: 'none', corners: null, orientation: 0, confidence: 0 };
    }

    const conf = 0.6 + 0.3 * Math.min(1, best.area / (0.10 * dw * dh)) + 0.1;
    const framed = fromQuad(best.quad, best.comp, {
      confidence: Math.min(0.99, conf),
      frameAspect: best.aspect,
    });

    // The printed frame sits inside the page, and a pink strip on a shadowed
    // page can look like a small frame.  Taking whichever was found first once
    // warped a piece of the sheet by a factor and read nonsense on some frames.
    // Both candidates are cheap to test: the one whose printed rings actually
    // line up wins.
    if (page) {
      const weigh = (candidate) => {
        const dst = candidate.mode === 'frame' ? FRAME_DST : PAGE_DST;
        const H = homography(dst, candidate.corners);
        if (!H) return -1;
        return ringFit(data, width, height, H, [0, 0, 1]);
      };
      const frameFit = weigh(framed);
      const pageFit = weigh(page);
      // The printed answer table wins ties.  It is the part that is read, and
      // the page outline is exactly the shape a second sheet, a table edge or a
      // notebook can imitate; a shadowed page edge has also dragged the grid a
      // whole option row off.  The page is kept for the case where its rings
      // line up clearly better.
      if (frameFit > (framed.ringFit || 0)) framed.ringFit = frameFit;
      // The printed table is preferred whenever it was found at all.  It is the
      // rectangle that is read, its size on the print is known, and it cannot be
      // confused with a second sheet or the edge of a desk -- while the page
      // outline is a shape anything rectangular can imitate.  A page outline is
      // also what made a sheet printed at 96% need a scale correction that the
      // ring measurement could not see.
      return framed;
    }
    return framed;
  }

  /**
   * Where a template point lands on the photo, including the alignment
   * correction the reader applied.  The overlay used to project with the raw
   * corners and no correction, so a read that had shifted the grid by a
   * millimetre drew its circles a whole bubble away from the answer and looked
   * like a mis-read.
   */
  function mapPagePoint(H, x, y, align) {
    const a = align || [0, 0, 1];
    const s = a[2] || 1;
    return applyH(H,
      GRID_CX + (x + a[0] - GRID_CX) * s,
      GRID_CY + (y + a[1] - GRID_CY) * s);
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

  /**
   * Ring colour contrast: how much pinker the printed ring circumference is
   * than the paper just outside it and the inside of the bubble, averaged over
   * an evenly spread sample of bubbles, scaled to 0..1.  This is the sharp
   * measurement of "is the grid on the print": on the rings it is high, one
   * ring-width away it is negative.  Darkness cannot do this job, because the
   * letters a/b/c/d inside every bubble are dark too.
   */
  /**
   * How pink the printed numbering band is, 0..1.  This is the cheapest honest
   * answer to "how was this sheet printed": the classroom print is pink and
   * scores about 0.10, a grey laser print or a black-and-white render scores
   * about 0.00.  The alignment search needs to know, because a ring should be
   * measured the way it was printed -- dark rings for a grey print, pink rings
   * for a pink one -- and measuring a pink print by darkness lets the letters
   * inside the bubbles drag the grid onto the next row.
   */
  function bandPink(data, width, height, H) {
    const map = H ? (x, y) => applyH(H, x, y) : (x, y) => [x, y];
    let sum = 0, n = 0;
    for (let r = 0; r < ROW_TOPS.length; r++) {
      const top = ROW_TOPS[r] - BAND_TOP_MM * PX_PER_MM;
      for (let y = top + 0.6 * PX_PER_MM; y < top + (BAND_LEN_MM - 0.6) * PX_PER_MM; y += 2) {
        for (let x = GRID_MIN_X + 2 * PX_PER_MM; x < GRID_MAX_X - 2 * PX_PER_MM; x += 5) {
          const p = map(x, y);
          const px = Math.round(p[0]), py = Math.round(p[1]);
          if (!(px > 1 && py > 1 && px < width - 2 && py < height - 2)) continue;
          const i = (py * width + px) * 4;
          const v = data[i] - data[i + 1];
          if (v > 0) sum += v;
          n++;
        }
      }
    }
    return n ? sum / n / 255 : 0;
  }

  function ringContrast(data, width, height, H, offset) {
    const a = offset || [0, 0, 1];
    const s = a[2] || 1;
    const map = H ? (x, y) => applyH(H, x, y) : (x, y) => [x, y];
    const samples = sampleBubbles(24);
    const pinkAt = (px, py) => {
      if (!(px > 1 && py > 1 && px < width - 2 && py < height - 2)) return NaN;
      const i = (Math.round(py) * width + Math.round(px)) * 4;
      return data[i] - data[i + 1];
    };
    let sum = 0, n = 0;
    for (const b of samples) {
      const centre = map(GRID_CX + (b.x + a[0] - GRID_CX) * s,
                         GRID_CY + (b.y + a[1] - GRID_CY) * s);
      let ring = 0, rn = 0;
      for (let k = 0; k < 16; k++) {
        const ang = (k / 16) * Math.PI * 2;
        const v = pinkAt(centre[0] + Math.cos(ang) * b.r * s,
                         centre[1] + Math.sin(ang) * b.r * s);
        if (!isNaN(v)) { ring += v; rn++; }
      }
      const inner = pinkAt(centre[0], centre[1]);
      const outer = pinkAt(centre[0], centre[1] + b.r * 1.9 * s);
      if (!rn || isNaN(inner) || isNaN(outer)) continue;
      sum += (ring / rn) - Math.max(inner, outer);
      n++;
    }
    return n ? Math.max(0, sum / n) / 255 : 0;
  }

  /**
   * The printed ring of every bubble is the strongest signal on the sheet, so
   * the displacement that lines the rings up with the template is the right one.
   * A phone photo is routinely a millimetre or two out — a third of a bubble —
   * and this finds that offset before anything is measured, instead of letting
   * it eat into every score.
   */
  function alignOffset(data, width, height, H, searchPx, stepPx, options) {
    const fast = Boolean(options && options.fast);
    const reach = searchPx || (typeof window.REACH_K === 'number' ? window.REACH_K : 28);
    // A live frame has a whole second of budget for the *next* frame too, so the
    // search starts coarse (fewer print samples, a wider step) and refines only
    // around the winner.  The photo path keeps the full-resolution search.
    const step = fast ? Math.max(stepPx || 0, Math.round(reach / 5)) : (stepPx || 4);
    const map = H ? (x, y) => applyH(H, x, y) : (x, y) => [x, y];
    const sample = sampleBubbles(fast ? 12 : 20);

    // Which way was this sheet printed?  A pink print is matched by its pink
    // rings and nothing else: on that print every shift or rescale driven by
    // darkness lands on the letters inside the bubbles, which is measurably
    // darker than the ring, so the search slides the grid onto the next row and
    // the sheet reads one row off.  A grey print has no colour to measure, and
    // there darkness is exactly the right signal.
    const pinkPrint = bandPink(data, width, height, H) > 0.05;
    const inkAt = (px, py, rad) => {
      let best = -1;
      for (let oy = -rad; oy <= rad; oy++) {
        for (let ox = -rad; ox <= rad; ox++) {
          const x = px + ox, y = py + oy;
          if (!(x > 1 && y > 1 && x < width - 2 && y < height - 2)) continue;
          const i = (y * width + x) * 4;
          const value = 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          if (value > best) best = value;
        }
      }
      return best;
    };
    const inkScoreAt = (dx, dy, s) => {
      let sum = 0, n = 0;
      for (const b of sample) {
        for (let k = 0; k < 8; k++) {
          const a = (k / 8) * Math.PI * 2;
          const cos = Math.cos(a), sin = Math.sin(a);
          for (const f of [0.95, 1.05]) {
            const lx = b.x + dx + cos * b.r * f;
            const ly = b.y + dy + sin * b.r * f;
            const p = map(
              GRID_CX + (lx - GRID_CX) * s,
              GRID_CY + (ly - GRID_CY) * s
            );
            const ink = inkAt(Math.round(p[0]), Math.round(p[1]), 2);
            if (ink < 0) continue;
            sum += ink;
            n++;
          }
        }
      }
      return n ? sum / n : 0;
    };
    const scoreAt = pinkPrint
      ? (dx, dy, s) => ringContrast(data, width, height, H, [dx, dy, s]) * 255
      : inkScoreAt;

    // Three things move a printed sheet away from its template: the photo is a
    // little off, the page sits at a slight angle, or the printer scaled the
    // page ("fit to page" at 95%).  The first two are a translation, the third
    // is a scale, so both are searched — coarse, then fine around the winner.
    const scales = fast
      ? [1, 0.97, 1.03, 0.94, 1.06]
      : [1, 0.985, 0.97, 0.955, 0.94, 1.015, 1.03, 1.045];
    // Occam bias: a large displacement must beat the home position clearly,
    // not by a whisker.  Without this a small real offset is retried for free,
    // while a mis-lock (the next band of ink, say) needs a real gain; the bias
    // is scaled against the reach so a wide retry pays proportionally more.
    const bias = (dx, dy) => BIAS_K * (Math.abs(dx) + Math.abs(dy)) / reach;
    void 0;
    let best = null;
    let total = 0;
    let count = 0;
    for (const s of scales) {
      for (let dy = -reach; dy <= reach; dy += step) {
        for (let dx = -reach; dx <= reach; dx += step) {
          const score = scoreAt(dx, dy, s);
          total += score;
          count++;
          const adj = score - bias(dx, dy);
          if (!best || adj > best.adj) best = { adj, score, dx, dy, s };
        }
      }
    }
    if (!best) return [0, 0, 1];
    const mean = count ? total / count : 0;
    // A flat field of ink means there is no ring to line up with, so stay put.
    if (best.score < mean * 1.05 || best.score < 6) return [0, 0, 1];

    for (const step of [2, 1]) {
      for (let ds = -1; ds <= 1; ds++) {
        const s = Math.round((best.s + ds * 0.0075) * 1000) / 1000;
        for (let dy = best.dy - 2; dy <= best.dy + 2; dy += step) {
            for (let dx = best.dx - 2; dx <= best.dx + 2; dx += step) {
              const score = scoreAt(dx, dy, s);
            const adj = score - bias(dx, dy);
            if (!best.adj || adj > best.adj) best = { adj, score, dx, dy, s };
          }
        }
      }
    }
    return [best.dx, best.dy, best.s];
  }

  /**
   * How well the printed bubble rings line up under a given correction, from 0
   * (nothing but paper) to about 0.9 (black rings).  This is the confidence
   * measure: a sheet that lines up reads correctly, and one that does not is
   * reported instead of being graded from the wrong places.
   */


  function ringFit(data, width, height, H, offset) {
    const off = offset || [0, 0, 1];
    const s = off[2] || 1;
    const map = H ? (x, y) => applyH(H, x, y) : (x, y) => [x, y];
    // The printed ring is barely a pixel wide once the photo is scaled down, so
    // every sample takes the darkest pixel in a 3x3 window: a one-pixel error in
    // the correction must not read as "no ring here".
    const ink = (px, py) => {
      let best = -1;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const x = px + ox, y = py + oy;
          if (!(x > 0 && y > 0 && x < width - 1 && y < height - 1)) continue;
          const i = (y * width + x) * 4;
          const value = 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          if (value > best) best = value;
        }
      }
      return best;
    };
    const vals = [];
    for (const b of sampleBubbles(24)) {
      const pts = [];
      for (let k = 0; k < 12; k++) {
        const a = (k / 12) * Math.PI * 2;
        const cos = Math.cos(a), sin = Math.sin(a);
        for (const f of [0.95, 1.05]) {
          const lx = b.x + off[0] + cos * b.r * f;
          const ly = b.y + off[1] + sin * b.r * f;
          const p = map(
            GRID_CX + (lx - GRID_CX) * s,
            GRID_CY + (ly - GRID_CY) * s
          );
          const value = ink(Math.round(p[0]), Math.round(p[1]));
          if (value >= 0) pts.push(value);
        }
      }
      if (pts.length) vals.push(percentile(pts, 0.7));
    }
    return vals.length ? median(vals) / 255 : 0;
  }

  function readBubbles(data, width, height, offset) {
    const off = offset || [0, 0];
    const scale = off[2] || 1;
    let cx = 0, cy = 0;
    for (const b of BUBBLES) { cx += b.x; cy += b.y; }
    cx /= BUBBLES.length;
    cy /= BUBBLES.length;

    const scores = [];
    const rings = [];
    const readings = [];

    // A warped photo never lands the whole grid exactly, and the residual error
    // varies across the sheet: some cells sit half a millimetre off while their
    // neighbours are perfect.  A 3x3 micro-search (plus the centre) finds the
    // ink of this bubble; the reach is a third of the bubble radius, far too
    // small to ever touch a neighbouring bubble.
    const peek = (px, py, br, ox, oy) => {
      let sum = 0;
      for (let k = 0; k < 9; k++) {
        const a = (k / 9) * Math.PI * 2;
        const f = k === 8 ? 0 : 0.34;
        const x = Math.round(px + ox + Math.cos(a) * br * f);
        const y = Math.round(py + oy + Math.sin(a) * br * f);
        if (!(x > 0 && y > 0 && x < width - 1 && y < height - 1)) return -1;
        const i = (y * width + x) * 4;
        sum += 255 - Math.min(data[i], data[i + 1], data[i + 2]);
      }
      return sum / 9;
    };

    for (const b of BUBBLES) {
      let bx = cx + (b.x - cx) * scale + off[0];
      let by = cy + (b.y - cy) * scale + off[1];
      const br = b.r * scale;
      const centreInk = peek(bx, by, br, 0, 0);
      if (centreInk >= 0) {
        const step = Math.max(1, Math.round(br * 0.12));
        let bestInk = centreInk, bestOx = 0, bestOy = 0;
        for (let oy = -step; oy <= step; oy++) {
          for (let ox = -step; ox <= step; ox++) {
            if (!ox && !oy) continue;
            const ink = peek(bx, by, br, ox, oy);
            if (ink > bestInk) { bestInk = ink; bestOx = ox; bestOy = oy; }
          }
        }
        if (bestInk > centreInk + 12) { bx += bestOx; by += bestOy; }
      }
      const inner = [], ann = [], outer = [];
      const x0 = Math.max(0, Math.floor(bx - br * 1.6));
      const x1 = Math.min(width - 1, Math.ceil(bx + br * 1.6));
      const y0 = Math.max(0, Math.floor(by - br * 1.6));
      const y1 = Math.min(height - 1, Math.ceil(by + br * 1.6));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x + 0.5 - bx, dy = y + 0.5 - by;
          const d2 = dx * dx + dy * dy;
          if (d2 > (1.60 * br) ** 2) continue;
          const i = (y * width + x) * 4;
          const ink = 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          if (d2 <= (0.52 * br) ** 2) inner.push(ink);
          else if (d2 >= (0.85 * br) ** 2 && d2 <= (1.15 * br) ** 2) ann.push(ink);
          else if (d2 >= (1.35 * br) ** 2) outer.push(ink);
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
    return classifyReadings(readings, ringBaseline);
  }

  /**
   * Turns per-bubble scores into MARKED / EMPTY / UNCERTAIN states and one
   * answer per question.  Shared by the still-photo path and the live path so
   * both behave the same for every sheet size.
   */
  function classifyReadings(readings, ringBaseline) {
    const scores = readings.map(r => r.score);
    const cal = calibrate(scores, ringBaseline);
    const threshold = cal.threshold;

    for (const r of readings) {
      if (r.state) continue; // a fused frame has already decided
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

    return {
      questions: questionsFromReadings(readings),
      bubbles: readings,
      calibration: cal,
      diagnostics: {
        ringBaseline,
        scoreMin: scores.length ? Math.min(...scores) : 0,
        scoreMax: scores.length ? Math.max(...scores) : 0,
        markedBubbles: readings.filter(r => r.state === 'MARKED').length,
      },
    };
  }

  function questionsFromReadings(readings) {
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
        // Faint pen in a shadowed photo lands here: the bubble has ink, but not
        // enough to pass the strict mark gate.  When exactly one bubble of the
        // question is uncertain and clearly out-scores its own row, it is that
        // question's answer -- grading it as blank threw away real answers on
        // real classroom photos.  It stays flagged for the teacher to check.
        const sorted = rs.slice().sort((a, b) => b.score - a.score);
        const top = sorted[0], next = sorted[1];
        const decisive = top && top.score >= 0.10 && top.fill >= 0.20
          && (!next || top.score >= next.score * 1.6);
        if (decisive && uncertain.length === 1) {
          state = 'REVIEW';
          selected = [top.option];
        } else {
          state = 'REVIEW';
        }
      }
      questions[q] = { q, state, selected };
    }
    return questions;
  }

  /** Marks for question `index` (0-based): a number, an array or a map. */
  function markFor(marks, index) {
    if (Array.isArray(marks)) {
      const value = Number(marks[index]);
      return Number.isFinite(value) && value >= 0 ? value : 1;
    }
    if (marks && typeof marks === 'object') {
      const value = Number(marks[index + 1]);
      return Number.isFinite(value) && value >= 0 ? value : 1;
    }
    const value = Number(marks);
    return Number.isFinite(value) && value > 0 ? value : 1;
  }

  function gradeRead(read, key, marks) {
    const details = [];
    let correct = 0, wrong = 0, blank = 0, multi = 0, review = 0, noKey = 0;
    let total = 0, maximum = 0;
    // One printed sheet holds 60 slots; an exam may use fewer.  The key says
    // how many questions this exam has, and only those are graded.
    const graded = key && key.length
      ? Math.min(QUESTION_COUNT, key.length) : QUESTION_COUNT;
    for (let q = 1; q <= graded; q++) {
      const r = read.questions[q] || { q, state: 'BLANK', selected: [] };
      const answer = key && Number.isInteger(key[q - 1]) ? key[q - 1] : null;
      const mark = markFor(marks, q - 1);
      let outcome;
      if (answer === null) outcome = 'NO_KEY';
      else if (r.state === 'ANSWERED') outcome = r.selected[0] === answer ? 'CORRECT' : 'WRONG';
      // A low-confidence read that still lands on the key's answer is a correct
      // answer: the flag is there so a teacher can look at a doubtful sheet, not
      // to throw a right answer away.
      else if (r.state === 'REVIEW' && r.selected.length === 1 && r.selected[0] === answer) outcome = 'CORRECT';
      else if (r.state === 'MULTI') outcome = 'MULTI';
      else if (r.state === 'BLANK') outcome = 'BLANK';
      else outcome = 'REVIEW';
      if (answer !== null) maximum += mark;
      if (outcome === 'CORRECT') { correct++; total += mark; }
      else if (outcome === 'WRONG') wrong++;
      else if (outcome === 'BLANK') blank++;
      else if (outcome === 'MULTI') multi++;
      else if (outcome === 'REVIEW') review++;
      else noKey++;
      details.push({ q, selected: r.selected, answer, state: r.state, outcome, mark });
    }
    return {
      correct, wrong, blank, multi, review, noKey,
      total: Math.round(total * 100) / 100,
      maximum: Math.round(maximum * 100) / 100,
      marksPerQuestion: Array.isArray(marks) ? marks.slice() : marks,
      questionCount: graded,
      details,
    };
  }

  /**
   * Reads the bubbles straight out of the camera frame: every bubble centre is
   * pushed through the sheet homography and sampled in place, instead of
   * warping a whole 1082x1530 page.  That is what makes multi-frame smoothing
   * affordable on a phone.
   */
  function readProjected(data, width, height, H, offset) {
    if (!H) return null;
    const off = offset || [0, 0];
    const scale = off[2] || 1;
    let cx = 0, cy = 0;
    for (const b of BUBBLES) { cx += b.x; cy += b.y; }
    cx /= BUBBLES.length;
    cy /= BUBBLES.length;
    const place = (b) => [
      cx + (b.x - cx) * scale + off[0],
      cy + (b.y - cy) * scale + off[1],
    ];

    // A geometry that does not even place the first bubble inside the frame is
    // wrong; measuring it would only produce noise.
    const first = place(BUBBLES[0]);
    const probe = applyH(H, first[0], first[1]);
    if (!(probe[0] >= 0 && probe[1] >= 0 && probe[0] < width && probe[1] < height)) return null;
    const probeEdge = applyH(H, first[0] + BUBBLES[0].r * scale, first[1]);
    const probeRadius = Math.hypot(probeEdge[0] - probe[0], probeEdge[1] - probe[1]);
    if (!(probeRadius >= 1.6)) return null; // bubbles too small: move closer

    const readings = [];
    for (const b of BUBBLES) {
      const p = place(b);
      const r = b.r * scale;
      const center = applyH(H, p[0], p[1]);
      const edgeX = applyH(H, p[0] + r, p[1]);
      const edgeY = applyH(H, p[0], p[1] + r);
      const radiusPx = 0.5 * (
        Math.hypot(edgeX[0] - center[0], edgeX[1] - center[1]) +
        Math.hypot(edgeY[0] - center[0], edgeY[1] - center[1])
      );
      if (!(center[0] >= 0 && center[1] >= 0 && center[0] < width && center[1] < height)) {
        return null;
      }

      // Same micro-search as the page reader: in live view the hand moves, so
      // the projected grid is never exactly where the template says.  Sampling
      // a ring of nine points a third of a bubble out lets the measurement find
      // this bubble's ink without ever reaching a neighbouring bubble.
      let cxp = center[0], cyp = center[1];
      {
        const peek = (ox, oy) => {
          let sum = 0;
          for (let k = 0; k < 9; k++) {
            const a = (k / 9) * Math.PI * 2;
            const f = k === 8 ? 0 : 0.34;
            const x = Math.round(cxp + ox + Math.cos(a) * radiusPx * f);
            const y = Math.round(cyp + oy + Math.sin(a) * radiusPx * f);
            if (!(x > 0 && y > 0 && x < width - 1 && y < height - 1)) return -1;
            const i = (y * width + x) * 4;
            sum += 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          }
          return sum / 9;
        };
        const step = Math.max(1, Math.round(radiusPx * 0.14));
        const base = peek(0, 0);
        if (base >= 0) {
          let bestInk = base, bestOx = 0, bestOy = 0;
          for (let oy = -step; oy <= step; oy++) {
            for (let ox = -step; ox <= step; ox++) {
              if (!ox && !oy) continue;
              const ink = peek(ox, oy);
              if (ink > bestInk) { bestInk = ink; bestOx = ox; bestOy = oy; }
            }
          }
          if (bestInk > base + 12) { cxp += bestOx; cyp += bestOy; }
        }
      }
      const inner = [], ann = [], outer = [];
      const x0 = Math.max(0, Math.floor(cxp - radiusPx * 1.6));
      const x1 = Math.min(width - 1, Math.ceil(cxp + radiusPx * 1.6));
      const y0 = Math.max(0, Math.floor(cyp - radiusPx * 1.6));
      const y1 = Math.min(height - 1, Math.ceil(cyp + radiusPx * 1.6));
      if (x1 <= x0 || y1 <= y0) return null;

      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x + 0.5 - cxp, dy = y + 0.5 - cyp;
          const d2 = dx * dx + dy * dy;
          if (d2 > (1.60 * radiusPx) ** 2) continue;
          const i = (y * width + x) * 4;
          const ink = 255 - Math.min(data[i], data[i + 1], data[i + 2]);
          if (d2 <= (0.52 * radiusPx) ** 2) inner.push(ink);
          else if (d2 >= (0.85 * radiusPx) ** 2 && d2 <= (1.15 * radiusPx) ** 2) ann.push(ink);
          else if (d2 >= (1.35 * radiusPx) ** 2) outer.push(ink);
        }
      }
      if (!inner.length) {
        // Keep one reading per bubble so indices stay aligned with the fusion
        // state; a clipped bubble simply reads as empty.
        readings.push({
          q: b.q, option: b.opt, score: 0, fill: 0, ring: 0, paper: 0,
          darkFrac: 0, radiusPx,
        });
        continue;
      }

      const paper = outer.length ? median(outer) : 0;
      const fillRel = Math.max(0, (median(inner) - paper) / 255);
      const darkFrac = inner.filter(v => v > paper + 70).length / inner.length;
      readings.push({
        q: b.q,
        option: b.opt,
        score: 0.70 * fillRel + 0.30 * darkFrac,
        fill: median(inner) / 255,
        ring: ann.length ? percentile(ann, 0.75) / 255 : 0,
        paper: paper / 255,
        darkFrac,
        radiusPx,
      });
    }
    return readings;
  }

  /**
   * Temporal fusion for the live camera.  Every bubble keeps the last `window`
   * frames of its score and reports the median, so a shaken, blurred or
   * half-aligned frame is outvoted instead of dragging an average along with
   * it.  A decision still has to be pushed the same way for `agree` frames in a
   * row before it is taken, and the sheet is only settled once nothing has
   * changed for a couple of frames.
   *
   * The cut between "inked" and "paper" comes from every frame's own score
   * spread, the same calibration the photo reader uses.  A fixed cut cannot
   * work on a phone: a mark seen from arm's length scores far lower than a
   * flatbed scan, so a hard threshold would leave every bubble empty and the
   * result would read zero.
   */
  function createFusion(options) {
    const opts = options || {};
    const need = opts.need || 7;
    const agree = opts.agree || 3;
    const window = Math.max(3, opts.window || 9);
    // How much darker than the paper a bubble must be to count as inked.  This
    // is a *difference*, not a brightness: a classroom camera in dim light, a
    // pink print and a black print all put the printed paper at very different
    // levels, and an absolute floor silently turned real marks off whenever the
    // light or the focus changed mid-scan.
    const inkFloor = opts.inkFloor === undefined ? 0.10 : opts.inkFloor;

    const states = BUBBLES.map(b => ({
      b, scores: [], fills: [], score: 0, fill: 0, marked: false, streak: 0,
    }));
    const papers = [];
    let frames = 0;
    let lastChange = 0;
    let threshold = null;
    let skipped = 0;

    function push(readings) {
      if (!readings || readings.length !== states.length) return false;

      const scores = readings.map(r => r.score);
      const rings = readings.map(r => r.ring || 0);
      // Most of the sheet is empty paper, so the middle fill of a frame *is* the
      // paper level for that frame; every ink test below is measured against it.
      const paperFill = median(readings.map(r => (r.fill === undefined ? r.score : r.fill)));
      papers.push(paperFill);
      if (papers.length > window) papers.shift();
      const frameCut = calibrate(scores, median(rings)).threshold;
      threshold = threshold === null
        ? frameCut
        : threshold + (frameCut - threshold) * 0.35;
      const hi = Math.max(0.16, threshold * 1.15);
      const lo = Math.max(0.09, Math.min(hi - 0.04, threshold * 0.85));

      // A frame that disagrees with the sheet already being read is shaken,
      // half-aligned or blurred: throwing it away beats letting it into the
      // window.  A run of rejects means the sheet really did change, so the
      // history starts again instead of blocking forever.
      if (frames >= 3) {
        let disagree = 0;
        for (let i = 0; i < states.length; i++) {
          if ((readings[i].score >= hi) !== states[i].marked) disagree++;
        }
        if (disagree > states.length * 0.25) {
          if (++skipped < 6) return false;
          skipped = 0;
          frames = 0;
          lastChange = 0;
          for (const st of states) {
            st.scores.length = 0;
            st.fills.length = 0;
            st.score = 0;
            st.fill = 0;
            st.streak = 0;
          }
        } else {
          skipped = 0;
        }
      }

      let changed = false;
      for (let i = 0; i < states.length; i++) {
        const st = states[i];
        const reading = readings[i];
        const fill = reading.fill === undefined ? reading.score : reading.fill;

        st.scores.push(reading.score);
        st.fills.push(fill);
        if (st.scores.length > window) st.scores.shift();
        if (st.fills.length > window) st.fills.shift();
        st.score = median(st.scores);
        st.fill = median(st.fills);

        // Two frames are not enough to decide anything: wait for the median to
        // mean something before the first verdict.
        if (st.scores.length < 3) continue;

        const paper = median(papers);
        let next = st.marked;
        if (st.score >= hi && st.fill - paper >= inkFloor) next = true;
        else if (st.score <= lo) next = false;
        if (next !== st.marked) {
          if (++st.streak >= agree) {
            st.marked = next;
            st.streak = 0;
            changed = true;
          }
        } else {
          st.streak = 0;
        }
      }
      frames++;
      if (changed) lastChange = frames;
      return true;
    }

    function readings() {
      return states.map(st => ({
        q: st.b.q,
        option: st.b.opt,
        score: st.score,
        fill: st.fill,
        ring: 0,
        paper: 0,
        darkFrac: 0,
        state: st.marked ? 'MARKED' : 'EMPTY',
      }));
    }

    return {
      push,
      readings,
      reset() {
        frames = 0;
        lastChange = 0;
        threshold = null;
        skipped = 0;
        papers.length = 0;
        for (const st of states) {
          st.scores.length = 0;
          st.fills.length = 0;
          st.score = 0;
          st.fill = 0;
          st.marked = false;
          st.streak = 0;
        }
      },
      get frames() { return frames; },
      get window() { return window; },
      get threshold() { return threshold; },
      isSettled() { return frames >= need && frames - lastChange >= 2; },
      progress() { return Math.min(1, frames / need); },
    };
  }

  /** Resamples `box` out of the frame, optionally enlarged (bilinear). */
  function cropBox(data, width, height, box, scale) {
    const outW = Math.max(1, Math.round(box.w * scale));
    const outH = Math.max(1, Math.round(box.h * scale));
    const out = new Uint8ClampedArray(outW * outH * 4);
    for (let y = 0; y < outH; y++) {
      const sy = box.y + (y + 0.5) / scale - 0.5;
      const y0 = Math.max(0, Math.min(height - 1, Math.floor(sy)));
      const y1 = Math.min(height - 1, y0 + 1);
      const fy = Math.max(0, Math.min(1, sy - y0));
      for (let x = 0; x < outW; x++) {
        const sx = box.x + (x + 0.5) / scale - 0.5;
        const x0 = Math.max(0, Math.min(width - 1, Math.floor(sx)));
        const x1 = Math.min(width - 1, x0 + 1);
        const fx = Math.max(0, Math.min(1, sx - x0));
        const o = (y * outW + x) * 4;
        const i00 = (y0 * width + x0) * 4, i10 = (y0 * width + x1) * 4;
        const i01 = (y1 * width + x0) * 4, i11 = (y1 * width + x1) * 4;
        for (let c = 0; c < 4; c++) {
          const top = data[i00 + c] * (1 - fx) + data[i10 + c] * fx;
          const bot = data[i01 + c] * (1 - fx) + data[i11 + c] * fx;
          out[o + c] = top * (1 - fy) + bot * fy;
        }
      }
    }
    return { data: out, width: outW, height: outH };
  }

  /**
   * Reads the answer area only.  The bubble block is cropped out of the frame
   * and enlarged when the sheet is far away, so every bubble is measured on
   * more pixels than a full-frame read would give it.  Everything outside the
   * block (title, instructions, handwriting) is never looked at.
   */
  // Correction carried from one live frame to the next; a hand that is not
  // moving keeps the same residual, so the search usually does not need to run.
  let lastAlign = null;
  const fastStep = 4;

  function readRegion(data, width, height, H, options) {
    if (!H) return null;
    const opts = options || {};
    const region = opts.region || omrRegion(opts.padMm);
    const page = regionCorners(region).map(p => applyH(H, p[0], p[1]));
    for (const p of page) {
      if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return null;
      if (p[0] < 0 || p[1] < 0 || p[0] > width || p[1] > height) return null;
    }

    // How big a printed bubble is in this frame decides whether the answer area
    // needs enlarging at all.  A sheet held at arm's length has eight to ten
    // pixels per bubble: enough to measure straight from the frame, which takes
    // a full resample of the whole area out of every live frame.  Only a sheet
    // far away is cropped and enlarged the way it used to be.
    let radiusPx = 0;
    let sumX = 0, sumY = 0;
    for (const b of BUBBLES) { sumX += b.x; sumY += b.y; }
    const gridX = sumX / BUBBLES.length, gridY = sumY / BUBBLES.length;
    const centre = applyH(H, gridX, gridY);
    const bubbleR = BUBBLES.length ? BUBBLES[0].r : 21;
    for (const edge of [applyH(H, gridX + bubbleR, gridY), applyH(H, gridX, gridY + bubbleR)]) {
      radiusPx = Math.max(radiusPx, Math.hypot(edge[0] - centre[0], edge[1] - centre[1]));
    }

    let src = data, srcW = width, srcH = height, useH = H, scale = 1, box = null;
    if (radiusPx < (opts.minRadiusPx || 3.4)) {
      const margin = opts.marginPx === undefined ? 3 : opts.marginPx;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of page) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
      box = {
        x: Math.max(0, Math.floor(minX - margin)),
        y: Math.max(0, Math.floor(minY - margin)),
      };
      box.w = Math.min(width - box.x, Math.ceil(maxX - minX + margin * 2));
      box.h = Math.min(height - box.y, Math.ceil(maxY - minY + margin * 2));
      if (box.w < 8 || box.h < 8) return null;
      const maxSide = opts.maxSide || 1000;
      scale = Math.max(1, Math.min(3, maxSide / Math.max(box.w, box.h)));
      const crop = cropBox(data, width, height, box, scale);
      const cropQuad = page.map(p => [(p[0] - box.x) * scale, (p[1] - box.y) * scale]);
      const cropH = homography(regionCorners(region), cropQuad);
      if (!cropH) return null;
      src = crop.data; srcW = crop.width; srcH = crop.height;
      useH = cropH;
      radiusPx *= scale;
    }

    const still = [0, 0, 1];
    let align = still;
    let reading = readProjected(src, srcW, srcH, useH, still);
    let fit = ringFit(src, srcW, srcH, useH, still);

    // "Does the grid sit on the printed rings?" cannot be answered by the ring
    // ink alone: a pink classroom print scores a third of a black one, so an
    // absolute cut either kept searching on a perfect frame -- a second per
    // frame, which is what made live scanning feel dead -- or accepted a bad
    // one.  Nor can a probe a bubble away: the printed lattice is dense (9 mm
    // between bubbles, 3.6 mm rings), so almost any shift lands on *some* ring.
    // Nudging by a third of a ring asks the question that matters -- is this
    // position a local peak of ring ink? -- and answers it the same way for a
    // pink print, a black one, or a shadowed photo.
    const nudge = 1.4 * PX_PER_MM;
    const nudgeFit = Math.max(
      ringFit(src, srcW, srcH, useH, [nudge, 0, 1]),
      ringFit(src, srcW, srcH, useH, [0, nudge, 1])
    );
    const onRings = fit >= 0.05 && fit >= nudgeFit * 1.20;
    if (!onRings) {
      // While a sheet is held still the same correction keeps working, so try
      // the one that worked on the previous frame before searching from scratch.
      let candidate = null;
      if (lastAlign && (lastAlign[0] || lastAlign[1] || lastAlign[2] !== 1)) {
        const reuseFit = ringFit(src, srcW, srcH, useH, lastAlign);
        if (reuseFit > fit) { candidate = lastAlign; fit = reuseFit; }
      }
      if (!candidate) {
        candidate = alignOffset(src, srcW, srcH, useH,
          Math.max(8, Math.round(radiusPx * 2.2)), fastStep, { fast: true });
        lastAlign = candidate;
      }
      if (candidate[0] || candidate[1] || candidate[2] !== 1) {
        const moved = readProjected(src, srcW, srcH, useH, candidate);
        const movedFit = ringFit(src, srcW, srcH, useH, candidate);
        if (moved && movedFit > fit + 0.01) {
          align = candidate;
          reading = moved;
          fit = movedFit;
        }
      }
    }
    const readings = reading;
    if (!readings) return null;
    return {
      readings,
      align,
      ringFit: fit,
      nudgeFit,
      onRings,
      region,
      radiusPx,
      crop: box ? { width: srcW, height: srcH, scale, box } : null,
    };
  }


  function scanRGBA(data, width, height, key, marksPerQuestion, precomputedFrame) {
    const frame = precomputedFrame || detectFrame(data, width, height);
    if (frame.mode === 'none') {
      return {
        ok: false,
        error: 'Could not find the sheet in this photo. Put the whole answer area in the picture, with the pink border visible, and try again.',
      };
    }

    // The page could be graded from two different turns of the sheet.  A wrong
    // turn reads marks from the wrong bubbles, which is worse than no answer,
    // so refuse until the sheet is laid the usual way up.
    if (frame.ambiguous) {
      return {
        ok: false,
        error: 'The sheet was found but the scan is not sure which way the page'
          + ' is turned. Put the sheet the right way up \u2014 top of the page at the'
          + ' top of the photo \u2014 and try again.',
      };
    }

    const attempt = (mode, searchPx) => {
      const warped = warpPage(
        data, width, height, frame.corners,
        mode === 'frame' ? FRAME_DST : PAGE_DST,
        PAGE_W, PAGE_H
      );
      const markedIn = (r) =>
        (r.bubbles || []).filter((b) => b.state === 'MARKED').length;

      // Line the printed rings up with the template before anything is
      // measured, and keep the plain read only when it fits better.  The score
      // is ring contrast, not the number of marks, so an empty sheet and a full
      // one are judged the same way.
      let align = [0, 0, 1];
      let read = readBubbles(warped, PAGE_W, PAGE_H, align);
      let fit = ringFit(warped, PAGE_W, PAGE_H, null, align);
      const candidate = alignOffset(warped, PAGE_W, PAGE_H, null,
        searchPx || 16, searchPx ? 8 : 4);
      if (candidate[0] || candidate[1] || candidate[2] !== 1) {
        const shifted = readBubbles(warped, PAGE_W, PAGE_H, candidate);
        const shiftedFit = ringFit(warped, PAGE_W, PAGE_H, null, candidate);
        if (shiftedFit > fit + 0.01) {
          align = candidate;
          read = shifted;
          fit = shiftedFit;
        }
      }
      const bubbles = read.bubbles || [];
      let marked = 0;
      for (const bubble of bubbles) if (bubble.state === 'MARKED') marked++;
      return {
        mode,
        warped,
        align,
        read,
        fit,
        grade: gradeRead(read, key, marksPerQuestion),
        marked,
      };
    };

    const primary = frame.mode === 'frame' ? 'frame' : 'page';
    let best = attempt(primary);
    // The detected quad can only be read against the destination it really is.
    // On this design the printed frame is 145 x 180 mm of a 210 x 297 mm page,
    // so warping a page outline onto the frame rectangle rescales the whole
    // sheet by ~1.45 and finds "rings" everywhere; that wrong mapping once beat
    // the right one and graded a correctly marked sheet at 21.  The other
    // destination is tried only when the quad's own aspect fits it too.
    if (frame.mode !== 'flat') {
      const sides = quadSides(orderQuad(frame.corners));
      const aspect = Math.min(...sides) / Math.max(...sides);
      const pageAspect = Math.min(PAGE_W, PAGE_H) / Math.max(PAGE_W, PAGE_H);
      const otherAspect = primary === 'frame' ? pageAspect : FRAME_ASPECT;
      if (Math.abs(aspect - otherAspect) < 0.02) {
        const other = attempt(primary === 'frame' ? 'page' : 'frame');
        if (other.fit > best.fit + 0.02) best = other;
        else if (Math.abs(other.fit - best.fit) <= 0.02 && other.marked > best.marked) best = other;
      }
    }

    // A strongly angled phone or a quad that is a few millimetres wrong puts the
    // grid further out than the normal search reaches.  Before refusing the
    // photo, search far and coarse; only a genuinely better ring fit is kept.
    if (frame.mode !== 'flat' && best.fit < 0.15) {
      const widened = attempt(best.mode, 48);
      if (widened.fit > best.fit + 0.02) best = widened;
    }

    // Nothing lined up: the bubbles are not where the sheet says they are, so
    // any total would be guesswork.  Say so instead of grading the wrong bubbles.
    if (frame.mode !== 'flat' && best.fit < 0.10) {
      return {
        ok: false,
        error: 'The sheet was found but the bubbles did not line up. Keep the whole'
          + ' answer area in the photo, straighten the paper, and print the sheet at'
          + ' 100% size (never \u201Cfit to page\u201D).',
      };
    }

    return {
      ok: true,
      mode: best.mode,
      confidence: frame.confidence,
      orientation: frame.orientation,
      oriMargin: frame.oriMargin,
      corners: frame.corners,
      warp: { data: best.warped, width: PAGE_W, height: PAGE_H },
      align: best.align,
      ringFit: best.fit,
      read: best.read,
      grade: best.grade,
    };
  }

  return {
    template: T,
    QUESTION_COUNT,
    OPTION_COUNT,
    scanRGBA,
    detectFrame,
    orientationAnalysis,
    bandAsymmetry,
    readBubbles,
    alignOffset,
    ringFit,
    ringContrast,
    bandPink,
    readProjected,
    readRegion,
    omrRegion,
    calibrate,
    classifyReadings,
    questionsFromReadings,
    createFusion,
    gradeRead,
    homography,
    applyH,
    warpPage,
    bubblePx,
    BUBBLES,
    FRAME_DST,
    PAGE_DST,
    GUIDE_DST,
    GUIDE_MM,
    mapPagePoint,
    PAGE_W,
    PAGE_H,
    DPI,
    OPTION_LABELS,
  };
  }

  root.OMREngineFactory = { createEngine };
  root.OMREngine = root.OMR_ACTIVE_TEMPLATE
    ? createEngine(root.OMR_ACTIVE_TEMPLATE)
    : null;
})(typeof window !== 'undefined' ? window : globalThis);
