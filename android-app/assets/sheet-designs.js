/*
 * The one production answer sheet: print/v3.pdf.
 *
 * Every number below was measured from that page (A4, millimetres): 15
 * question columns per row, four bubbles A..D stacked in each column, four
 * rows of 15 = 60 question slots.  This is the only design the app prints and
 * the only design it reads.  An exam with fewer than 60 questions uses the top
 * rows of the same printed sheet, and the app grades only those questions.
 */
(function (root) {
  'use strict';

  const LABELS = ['A', 'B', 'C', 'D'];
  const round = (value) => Math.round(value * 1000) / 1000;

  const DESIGN = {
    page: [210, 297],                 // A4
    perRow: 15,
    optionCount: 4,

    // Column centres, left to right.  The pitch alternates slightly (9.54 /
    // 9.80 mm) because the printed table has a border between every column, so
    // the measured centres are kept instead of an idealised pitch.
    colX: [
      13.21, 22.86, 32.51, 42.25, 51.90,
      61.56, 71.21, 80.86, 90.59, 100.25,
      109.90, 119.55, 129.20, 138.85, 148.59,
    ],
    // First option (A) of each question row, top to bottom.  Recalibrated
    // against scans of the printed school sheet (the q-paper samples): the
    // block line A sits at 74.1 mm, 9.01 mm between option rows, 43.4 mm
    // between blocks.  The printed numbers band spans ~11.4 mm to ~4.9 mm
    // above each row-A line, which the reader's band witness samples.
    // Measured on print/v3.pdf: option-A centres 78.74 / 122.09 / 165.61 /
    // 208.96 mm, from the printed band lines (band bottom + 5.08 mm).
    rowY: [78.74, 122.09, 165.61, 208.96],
    optionPitch: 9.0,                 // A to B, downwards
    radius: 3.56,
    cellW: 9.6735,

    gridTop: 64.35,                   // top border of the answer table
    gridBottom: 244.01,               // bottom border of the answer table
    bandHeight: 7.88,                 // numbered band above each row
    bandGap: 1.52,

    right: { x: 157.82, w: 43.86 },   // Student ID / Subject Code / Total Marks

    // The printed sheet always carries 60 question slots ...
    counts: [60],
    // ... and an exam may ask any of these numbers of questions on it.
    sizes: [15, 20, 25, 30, 40, 45, 50, 60],
  };

  const rightBoxes = (rows) => ({
    student_id: { x_mm: DESIGN.right.x, y_mm: 73.66, w_mm: DESIGN.right.w, h_mm: 15.07 },
    subject_code: { x_mm: DESIGN.right.x, y_mm: 98.04, w_mm: DESIGN.right.w, h_mm: 15.07 },
    total_marks: { x_mm: DESIGN.right.x, y_mm: 122.43, w_mm: DESIGN.right.w, h_mm: 29.8 },
  });

  /** One sheet for `questionCount` questions: same design, more rows. */
  function buildTemplate(questionCount) {
    const count = Math.max(1, Math.min(DESIGN.counts[DESIGN.counts.length - 1], questionCount | 0));
    const rows = Math.ceil(count / DESIGN.perRow);
    const rowY = DESIGN.rowY.slice(0, rows);
    const colX = DESIGN.colX.slice();

    const bubbles = [];
    for (let q = 1; q <= count; q++) {
      const row = Math.floor((q - 1) / DESIGN.perRow);
      const column = (q - 1) % DESIGN.perRow;
      for (let option = 0; option < DESIGN.optionCount; option++) {
        bubbles.push([
          q,
          option,
          colX[column],
          round(rowY[row] + option * DESIGN.optionPitch),
          DESIGN.radius,
        ]);
      }
    }

    const frame = [
      round(colX[0] - DESIGN.cellW / 2),
      DESIGN.gridTop,
      round(colX[colX.length - 1] + DESIGN.cellW / 2),
      rows === DESIGN.rowY.length
        ? DESIGN.gridBottom
        : round(rowY[rows - 1] + (DESIGN.optionCount - 1) * DESIGN.optionPitch
          + DESIGN.radius + DESIGN.bandGap),
    ];

    return {
      id: 'omr_' + count,
      name: count + ' questions',
      question_count: count,
      option_count: DESIGN.optionCount,
      generated: true,
      page_mm: DESIGN.page.slice(),
      frame_mm: frame,
      regions: rightBoxes(rows),
      bubbles,
      layout: {
        perRow: DESIGN.perRow,
        rows,
        colX,
        rowY,
        optionPitch: DESIGN.optionPitch,
        radius: DESIGN.radius,
        cellW: DESIGN.cellW,
        gridTop: DESIGN.gridTop,
        bandHeight: DESIGN.bandHeight,
        bandGap: DESIGN.bandGap,
        optionLabels: LABELS,
      },
    };
  }

  /** The layout of any template, derived from its own bubbles when missing. */
  function layoutOf(template) {
    if (template.layout) return template.layout;
    const xs = Array.from(new Set(template.bubbles.map((b) => b[2]))).sort((a, b) => a - b);
    const ys = Array.from(new Set(template.bubbles.map((b) => b[3]))).sort((a, b) => a - b);
    const perRow = xs.length || 1;
    const rows = Math.max(1, Math.ceil(template.question_count / perRow));
    const optionPitch = ys.length > 1 ? round(ys[1] - ys[0]) : DESIGN.optionPitch;
    const rowPitch = ys.length > DESIGN.optionCount
      ? round(ys[DESIGN.optionCount] - ys[0]) : 0;
    const rowY = [];
    for (let row = 0; row < rows; row++) rowY.push(round(ys[0] + row * rowPitch));
    return {
      perRow,
      rows,
      colX: xs,
      rowY,
      optionPitch,
      radius: template.bubbles.length ? template.bubbles[0][4] : DESIGN.radius,
      cellW: DESIGN.cellW,
      gridTop: DESIGN.gridTop,
      bandHeight: DESIGN.bandHeight,
      bandGap: DESIGN.bandGap,
      optionLabels: LABELS,
    };
  }

  /* ----------------------------------------------------------- the registry */

  const builtin = {};
  DESIGN.counts.forEach((count) => { builtin[count] = buildTemplate(count); });
  const all = DESIGN.counts.map((count) => builtin[count]);
  const byId = {};
  all.forEach((sheet) => { byId[sheet.id] = sheet; });

  /* --------------------------------------------------------------- printing */

  const escapeHtml = (text) => String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const box = (className, left, top, width, height) =>
    `<div class="${className}" style="left:${round(left)}mm;top:${round(top)}mm;`
    + `width:${round(width)}mm;height:${round(height)}mm"></div>`;

  const line = (left, top, width, height, dark) =>
    `<div class="sheet-rule${dark ? ' is-dark' : ''}" style="left:${round(left)}mm;`
    + `top:${round(top)}mm;width:${round(width)}mm;height:${round(height)}mm"></div>`;

  /**
   * The printable sheet.  Every position is in millimetres and comes from the
   * same numbers the reader uses, so a printed sheet always lines up.  Print at
   * 100% (actual size), never "fit to page".
   */
  function sheetHtml(template, options) {
    const t = template;
    const opts = options || {};
    const L = layoutOf(t);
    const [fx0, fy0, fx1, fy1] = t.frame_mm;
    const pageW = DESIGN.page[0];
    const parts = [];

    parts.push(`<div class="sheet-page" style="width:${pageW}mm;height:${DESIGN.page[1]}mm">`);

    // Header: the plain sheet, or a heading printed from the school profile.
    const title = opts.title || 'OMR ANSWER SHEET';
    const subtitle = opts.subtitle
      || `${t.question_count} questions · 4 options · fill one bubble per question`;
    const titleTop = opts.logo ? 9 : 11;
    if (opts.logo) parts.push(`<img class="sheet-logo" src="${escapeHtml(opts.logo)}" alt="">`);
    parts.push(`<div class="sheet-title" style="left:0;top:${titleTop}mm;width:${pageW}mm">`
      + escapeHtml(title) + '</div>');
    parts.push(`<div class="sheet-sub" style="left:0;top:${titleTop + 7.5}mm;width:${pageW}mm">`
      + escapeHtml(subtitle) + '</div>');
    if (opts.line2) {
      parts.push(`<div class="sheet-sub" style="left:0;top:${titleTop + 11}mm;width:${pageW}mm">`
        + escapeHtml(opts.line2) + '</div>');
    }

    const fields = opts.fields || ['Name', 'Class / Roll', 'Subject'];
    const fieldTop = opts.line2 ? 28 : 25;
    fields.forEach((field, i) => {
      const top = fieldTop + i * 7;
      parts.push(`<div class="sheet-field" style="left:12mm;top:${top}mm">`
        + escapeHtml(field) + '</div>');
      parts.push(line(34, top + 5.2, 118, 0.3, true));
    });

    parts.push(line(12, DESIGN.gridTop - 6, 186, 0.4, false));

    // The answer table: outer border, a column line between neighbours, and a
    // band above every row of questions for its numbers.
    parts.push(box('sheet-frame', fx0, fy0, fx1 - fx0, fy1 - fy0));
    const edges = [round(fx0)];
    for (let i = 1; i < L.colX.length; i++) edges.push(round((L.colX[i - 1] + L.colX[i]) / 2));
    edges.push(round(fx1));
    for (let i = 1; i < edges.length - 1; i++) {
      parts.push(line(edges[i], fy0, 0.2, fy1 - fy0, false));
    }

    // The numbered band above every row, printed as a pale pink strip exactly
    // as on print/v3 (the reader uses the same strip as its page-side witness).
    L.rowY.forEach((rowTop) => {
      const bandTop = round(rowTop - L.radius - L.bandHeight);
      const bandBottom = round(rowTop - L.radius - L.bandGap);
      if (bandTop > fy0 + 0.5) {
        parts.push(box('sheet-band', fx0, bandTop, fx1 - fx0, bandBottom - bandTop));
      }
    });

    // Question numbers, centred over their column inside the band.
    for (let q = 1; q <= t.question_count; q++) {
      const row = Math.floor((q - 1) / L.perRow);
      const column = (q - 1) % L.perRow;
      const bandTop = L.rowY[row] - L.radius - L.bandHeight;
      parts.push(`<span class="sheet-q" style="left:${round(L.colX[column] - 4)}mm;`
        + `top:${round(bandTop + L.bandHeight / 2 - 2)}mm;width:8mm">${q}</span>`);
    }

    // The bubbles, with their letter printed inside as on the paper sheet.
    for (const [, option, x, y, r] of t.bubbles) {
      parts.push(`<span class="sheet-bubble" style="left:${round(x - r)}mm;`
        + `top:${round(y - r)}mm;width:${round(r * 2)}mm;height:${round(r * 2)}mm">`
        + `<b>${LABELS[option]}</b></span>`);
    }

    // Student ID, Subject Code and Total Marks, as on the printed sheet.
    const regionLabels = {
      student_id: 'Student ID',
      subject_code: 'Subject Code',
      total_marks: 'Total Marks',
    };
    Object.keys(regionLabels).forEach((key) => {
      const region = t.regions[key];
      if (!region) return;
      parts.push(`<div class="sheet-side" style="left:${round(region.x_mm)}mm;`
        + `top:${round(region.y_mm - 6)}mm;width:${round(region.w_mm)}mm">`
        + `${regionLabels[key]}</div>`);
      parts.push(box('sheet-markbox', region.x_mm, region.y_mm, region.w_mm, region.h_mm));
    });

    const footer = opts.footer
      || 'Fill one bubble for every question. Keep the answer table clean — the app'
      + ' reads only the bubbles.';
    parts.push(`<div class="sheet-foot" style="left:0;top:${round(fy1 + 10)}mm;`
      + `width:${pageW}mm">${escapeHtml(footer)}</div>`);
    parts.push('</div>');
    return parts.join('');
  }

  function printSheet(template, options) {
    const area = root.document && document.getElementById('printArea');
    if (!area || !template) return false;
    area.innerHTML = sheetHtml(template, options);

    const bridge = root.AndroidBridge;
    if (bridge && typeof bridge.printSheet === 'function') {
      try {
        bridge.printSheet();
        return true;
      } catch (error) {
        // Fall through to the browser path.
      }
    }
    if (typeof root.print === 'function') {
      root.print();
      return true;
    }
    return false;
  }

  root.OMR_SHEETS = {
    counts: DESIGN.counts.slice(),          // printed sheet formats: one (60)
    sizeChoices: DESIGN.sizes.slice(),      // how many questions an exam may ask
    master: builtin[DESIGN.counts[0]],
    all,
    builtin,
    byId,
    // There is one design, so any question count prints and reads the same
    // 60-slot sheet.
    forCount: () => builtin[DESIGN.counts[0]],
    // A grid covering only `count` questions, for tests that simulate a sheet
    // whose lower rows are not part of the exam.
    build: buildTemplate,
  };
  root.OMR_SHEETDESIGN = {
    design: DESIGN,
    buildTemplate,
    layoutOf,
    sheetHtml,
    printSheet,
    labels: LABELS,
    escapeHtml,
  };
  root.OMR_ACTIVE_TEMPLATE = builtin[DESIGN.counts[0]];
})(typeof window !== 'undefined' ? window : globalThis);
