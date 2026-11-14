/*
 * Minimal PNG reader for the tests: 8-bit RGB/RGBA, non-interlaced.
 * Enough to load the committed screenshots and frames captured from a phone.
 */
'use strict';

const fs = require('fs');
const zlib = require('zlib');

function readPNG(file) {
  const buf = fs.readFileSync(file);
  let offset = 8;
  let width = 0;
  let height = 0;
  let colorType = 6;
  let depth = 8;
  const idat = [];
  while (offset < buf.length) {
    const length = buf.readUInt32BE(offset);
    const type = buf.toString('ascii', offset + 4, offset + 8);
    const data = buf.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      depth = data[8];
      colorType = data[9];
      if (data[12] !== 0) throw new Error('interlaced PNG is not supported');
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    offset += 12 + length;
  }
  if (depth !== 8) throw new Error('only 8-bit PNG is supported');
  const bpp = colorType === 2 ? 3 : colorType === 6 ? 4 : colorType === 0 ? 1 : 0;
  if (!bpp) throw new Error('unsupported PNG colour type ' + colorType);

  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * bpp;
  const out = new Uint8ClampedArray(width * height * 4);
  let previous = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = Buffer.from(raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)));
    for (let i = 0; i < stride; i++) {
      const left = i >= bpp ? line[i - bpp] : 0;
      const up = previous[i];
      const upLeft = i >= bpp ? previous[i - bpp] : 0;
      let value = line[i];
      if (filter === 1) value += left;
      else if (filter === 2) value += up;
      else if (filter === 3) value += (left + up) >> 1;
      else if (filter === 4) {
        const p = left + up - upLeft;
        const pa = Math.abs(p - left);
        const pb = Math.abs(p - up);
        const pc = Math.abs(p - upLeft);
        value += pa <= pb && pa <= pc ? left : pb <= pc ? up : upLeft;
      }
      line[i] = value & 255;
    }
    for (let x = 0; x < width; x++) {
      const s = x * bpp;
      const t = (y * width + x) * 4;
      if (bpp === 1) {
        out[t] = line[s]; out[t + 1] = line[s]; out[t + 2] = line[s];
      } else {
        out[t] = line[s]; out[t + 1] = line[s + 1]; out[t + 2] = line[s + 2];
      }
      out[t + 3] = 255;
    }
    previous = line;
  }
  return { data: out, width, height };
}

module.exports = { readPNG };
