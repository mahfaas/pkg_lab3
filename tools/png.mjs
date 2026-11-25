import zlib from 'node:zlib';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

export function encodePng(width, height, rgba) {
  const raw = Buffer.alloc(height * (width * 4 + 1));
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0))
  ]);
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

export function decodePng(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let pos = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat = [];
  let palette = null;
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      interlace = data[12];
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'PLTE') {
      palette = data;
    } else if (type === 'IEND') {
      break;
    }
    pos += 12 + len;
  }
  if (bitDepth !== 8) throw new Error('unsupported bit depth');
  if (interlace !== 0) throw new Error('interlaced PNG unsupported');
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
  if (!channels) throw new Error('unsupported color type');
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8ClampedArray(width * height * 4);
  const prev = Buffer.alloc(stride);
  const cur = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    raw.copy(cur, 0, y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const left = i >= channels ? cur[i - channels] : 0;
      const up = prev[i];
      const ul = i >= channels ? prev[i - channels] : 0;
      let v = cur[i];
      if (filter === 1) v += left;
      else if (filter === 2) v += up;
      else if (filter === 3) v += (left + up) >> 1;
      else if (filter === 4) v += paeth(left, up, ul);
      cur[i] = v & 0xff;
    }
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      if (colorType === 6) {
        out[o] = cur[x * 4];
        out[o + 1] = cur[x * 4 + 1];
        out[o + 2] = cur[x * 4 + 2];
        out[o + 3] = cur[x * 4 + 3];
      } else if (colorType === 2) {
        out[o] = cur[x * 3];
        out[o + 1] = cur[x * 3 + 1];
        out[o + 2] = cur[x * 3 + 2];
        out[o + 3] = 255;
      } else if (colorType === 0) {
        out[o] = out[o + 1] = out[o + 2] = cur[x];
        out[o + 3] = 255;
      } else if (colorType === 4) {
        out[o] = out[o + 1] = out[o + 2] = cur[x * 2];
        out[o + 3] = cur[x * 2 + 1];
      } else if (colorType === 3) {
        const idx = cur[x] * 3;
        out[o] = palette[idx];
        out[o + 1] = palette[idx + 1];
        out[o + 2] = palette[idx + 2];
        out[o + 3] = 255;
      }
    }
    cur.copy(prev);
  }
  return { width, height, rgba: out };
}

export function pbToPng(pb) {
  return encodePng(pb.width, pb.height, pb.data);
}

export function imageToPng(img) {
  return encodePng(img.width, img.height, img.data);
}

export function sideBySide(items, gap = 10, bg = [23, 29, 41]) {
  const h = Math.max(...items.map((i) => i.height));
  const w = items.reduce((s, i) => s + i.width, 0) + gap * (items.length - 1);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = bg[0];
    out[i + 1] = bg[1];
    out[i + 2] = bg[2];
    out[i + 3] = 255;
  }
  let xoff = 0;
  for (const it of items) {
    for (let y = 0; y < it.height; y++) {
      for (let x = 0; x < it.width; x++) {
        const s = (y * it.width + x) * 4;
        const d = (y * w + xoff + x) * 4;
        out[d] = it.data[s];
        out[d + 1] = it.data[s + 1];
        out[d + 2] = it.data[s + 2];
        out[d + 3] = it.data[s + 3];
      }
    }
    xoff += it.width + gap;
  }
  return { width: w, height: h, data: out };
}

export function histChartPng(hists, width = 512, height = 220) {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = 23;
    out[i + 1] = 29;
    out[i + 2] = 41;
    out[i + 3] = 255;
  }
  let max = 0;
  for (const h of [hists.r, hists.g, hists.b]) {
    for (let i = 0; i < 256; i++) if (h[i] > max) max = h[i];
  }
  if (max <= 0) return { width, height, data: out };
  const bw = width / 256;
  const colors = [
    [255, 90, 90],
    [90, 230, 110],
    [100, 140, 255]
  ];
  [hists.r, hists.g, hists.b].forEach((h, c) => {
    const [cr, cg, cb] = colors[c];
    for (let i = 0; i < 256; i++) {
      const bh = (h[i] / max) * (height - 6);
      if (bh <= 0) continue;
      const x0 = Math.floor(i * bw);
      const x1 = Math.min(width - 1, Math.ceil((i + 1) * bw) - 1);
      const y0 = Math.floor(height - bh);
      const y1 = height - 1;
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const p = (y * width + x) * 4;
          out[p] = Math.round(out[p] * 0.35 + cr * 0.65);
          out[p + 1] = Math.round(out[p + 1] * 0.35 + cg * 0.65);
          out[p + 2] = Math.round(out[p + 2] * 0.35 + cb * 0.65);
        }
      }
    }
  });
  return { width, height, data: out };
}
