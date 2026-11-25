import { PixelBuffer, rgbToHsv, hsvToRgb } from '../core/pixel.js';

export function channelHists(pb) {
  const r = new Uint32Array(256), g = new Uint32Array(256), b = new Uint32Array(256);
  const d = pb.data;
  for (let p = 0; p < d.length; p += 4) {
    r[d[p]]++;
    g[d[p + 1]]++;
    b[d[p + 2]]++;
  }
  return { r, g, b };
}

export function eqLUT(hist) {
  const lut = new Uint8Array(256);
  let total = 0;
  let first = -1;
  for (let i = 0; i < 256; i++) {
    total += hist[i];
    if (first < 0 && hist[i] > 0) first = i;
  }
  if (total <= 0 || first < 0) {
    for (let i = 0; i < 256; i++) lut[i] = i;
    return lut;
  }
  const cdfMin = hist[first];
  let cum = 0;
  for (let i = 0; i < 256; i++) {
    cum += hist[i];
    let v = Math.round(((cum - cdfMin) / (total - cdfMin)) * 255);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    lut[i] = i < first ? 0 : v;
  }
  return lut;
}

export function linearLUT(a, b) {
  const lut = new Uint8Array(256);
  const span = Math.max(1, b - a);
  for (let i = 0; i < 256; i++) {
    if (i <= a) lut[i] = 0;
    else if (i >= b) lut[i] = 255;
    else lut[i] = Math.round(((i - a) * 255) / span);
  }
  return lut;
}

function applyLUTs(pb, lr, lg, lb) {
  const out = new PixelBuffer(pb.width, pb.height);
  const s = pb.data, d = out.data;
  for (let p = 0; p < s.length; p += 4) {
    d[p] = lr[s[p]];
    d[p + 1] = lg[s[p + 1]];
    d[p + 2] = lb[s[p + 2]];
    d[p + 3] = 255;
  }
  return out;
}

export function linearContrast(pb, a, b) {
  const lut = linearLUT(a, b);
  return applyLUTs(pb, lut, lut, lut);
}

export function equalizeRGB(pb) {
  const hists = channelHists(pb);
  return applyLUTs(pb, eqLUT(hists.r), eqLUT(hists.g), eqLUT(hists.b));
}

export function equalizeHSV(pb) {
  const n = pb.width * pb.height;
  const d = pb.data;
  const H = new Float32Array(n);
  const S = new Float32Array(n);
  const V = new Uint8Array(n);
  const hsv = [0, 0, 0];
  const hist = new Uint32Array(256);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    rgbToHsv(d[p], d[p + 1], d[p + 2], hsv);
    H[i] = hsv[0];
    S[i] = hsv[1];
    V[i] = Math.round(hsv[2]);
    hist[V[i]]++;
  }
  const lut = eqLUT(hist);
  const out = new PixelBuffer(pb.width, pb.height);
  const od = out.data;
  const rgb = [0, 0, 0];
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    hsvToRgb(H[i], S[i], lut[V[i]], rgb);
    od[p] = rgb[0];
    od[p + 1] = rgb[1];
    od[p + 2] = rgb[2];
    od[p + 3] = 255;
  }
  return out;
}

function clampTile(t, tiles) {
  return t < 0 ? 0 : t >= tiles ? tiles - 1 : t;
}

export function clahe(pb, tiles, clip) {
  const w = pb.width, h = pb.height, n = w * h;
  const d = pb.data;
  const H = new Float32Array(n);
  const S = new Float32Array(n);
  const V = new Uint8Array(n);
  const hsv = [0, 0, 0];
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    rgbToHsv(d[p], d[p + 1], d[p + 2], hsv);
    H[i] = hsv[0];
    S[i] = hsv[1];
    V[i] = Math.round(hsv[2]);
  }
  const tileW = w / tiles, tileH = h / tiles;
  const luts = [];
  for (let ty = 0; ty < tiles; ty++) {
    const row = [];
    for (let tx = 0; tx < tiles; tx++) {
      const hist = new Float64Array(256);
      const x0 = Math.floor(tx * tileW), x1 = Math.floor((tx + 1) * tileW);
      const y0 = Math.floor(ty * tileH), y1 = Math.floor((ty + 1) * tileH);
      let cnt = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          hist[V[y * w + x]]++;
          cnt++;
        }
      }
      const limit = (clip * cnt) / 256;
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }
      const add = excess / 256;
      for (let i = 0; i < 256; i++) hist[i] += add;
      row.push(eqLUT(hist));
    }
    luts.push(row);
  }
  const out = new PixelBuffer(w, h);
  const od = out.data;
  const rgb = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    const gy = (y + 0.5) / tileH - 0.5;
    const ty = Math.floor(gy);
    const fy = gy - ty;
    const tyA = clampTile(ty, tiles), tyB = clampTile(ty + 1, tiles);
    for (let x = 0; x < w; x++) {
      const gx = (x + 0.5) / tileW - 0.5;
      const tx = Math.floor(gx);
      const fx = gx - tx;
      const txA = clampTile(tx, tiles), txB = clampTile(tx + 1, tiles);
      const i = y * w + x;
      const v = V[i];
      const nv =
        luts[tyA][txA][v] * (1 - fx) * (1 - fy) +
        luts[tyA][txB][v] * fx * (1 - fy) +
        luts[tyB][txA][v] * (1 - fx) * fy +
        luts[tyB][txB][v] * fx * fy;
      hsvToRgb(H[i], S[i], nv, rgb);
      const p = i * 4;
      od[p] = rgb[0];
      od[p + 1] = rgb[1];
      od[p + 2] = rgb[2];
      od[p + 3] = 255;
    }
  }
  return out;
}
