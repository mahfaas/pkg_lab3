export const EDGE_CLAMP = 'clamp';
export const EDGE_MIRROR = 'mirror';

export function edgeIndex(v, max, mode) {
  if (v >= 0 && v <= max) return v;
  if (mode === EDGE_CLAMP) return v < 0 ? 0 : max;
  for (;;) {
    if (v < 0) v = -v;
    else if (v > max) v = 2 * max - v;
    else return v;
  }
}

export class PixelBuffer {
  constructor(width, height, data = null) {
    this.width = width;
    this.height = height;
    this.data = data ?? new Uint8ClampedArray(width * height * 4);
  }

  static fromImageData(img) {
    return new PixelBuffer(img.width, img.height, new Uint8ClampedArray(img.data));
  }

  clone() {
    return new PixelBuffer(this.width, this.height, new Uint8ClampedArray(this.data));
  }

  toImageData() {
    return new ImageData(new Uint8ClampedArray(this.data), this.width, this.height);
  }

  getGray() {
    const d = this.data;
    const g = new Float32Array(this.width * this.height);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    }
    return g;
  }

  minMax() {
    const d = this.data;
    let mn = 255, mx = 0;
    for (let p = 0; p < d.length; p += 4) {
      if (d[p] < mn) mn = d[p];
      if (d[p] > mx) mx = d[p];
      if (d[p + 1] < mn) mn = d[p + 1];
      if (d[p + 1] > mx) mx = d[p + 1];
      if (d[p + 2] < mn) mn = d[p + 2];
      if (d[p + 2] > mx) mx = d[p + 2];
    }
    return [mn, mx];
  }
}

export function rgbToHsv(r, g, b, out) {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = 60 * (((g - b) / d) % 6);
    else if (mx === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
  }
  if (h < 0) h += 360;
  out[0] = h;
  out[1] = mx === 0 ? 0 : d / mx;
  out[2] = mx;
}

export function hsvToRgb(h, s, v, out) {
  const c = v * s;
  const hp = ((h % 360) + 360) % 360;
  const x = c * (1 - Math.abs((hp / 60) % 2 - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 60) { r = c; g = x; }
  else if (hp < 120) { r = x; g = c; }
  else if (hp < 180) { g = c; b = x; }
  else if (hp < 240) { g = x; b = c; }
  else if (hp < 300) { r = x; b = c; }
  else { r = c; b = x; }
  const m = v - c;
  out[0] = r + m;
  out[1] = g + m;
  out[2] = b + m;
}
