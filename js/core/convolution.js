import { edgeIndex } from './pixel.js';

export function convolve2D(src, w, h, kernel, ksize, edge) {
  const r = (ksize - 1) >> 1;
  const dst = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = -r; ky <= r; ky++) {
        const sy = edgeIndex(y + ky, h - 1, edge) * w;
        for (let kx = -r; kx <= r; kx++) {
          sum += src[sy + edgeIndex(x + kx, w - 1, edge)] * kernel[(ky + r) * ksize + (kx + r)];
        }
      }
      dst[y * w + x] = sum;
    }
  }
  return dst;
}

export function convolveSeparable(src, w, h, kernel, radius, edge) {
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -radius; i <= radius; i++) {
        s += src[row + edgeIndex(x + i, w - 1, edge)] * kernel[i + radius];
      }
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -radius; i <= radius; i++) {
        s += tmp[edgeIndex(y + i, h - 1, edge) * w + x] * kernel[i + radius];
      }
      dst[y * w + x] = s;
    }
  }
  return dst;
}

export function gaussianKernel1D(sigma) {
  const r = Math.max(1, Math.ceil(3 * sigma));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    s += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return { kernel: k, radius: r };
}

export function boxKernel1D(radius) {
  const n = 2 * radius + 1;
  return { kernel: new Float32Array(n).fill(1 / n), radius };
}
