import { edgeIndex } from '../core/pixel.js';

export function hPassStripe(src, w, y0, y1, kernel, radius, edge) {
  const out = new Float32Array(w * (y1 - y0));
  for (let y = y0; y < y1; y++) {
    const row = (y - y0) * w;
    const srow = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -radius; i <= radius; i++) {
        s += src[srow + edgeIndex(x + i, w - 1, edge)] * kernel[i + radius];
      }
      out[row + x] = s;
    }
  }
  return out;
}

export function vPassStripe(tmp, w, h, y0, y1, kernel, radius, edge) {
  const out = new Float32Array(w * (y1 - y0));
  for (let y = y0; y < y1; y++) {
    const row = (y - y0) * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -radius; i <= radius; i++) {
        s += tmp[edgeIndex(y + i, h - 1, edge) * w + x] * kernel[i + radius];
      }
      out[row + x] = s;
    }
  }
  return out;
}

export function conv2Stripe(src, w, h, y0, y1, kernel, ksize, edge) {
  const r = (ksize - 1) >> 1;
  const dst = new Float32Array(w * (y1 - y0));
  for (let y = y0; y < y1; y++) {
    const row = (y - y0) * w;
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = -r; ky <= r; ky++) {
        const sy = edgeIndex(y + ky, h - 1, edge) * w;
        for (let kx = -r; kx <= r; kx++) {
          sum += src[sy + edgeIndex(x + kx, w - 1, edge)] * kernel[(ky + r) * ksize + (kx + r)];
        }
      }
      dst[row + x] = sum;
    }
  }
  return dst;
}

export function gradStripe(src, w, h, y0, y1, op, edge) {
  const sh = y1 - y0;
  const gx = new Float32Array(w * sh);
  const gy = new Float32Array(w * sh);
  const mag = new Float32Array(w * sh);
  const dir = new Float32Array(w * sh);
  if (op === 'roberts') {
    for (let y = y0; y < y1; y++) {
      const row = (y - y0) * w;
      for (let x = 0; x < w; x++) {
        const a = src[y * w + x];
        const b = src[y * w + edgeIndex(x + 1, w - 1, edge)];
        const c = src[edgeIndex(y + 1, h - 1, edge) * w + x];
        const d = src[edgeIndex(y + 1, h - 1, edge) * w + edgeIndex(x + 1, w - 1, edge)];
        const vx = a - d;
        const vy = b - c;
        gx[row + x] = vx;
        gy[row + x] = vy;
        mag[row + x] = Math.hypot(vx, vy);
        dir[row + x] = Math.atan2(vy, vx);
      }
    }
    return { gx, gy, mag, dir };
  }
  const side = op === 'sobel' ? 2 : 1;
  for (let y = y0; y < y1; y++) {
    const row = (y - y0) * w;
    for (let x = 0; x < w; x++) {
      const xm = edgeIndex(x - 1, w - 1, edge);
      const xp = edgeIndex(x + 1, w - 1, edge);
      const ym = edgeIndex(y - 1, h - 1, edge);
      const yp = edgeIndex(y + 1, h - 1, edge);
      const tl = src[ym * w + xm], tc = src[ym * w + x], tr = src[ym * w + xp];
      const ml = src[y * w + xm], mr = src[y * w + xp];
      const bl = src[yp * w + xm], bc = src[yp * w + x], br = src[yp * w + xp];
      const vx = (tr + side * mr + br) - (tl + side * ml + bl);
      const vy = (bl + side * bc + br) - (tl + side * tc + tr);
      gx[row + x] = vx;
      gy[row + x] = vy;
      mag[row + x] = Math.hypot(vx, vy);
      dir[row + x] = Math.atan2(vy, vx);
    }
  }
  return { gx, gy, mag, dir };
}

const LINE_KERNELS = [
  [-1, -1, -1, 2, 2, 2, -1, -1, -1],
  [2, -1, -1, -1, 2, -1, -1, -1, 2],
  [-1, 2, -1, -1, 2, -1, -1, 2, -1],
  [-1, -1, 2, -1, 2, -1, 2, -1, -1]
];

export function linesStripe(src, w, h, y0, y1, edge) {
  const sh = y1 - y0;
  const resp = new Float32Array(w * sh);
  const dirIdx = new Uint8Array(w * sh);
  for (let y = y0; y < y1; y++) {
    const row = (y - y0) * w;
    for (let x = 0; x < w; x++) {
      let bestVal = 0;
      let bestAbs = -1;
      let bestIdx = 0;
      for (let k = 0; k < 4; k++) {
        const kern = LINE_KERNELS[k];
        let sum = 0;
        let ki = 0;
        for (let ky = -1; ky <= 1; ky++) {
          const sy = edgeIndex(y + ky, h - 1, edge) * w;
          for (let kx = -1; kx <= 1; kx++, ki++) {
            sum += src[sy + edgeIndex(x + kx, w - 1, edge)] * kern[ki];
          }
        }
        const a = Math.abs(sum);
        if (a > bestAbs) { bestAbs = a; bestVal = sum; bestIdx = k; }
      }
      resp[row + x] = bestVal;
      dirIdx[row + x] = bestIdx;
    }
  }
  return { resp, dirIdx };
}
