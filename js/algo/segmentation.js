import { hsvToRgb } from '../core/pixel.js';
import { runOp, separableConv } from './parallel.js';
import { gaussianKernel1D } from '../core/convolution.js';

const POINT_KERNEL = new Float32Array([-1, -1, -1, -1, 8, -1, -1, -1, -1]);
const LINE_COLORS = [[255, 70, 70], [80, 230, 90], [90, 130, 255], [255, 215, 70]];

function grayImage(arr, w, h, absolute, normalize) {
  const img = new ImageData(w, h);
  const d = img.data;
  let max = 1e-6;
  if (normalize) {
    for (let i = 0; i < arr.length; i++) {
      const v = absolute ? Math.abs(arr[i]) : arr[i];
      if (v > max) max = v;
    }
  } else {
    max = 255;
  }
  for (let i = 0, p = 0; i < arr.length; i++, p += 4) {
    let v = absolute ? Math.abs(arr[i]) : arr[i];
    if (normalize) v = (v / max) * 255;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    d[p] = v; d[p + 1] = v; d[p + 2] = v; d[p + 3] = 255;
  }
  return img;
}

function directionImage(mag, dir, w, h) {
  const img = new ImageData(w, h);
  const d = img.data;
  const rgb = [0, 0, 0];
  let max = 1e-6;
  for (let i = 0; i < mag.length; i++) if (mag[i] > max) max = mag[i];
  for (let i = 0, p = 0; i < mag.length; i++, p += 4) {
    if (mag[i] <= 0) { d[p + 3] = 255; continue; }
    const hue = ((dir[i] + Math.PI) / (2 * Math.PI)) * 360;
    hsvToRgb(hue, 1, (mag[i] / max) * 255, rgb);
    d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2]; d[p + 3] = 255;
  }
  return img;
}

function maskImage(w, h, predicate) {
  const img = new ImageData(w, h);
  const d = img.data;
  let count = 0;
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    const c = predicate(i);
    if (c) {
      d[p] = c[0]; d[p + 1] = c[1]; d[p + 2] = c[2]; d[p + 3] = 255;
      count++;
    }
  }
  img.detectedCount = count;
  return img;
}

function bilinearSample(arr, w, h, x, y) {
  if (x < 0) x = 0;
  if (x > w - 1) x = w - 1;
  if (y < 0) y = 0;
  if (y > h - 1) y = h - 1;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const fx = x - x0, fy = y - y0;
  return arr[y0 * w + x0] * (1 - fx) * (1 - fy) + arr[y0 * w + x1] * fx * (1 - fy) +
         arr[y1 * w + x0] * (1 - fx) * fy + arr[y1 * w + x1] * fx * fy;
}

function nmsClassic(mag, dir, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m <= 0) continue;
      const deg = ((dir[i] * 180 / Math.PI) + 180) % 180;
      let n1, n2;
      if (deg < 22.5 || deg >= 157.5) {
        n1 = mag[y * w + Math.min(x + 1, w - 1)];
        n2 = mag[y * w + Math.max(x - 1, 0)];
      } else if (deg < 67.5) {
        n1 = mag[Math.min(y + 1, h - 1) * w + Math.min(x + 1, w - 1)];
        n2 = mag[Math.max(y - 1, 0) * w + Math.max(x - 1, 0)];
      } else if (deg < 112.5) {
        n1 = mag[Math.min(y + 1, h - 1) * w + x];
        n2 = mag[Math.max(y - 1, 0) * w + x];
      } else {
        n1 = mag[Math.min(y + 1, h - 1) * w + Math.max(x - 1, 0)];
        n2 = mag[Math.max(y - 1, 0) * w + Math.min(x + 1, w - 1)];
      }
      if (m >= n1 && m >= n2) out[i] = m;
    }
  }
  return out;
}

function nmsSubpixel(mag, dir, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m <= 0) continue;
      const ux = Math.cos(dir[i]), uy = Math.sin(dir[i]);
      const m1 = bilinearSample(mag, w, h, x + ux, y + uy);
      const m2 = bilinearSample(mag, w, h, x - ux, y - uy);
      if (m >= m1 && m >= m2) out[i] = m;
    }
  }
  return out;
}

function hysteresis(nms, w, h, low, high) {
  const res = new Uint8Array(w * h);
  const state = new Uint8Array(w * h);
  const stack = [];
  for (let i = 0; i < nms.length; i++) {
    if (nms[i] >= high) { state[i] = 2; stack.push(i); }
    else if (nms[i] >= low) state[i] = 1;
  }
  while (stack.length) {
    const i = stack.pop();
    const x = i % w, y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const j = ny * w + nx;
        if (state[j] === 1) {
          state[j] = 2;
          stack.push(j);
        }
      }
    }
  }
  let count = 0;
  for (let i = 0; i < state.length; i++) {
    if (state[i] === 2) { res[i] = 255; count++; }
  }
  return { res, count };
}

export async function segment(pb, params, onProgress) {
  const w = pb.width, h = pb.height;
  const gray = pb.getGray();
  const { method, edge, threshold, normalize } = params;
  const prog = (fn) => (p) => { if (onProgress) onProgress(fn(p)); };

  if (method === 'point') {
    const { out } = await runOp('conv2', gray, w, h, { kernel: POINT_KERNEL, ksize: 3 }, edge, prog((p) => p));
    const mask = maskImage(w, h, (i) => Math.abs(out[i]) >= threshold ? [255, 255, 255] : null);
    return {
      views: [
        { id: 'mask', label: 'Обнаруженные точки', image: mask },
        { id: 'resp', label: 'Карта отклика лапласиана', image: grayImage(out, w, h, true, normalize) }
      ],
      info: `обнаружено пикселей: ${mask.detectedCount}`
    };
  }

  if (method === 'line') {
    const { resp, dirIdx } = await runOp('lines', gray, w, h, {}, edge, prog((p) => p));
    const mask = maskImage(w, h, (i) => Math.abs(resp[i]) >= threshold ? LINE_COLORS[dirIdx[i]] : null);
    return {
      views: [
        { id: 'mask', label: 'Обнаруженные линии (цвет = ориентация)', image: mask },
        { id: 'resp', label: 'Карта отклика (max по 4 маскам)', image: grayImage(resp, w, h, true, normalize) }
      ],
      info: `обнаружено пикселей: ${mask.detectedCount} (красный 0°, зелёный 45°, синий 90°, жёлтый 135°)`
    };
  }

  if (method === 'canny') {
    const gk = gaussianKernel1D(params.sigma);
    const smooth = await separableConv(gray, w, h, gk.kernel, gk.radius, edge, prog((p) => p * 0.35));
    const { mag, dir } = await runOp('grad', smooth, w, h, { op: 'sobel' }, edge, prog((p) => 0.35 + p * 0.3));
    if (onProgress) onProgress(0.75);
    const nms = params.subpixel ? nmsSubpixel(mag, dir, w, h) : nmsClassic(mag, dir, w, h);
    const { res, count } = hysteresis(nms, w, h, params.low, params.high);
    if (onProgress) onProgress(1);
    return {
      views: [
        { id: 'canny', label: 'Результат Кэнни', image: grayImage(res, w, h, false, false) },
        { id: 'nms', label: 'Подавление немаксимумов', image: grayImage(nms, w, h, false, normalize) },
        { id: 'mag', label: 'Модуль градиента', image: grayImage(mag, w, h, false, normalize) },
        { id: 'dir', label: 'Карта направлений градиента', image: directionImage(mag, dir, w, h) }
      ],
      info: `граничных пикселей: ${count}; σ = ${params.sigma}; NMS: ${params.subpixel ? 'субпиксельный' : 'классический'}`
    };
  }

  const { mag, dir } = await runOp('grad', gray, w, h, { op: method }, edge, prog((p) => p));
  const mask = maskImage(w, h, (i) => mag[i] >= threshold ? [255, 255, 255] : null);
  return {
    views: [
      { id: 'mask', label: 'Бинарные границы (порог T)', image: mask },
      { id: 'mag', label: 'Модуль градиента', image: grayImage(mag, w, h, false, normalize) },
      { id: 'dir', label: 'Карта направлений градиента', image: directionImage(mag, dir, w, h) }
    ],
    info: `обнаружено пикселей: ${mask.detectedCount}`
  };
}
