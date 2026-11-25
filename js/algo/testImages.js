import { PixelBuffer, EDGE_CLAMP, rgbToHsv, hsvToRgb } from '../core/pixel.js';
import { convolveSeparable, boxKernel1D } from '../core/convolution.js';

const SIZE = 512;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function setPx(d, i, r, g, b) {
  d[i] = r;
  d[i + 1] = g;
  d[i + 2] = b;
  d[i + 3] = 255;
}

function fillRectPx(d, x0, y0, w, h, color) {
  for (let y = Math.max(0, y0); y < Math.min(SIZE, y0 + h); y++) {
    for (let x = Math.max(0, x0); x < Math.min(SIZE, x0 + w); x++) {
      setPx(d, (y * SIZE + x) * 4, color[0], color[1], color[2]);
    }
  }
}

function fillCirclePx(d, cx, cy, r, color) {
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(SIZE - 1, Math.ceil(cy + r)); y++) {
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(SIZE - 1, Math.ceil(cx + r)); x++) {
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
        setPx(d, (y * SIZE + x) * 4, color[0], color[1], color[2]);
      }
    }
  }
}

function segDist(px, py, x0, y0, x1, y1) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len2 = dx * dx + dy * dy;
  let t = len2 === 0 ? 0 : ((px - x0) * dx + (py - y0) * dy) / len2;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  const qx = x0 + t * dx;
  const qy = y0 + t * dy;
  return Math.hypot(px - qx, py - qy);
}

function strokeLinePx(d, x0, y0, x1, y1, halfW, color) {
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - halfW - 1));
  const maxX = Math.min(SIZE - 1, Math.ceil(Math.max(x0, x1) + halfW + 1));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - halfW - 1));
  const maxY = Math.min(SIZE - 1, Math.ceil(Math.max(y0, y1) + halfW + 1));
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (segDist(x, y, x0, y0, x1, y1) <= halfW) {
        setPx(d, (y * SIZE + x) * 4, color[0], color[1], color[2]);
      }
    }
  }
}

function strokeTrianglePx(d, pts, halfW, color) {
  for (let i = 0; i < 3; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % 3];
    strokeLinePx(d, a[0], a[1], b[0], b[1], halfW, color);
  }
}

function hslToRgb(h, s, l, out) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = ((h % 360) + 360) % 360;
  const x = c * (1 - Math.abs((hp / 60) % 2 - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 60) { r = c; g = x; }
  else if (hp < 120) { r = x; g = c; }
  else if (hp < 180) { g = c; b = x; }
  else if (hp < 240) { g = x; b = c; }
  else if (hp < 300) { r = x; b = c; }
  else { r = c; b = x; }
  const m = l - c / 2;
  out[0] = Math.round((r + m) * 255);
  out[1] = Math.round((g + m) * 255);
  out[2] = Math.round((b + m) * 255);
}

function emptyBuffer() {
  return new Uint8ClampedArray(SIZE * SIZE * 4);
}

export function sceneSeg() {
  const d = emptyBuffer();
  for (let y = 0; y < SIZE; y++) {
    const t = y / (SIZE - 1);
    const r = Math.round(43 + 15 * t);
    const g = Math.round(47 + 16 * t);
    const b = Math.round(56 + 20 * t);
    for (let x = 0; x < SIZE; x++) setPx(d, (y * SIZE + x) * 4, r, g, b);
  }
  fillCirclePx(d, 140, 150, 70, [223, 230, 242]);
  fillRectPx(d, 300, 90, 140, 110, [152, 163, 184]);
  strokeTrianglePx(d, [[320, 330], [470, 445], [180, 445]], 4, [242, 242, 242]);
  strokeLinePx(d, 55, 485, 260, 485, 0.5, [255, 255, 255]);
  strokeLinePx(d, 70, 295, 205, 430, 0.5, [255, 255, 255]);
  strokeLinePx(d, 240, 275, 240, 465, 0.5, [255, 255, 255]);
  strokeLinePx(d, 445, 295, 325, 415, 0.5, [255, 255, 255]);
  for (const [px, py] of [[50, 60], [480, 80], [492, 262], [42, 242], [262, 205]]) {
    fillCirclePx(d, px, py, 0.5, [255, 255, 255]);
  }
  return new PixelBuffer(SIZE, SIZE, d);
}

function remapRange(pb, lo, hi) {
  const out = new PixelBuffer(pb.width, pb.height);
  const s = pb.data, d = out.data;
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.round(lo + (i / 255) * (hi - lo));
  for (let p = 0; p < s.length; p += 4) {
    d[p] = lut[s[p]];
    d[p + 1] = lut[s[p + 1]];
    d[p + 2] = lut[s[p + 2]];
    d[p + 3] = 255;
  }
  return out;
}

function addGaussianNoise(pb, sigma, seed = 20260225) {
  const rnd = mulberry32(seed);
  const out = pb.clone();
  const d = out.data;
  for (let p = 0; p < d.length; p += 4) {
    for (let c = 0; c < 3; c++) {
      let u = 0, v = 0;
      while (u === 0) u = rnd();
      v = rnd();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      d[p + c] += g * sigma;
    }
  }
  return out;
}

function addSaltPepper(pb, p, seed = 777) {
  const rnd = mulberry32(seed);
  const out = pb.clone();
  const d = out.data;
  for (let i = 0; i < d.length / 4; i++) {
    const r = rnd();
    if (r < p / 2) { d[i * 4] = 255; d[i * 4 + 1] = 255; d[i * 4 + 2] = 255; }
    else if (r < p) { d[i * 4] = 0; d[i * 4 + 1] = 0; d[i * 4 + 2] = 0; }
  }
  return out;
}

function blurColor(pb, radius) {
  const { kernel, radius: r } = boxKernel1D(radius);
  const w = pb.width, h = pb.height, d = pb.data;
  const planes = [0, 1, 2].map((c) => {
    const pl = new Float32Array(w * h);
    for (let i = 0, p = c; i < pl.length; i++, p += 4) pl[i] = d[p];
    return pl;
  }).map((pl) => convolveSeparable(pl, w, h, kernel, r, EDGE_CLAMP));
  const out = new PixelBuffer(w, h);
  const od = out.data;
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    od[p] = planes[0][i];
    od[p + 1] = planes[1][i];
    od[p + 2] = planes[2][i];
    od[p + 3] = 255;
  }
  return out;
}

function unevenLight(pb) {
  const w = pb.width, h = pb.height;
  const out = new PixelBuffer(w, h);
  const s = pb.data, d = out.data;
  const cx = w / 2, cy = h / 2;
  const maxR = Math.hypot(cx, cy);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const k = 0.35 + 0.65 * (1 - Math.hypot(x - cx, y - cy) / maxR);
      const p = (y * w + x) * 4;
      d[p] = s[p] * k;
      d[p + 1] = s[p + 1] * k;
      d[p + 2] = s[p + 2] * k;
      d[p + 3] = 255;
    }
  }
  return out;
}

function colorLow() {
  const d = emptyBuffer();
  const stops = [[210, 0.55, 0.30], [160, 0.50, 0.45], [20, 0.60, 0.35]];
  const rgb = [0, 0, 0];
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const t = x / (SIZE - 1);
      const seg = t < 0.5 ? 0 : 1;
      const lt = seg === 0 ? t / 0.5 : (t - 0.5) / 0.5;
      const a = stops[seg], b = stops[seg + 1];
      const h = a[0] + (b[0] - a[0]) * lt;
      const s = a[1] + (b[1] - a[1]) * lt;
      const l = a[2] + (b[2] - a[2]) * lt;
      hslToRgb(h, s, l, rgb);
      setPx(d, (y * SIZE + x) * 4, rgb[0], rgb[1], rgb[2]);
    }
  }
  hslToRgb(0, 0.75, 0.60, rgb);
  fillCirclePx(d, 150, 160, 85, [rgb[0], rgb[1], rgb[2]]);
  hslToRgb(110, 0.70, 0.55, rgb);
  fillRectPx(d, 300, 90, 150, 130, [rgb[0], rgb[1], rgb[2]]);
  hslToRgb(45, 0.90, 0.65, rgb);
  strokeTrianglePx(d, [[330, 330], [470, 445], [190, 445]], 5, [rgb[0], rgb[1], rgb[2]]);
  hslToRgb(270, 0.60, 0.70, rgb);
  const violet = [rgb[0], rgb[1], rgb[2]];
  fillRectPx(d, 60, 260, 90, 16, violet);
  fillRectPx(d, 165, 260, 55, 16, violet);
  fillRectPx(d, 60, 292, 70, 16, violet);
  return remapRange(new PixelBuffer(SIZE, SIZE, d), 70, 185);
}

function rampText() {
  const d = emptyBuffer();
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = Math.round(70 + (120 * x) / (SIZE - 1));
      setPx(d, (y * SIZE + x) * 4, v, v, v);
    }
  }
  const bars1 = [[70, 190, 60], [150, 190, 45], [210, 190, 70], [295, 190, 55], [365, 190, 40]];
  for (const [x, y, w] of bars1) fillRectPx(d, x, y, w, 22, [111, 111, 111]);
  const bars2 = [[80, 250, 45], [145, 250, 60], [225, 250, 35], [280, 250, 65], [365, 250, 50]];
  for (const [x, y, w] of bars2) fillRectPx(d, x, y, w, 22, [154, 154, 154]);
  fillCirclePx(d, 120, 350, 28, [128, 128, 128]);
  fillCirclePx(d, 380, 350, 28, [90, 90, 90]);
  return new PixelBuffer(SIZE, SIZE, d);
}

export const TEST_IMAGES = [
  { id: 'seg', title: 'Синтетическая сцена', desc: 'Точки, линии 0°/45°/90°/135° и перепады яркости — для всех детекторов', gen: sceneSeg },
  { id: 'segLow', title: 'Малоконтрастная сцена', desc: 'Диапазон яркостей сжат до 100–160 — линейное контрастирование и эквализация', gen: () => remapRange(sceneSeg(), 100, 160) },
  { id: 'noiseG', title: 'Гауссов шум (σ = 25)', desc: 'Проверка устойчивости детекторов к аддитивному шуму', gen: () => addGaussianNoise(sceneSeg(), 25) },
  { id: 'noiseSP', title: 'Импульсный шум (5%)', desc: '«Соль и перец» — порождает ложные отклики детектора точек', gen: () => addSaltPepper(sceneSeg(), 0.05) },
  { id: 'blur', title: 'Размытие (бокс 7×7)', desc: 'Ослабленные и размытые границы — влияние на операторы перепадов', gen: () => blurColor(sceneSeg(), 3) },
  { id: 'light', title: 'Неравномерная засветка', desc: 'Радиальное затенение сцены — проверка адаптивных методов', gen: () => unevenLight(sceneSeg()) },
  { id: 'colorLow', title: 'Цветная малоконтрастная', desc: 'Сравнение эквализации RGB и эквализации V в HSV', gen: colorLow },
  { id: 'ramp', title: 'Градиент с объектами', desc: 'Узкая гистограмма — сравнение глобальной эквализации и CLAHE', gen: rampText }
];

const cache = new Map();

export function getTestImage(id) {
  if (!cache.has(id)) {
    const item = TEST_IMAGES.find((t) => t.id === id);
    cache.set(id, item.gen());
  }
  return cache.get(id).clone();
}

export function downloadPB(pb, name) {
  const c = document.createElement('canvas');
  c.width = pb.width;
  c.height = pb.height;
  c.getContext('2d').putImageData(pb.toImageData(), 0, 0);
  const a = document.createElement('a');
  a.href = c.toDataURL('image/png');
  a.download = name;
  a.click();
}
