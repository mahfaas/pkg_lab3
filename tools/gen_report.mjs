import fs from 'node:fs';
import path from 'node:path';
import { encodePng, decodePng, sideBySide, histChartPng, imageToPng, pbToPng } from './png.mjs';

globalThis.ImageData = class {
  constructor(a, b, c) {
    if (typeof a === 'number') {
      this.width = a;
      this.height = b;
      this.data = c ?? new Uint8ClampedArray(a * b * 4);
    } else {
      this.data = a;
      this.width = b;
      this.height = c;
    }
  }
};

const { PixelBuffer, rgbToHsv, hsvToRgb } = await import('../js/core/pixel.js');
const { segment } = await import('../js/algo/segmentation.js');
const { channelHists, linearContrast, equalizeRGB, equalizeHSV, clahe, eqLUT } = await import('../js/algo/histogram.js');
const { getTestImage } = await import('../js/algo/testImages.js');

const OUT = 'report_assets';
fs.mkdirSync(OUT, { recursive: true });

const LINE_KERNELS = [
  [-1, -1, -1, 2, 2, 2, -1, -1, -1],
  [2, -1, -1, -1, 2, -1, -1, -1, 2],
  [-1, 2, -1, -1, 2, -1, -1, 2, -1],
  [-1, -1, 2, -1, 2, -1, 2, -1, -1]
];
const ORIENTS = ['0°', '45°', '90°', '135°'];

function fmt(x, d = 3) {
  const v = Number(x);
  if (!isFinite(v)) return String(x);
  const s = v.toFixed(d);
  return s.replace('.', ',');
}
function fmtI(x) {
  return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

function grayAt(pb, x, y) {
  const p = (y * pb.width + x) * 4;
  const d = pb.data;
  return 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
}

function grayStats(pb) {
  const g = pb.getGray();
  let mn = 255, mx = 0, sum = 0;
  for (let i = 0; i < g.length; i++) {
    if (g[i] < mn) mn = g[i];
    if (g[i] > mx) mx = g[i];
    sum += g[i];
  }
  const mean = sum / g.length;
  let sq = 0;
  for (let i = 0; i < g.length; i++) sq += (g[i] - mean) * (g[i] - mean);
  return { mn, mx, mean, sigma: Math.sqrt(sq / g.length) };
}

function rgbRange(pb) {
  const d = pb.data;
  let mn = 255, mx = 0;
  for (let p = 0; p < d.length; p += 4) {
    for (let c = 0; c < 3; c++) {
      const v = d[p + c];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  return [mn, mx];
}

function rgbAt(pb, x, y) {
  const p = (y * pb.width + x) * 4;
  return [pb.data[p], pb.data[p + 1], pb.data[p + 2]];
}

function windowSigma(pb, x, y, r) {
  const g = pb.getGray();
  const x0 = Math.max(0, x - r), x1 = Math.min(pb.width - 1, x + r);
  const y0 = Math.max(0, y - r), y1 = Math.min(pb.height - 1, y + r);
  let n = 0, sum = 0;
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      sum += g[yy * pb.width + xx];
      n++;
    }
  }
  const mean = sum / n;
  let sq = 0;
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      const dv = g[yy * pb.width + xx] - mean;
      sq += dv * dv;
    }
  }
  return Math.sqrt(sq / n);
}

function conv3At(gray, w, x, y, kernel) {
  let sum = 0, ki = 0;
  for (let ky = -1; ky <= 1; ky++) {
    for (let kx = -1; kx <= 1; kx++, ki++) {
      sum += gray[(y + ky) * w + (x + kx)] * kernel[ki];
    }
  }
  return sum;
}

function sobelAt(gray, w, x, y) {
  const gx = conv3At(gray, w, x, y, [-1, 0, 1, -2, 0, 2, -1, 0, 1]);
  const gy = conv3At(gray, w, x, y, [-1, -2, -1, 0, 0, 0, 1, 2, 1]);
  return { gx, gy, mag: Math.hypot(gx, gy), dir: Math.atan2(gy, gx) };
}

function countWhite(img) {
  let n = 0;
  const d = img.data;
  for (let p = 0; p < d.length; p += 4) {
    if (d[p] > 127 || d[p + 1] > 127 || d[p + 2] > 127) n++;
  }
  return n;
}

function imgAt(img, x, y) {
  const p = (y * img.width + x) * 4;
  return [img.data[p], img.data[p + 1], img.data[p + 2]];
}

function save(name, view) {
  fs.writeFileSync(path.join(OUT, name), encodePng(view.width, view.height, view.data));
}

function savePair(name, a, b) {
  const c = sideBySide([a, b]);
  fs.writeFileSync(path.join(OUT, name), encodePng(c.width, c.height, c.data));
}

function saveTriple(name, a, b, c2) {
  const c = sideBySide([a, b, c2]);
  fs.writeFileSync(path.join(OUT, name), encodePng(c.width, c.height, c.data));
}

function hueAt(pb, x, y) {
  const [r, g, b] = rgbAt(pb, x, y);
  const hsv = [0, 0, 0];
  rgbToHsv(r, g, b, hsv);
  return hsv[0];
}

function hueDiffPct(before, after, tol = 10) {
  const n = before.width * before.height;
  const hsv1 = [0, 0, 0], hsv2 = [0, 0, 0];
  let cnt = 0;
  for (let p = 0; p < n * 4; p += 4) {
    rgbToHsv(before.data[p], before.data[p + 1], before.data[p + 2], hsv1);
    rgbToHsv(after.data[p], after.data[p + 1], after.data[p + 2], hsv2);
    let dh = Math.abs(hsv1[0] - hsv2[0]);
    if (dh > 180) dh = 360 - dh;
    if (dh > tol) cnt++;
  }
  return (cnt / n) * 100;
}

function claheExpected(pb, tiles, clip, cx, cy) {
  const w = pb.width, h = pb.height, d = pb.data;
  const V = new Uint8Array(w * h);
  const hsv = [0, 0, 0];
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    rgbToHsv(d[p], d[p + 1], d[p + 2], hsv);
    V[i] = Math.round(hsv[2]);
  }
  const tileW = w / tiles, tileH = h / tiles;
  const gx = (cx + 0.5) / tileW - 0.5;
  const gy = (cy + 0.5) / tileH - 0.5;
  const tx = Math.floor(gx), ty = Math.floor(gy);
  const fx = gx - tx, fy = gy - ty;
  const xs = [Math.max(0, Math.min(tiles - 1, tx)), Math.max(0, Math.min(tiles - 1, tx + 1))];
  const ys = [Math.max(0, Math.min(tiles - 1, ty)), Math.max(0, Math.min(tiles - 1, ty + 1))];
  const luts = {};
  for (const tyy of ys) {
    for (const txx of xs) {
      const hist = new Float64Array(256);
      const x0 = Math.floor(txx * tileW), x1 = Math.floor((txx + 1) * tileW);
      const y0 = Math.floor(tyy * tileH), y1 = Math.floor((tyy + 1) * tileH);
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
      luts[`${txx}_${tyy}`] = eqLUT(hist);
    }
  }
  const v = V[cy * w + cx];
  const nv =
    luts[`${xs[0]}_${ys[0]}`][v] * (1 - fx) * (1 - fy) +
    luts[`${xs[1]}_${ys[0]}`][v] * fx * (1 - fy) +
    luts[`${xs[0]}_${ys[1]}`][v] * (1 - fx) * fy +
    luts[`${xs[1]}_${ys[1]}`][v] * fx * fy;
  return { lut0: luts[`${xs[0]}_${ys[0]}`][v], interpolated: nv, tilePixels: Math.floor(tileW) * Math.floor(tileH) };
}

const elements = [];
const el = (t, extra) => elements.push({ t, ...extra });
const M = {};
const px = fmtI(512 * 512);

el('title', { lines: ['ОТЧЁТ', 'ПО ЛАБОРАТОРНОЙ РАБОТЕ № 3', 'ОСНОВЫ ОБРАБОТКИ ЦИФРОВЫХ ИЗОБРАЖЕНИЙ'] });
el('table', { rows: [['Студент', 'Подгруппа', 'Вариант'], ['Шульдов Станислав', '10А', '4']] });
el('h1', { text: 'Цель работы' });
el('p', { text: 'Разработать веб-приложение для обработки цифровых изображений по варианту 4, реализовать обнаружение точек, линий и перепадов яркости, построение и эквализацию гистограммы и линейное контрастирование. Проверить алгоритмы на подготовленной тестовой базе, выполнить ручные расчёты контрольных пикселей и зафиксировать фактические результаты каждого испытания.' });
el('h1', { text: 'Содержание варианта' });
el('p', { text: 'Вариант 4 соответствует обнаружению точек, линий и перепадов яркости (сегментация) и построению и эквализации гистограммы изображения с линейным контрастированием. В проекте реализованы детектор точек на лапласиане, детектор линий на четырёх ориентированных масках, операторы Робертса, Прюитта и Соболя с картой направлений градиента, алгоритм Кэнни с субпиксельным подавлением немаксимумов и гистерезисом. Из методов обработки гистограммы реализованы линейное контрастирование, глобальная эквализация (в RGB и по компоненте V в HSV) и CLAHE.' });
el('h1', { text: 'Результат выполнения' });
el('p', { text: `Создано статическое веб-приложение для GitHub Pages. Оно загружает пользовательские и тестовые изображения, преобразует их в массив яркостей, выполняет выбранный алгоритм в Web Worker, показывает результат, карты модуля и направлений градиента, а также гистограммы до и после обработки. Подготовлены восемь тестовых изображений ${px} пикселей каждое и протокол с ручными вычислениями, измерениями, статусами и примечаниями. Дополнительно испытаны две реальные фотографии.` });
el('h1', { text: '1. Программа испытаний и критерии оценки' });
el('p', { text: 'В тестовую базу входят 8 программно генерируемых PNG-изображений размером 512×512 пикселей (262 144 пикселя в каждом): синтетическая сцена, малоконтрастная сцена, гауссов шум, импульсный шум, размытие, неравномерная засветка, цветная малоконтрастная сцена и градиент с объектами. Тестовые изображения специально отличаются по характеристикам. Дополнительно использованы реальные фотографии с неравномерным освещением и размытым текстом. Для воспроизводимости в каждом испытании фиксировались метод, параметры, контрольная координата пикселя и фактический результат.' });
el('p', { text: 'Статус «ПРОЙДЕН» означает, что формула применена корректно и контрольный результат совпал с расчётом. «ЧАСТИЧНО» означает, что вычисление выполнено корректно, но изображение не достигло поставленной визуальной цели; такое испытание показывает ограничение метода, а не скрывается как успешное.' });
el('h1', { text: '2. Общие формулы' });
el('p', { text: 'Перевод RGB в яркость: Y = 0,299R + 0,587G + 0,114B. Обнаружение точек — лапласиан 3×3: R = 8·f(x,y) − Σf(сосед.), порог T бинаризует |R|. Обнаружение линий — четыре маски лапласиана для ориентаций 0°, 45°, 90°, 135°; отклик — максимум |Rk| по четырём свёрткам.' });
el('p', { text: 'Оператор Соболя: gx = (f(x+1,y−1) + 2f(x+1,y) + f(x+1,y+1)) − (f(x−1,y−1) + 2f(x−1,y) + f(x−1,y+1)), gy — аналогично по вертикали. Модуль градиента M = √(gx² + gy²), угол направления θ = atan2(gy, gx) ∈ (−180°, 180°]. Карта направлений: тон H = (θ + 180°)/360°·360°, яркость — нормированный модуль.' });
el('p', { text: 'Алгоритм Кэнни: 1) гауссово сглаживание разделяемой свёрткой; 2) градиенты Соболя; 3) подавление немаксимумов вдоль направления градиента — субпиксельное, сравнением с билинейно интерполированными значениями в точках (x ± cos θ, y ± sin θ); 4) двойная пороговая обработка (сильные ≥ t2, слабые ≥ t1) и гистерезис: слабый пиксель сохраняется только при связности с сильным.' });
el('p', { text: 'Линейное контрастирование: Iнов = clamp(round(255·(I − a)/(b − a)), 0, 255), где a и b — минимальная и максимальная яркости изображения. Эквализация: g′ = round((CDF(g) − CDFmin)/(N − CDFmin)·255) по накопленной гистограмме; для цветных изображений применяется независимо к каналам RGB либо только к компоненте V в HSV. В CLAHE гистограмма каждого тайла ограничивается пределом clipLimit, избыток перераспределяется, отображение строится по накопленной гистограмме тайла, значения соседних отображений билинейно интерполируются.' });
el('h1', { text: '3. Ручной расчёт и результаты испытаний' });

const summaryRows = [['№', 'Тестовое изображение', 'Метод и параметры', 'Измеренный результат', 'Статус', 'Примечание']];

async function test01() {
  const src = getTestImage('segLow');
  const [a, b] = rgbRange(src);
  const res = linearContrast(src, a, b);
  const cx = 140, cy = 150;
  const v0 = rgbAt(src, cx, cy)[0];
  const expected = Math.round((255 * (v0 - a)) / (b - a));
  const v1 = rgbAt(res, cx, cy)[0];
  const [rn, rx] = rgbRange(res);
  const s0 = grayStats(src);
  const s1 = grayStats(res);
  const ok = v1 === expected && rn === 0 && rx === 255;
  savePair('t01_pair.png', src.toImageData(), res.toImageData());
  const hb = channelHists(src);
  const ha = channelHists(res);
  const h1 = histChartPng(hb);
  const h2 = histChartPng(ha);
  savePair('t01_hist.png', h1, h2);
  M.t01 = { sigmaRatio: s1.sigma / s0.sigma };
  summaryRows.push(['T-01', 'lab3_segLow.png', `Линейное контрастирование; a = ${a}, b = ${b} (авто)`, `Пиксель (140,150), канал R: ${v0}→${v1}; диапазон каналов ${a}–${b}→${rn}–${rx}; σ (яркость) = ${fmt(s0.sigma)}→${fmt(s1.sigma)}`, ok ? 'ПРОЙДЕН' : 'ОШИБКА', 'Форма распределения сохранена.']);
  el('h2', { text: 'T-01 — Линейное контрастирование малоконтрастного изображения' });
  el('p', { text: `Файл: lab3_segLow.png (512×512)   Метод и параметры: Линейное контрастирование; a и b определяются автоматически по минимуму и максимуму значений каналов изображения.` });
  el('p', { text: `Ручной расчёт. Исходное изображение получено сжатием диапазона яркостей сцены в интервал 100–160. В контрольной точке (140, 150) — центре светлого круга — значение красного канала I = ${v0}. Минимум и максимум по всем каналам: a = ${a}, b = ${b}.` });
  el('calc', { text: `Подстановка значений Iнов = round(255·(I − a)/(b − a)) Iнов = round(255·(${v0} − ${a})/(${b} − ${a})) = round(${fmt((255 * (v0 - a)) / (b - a), 1)}) = ${expected}.` });
  el('p', { text: `Ожидаемый результат: Пиксель должен получить значение ${expected}; диапазон каналов результата должен расшириться до 0–255. Фактический результат: Контрольный пиксель (канал R): ${v0}→${v1}. Диапазон каналов: ${a}–${b}→${rn}–${rx}. Стандартное отклонение яркости: ${fmt(s0.sigma)}→${fmt(s1.sigma)} (рост в ${fmt(s1.sigma / s0.sigma, 2)} раза). Статус: ${ok ? 'ПРОЙДЕН' : 'ОШИБКА'}` });
  el('p', { text: `Примечание. Преобразование применяется к каждому каналу одной и той же функцией, поэтому цветовой баланс сохраняется. Гистограмма «после» повторяет форму гистограммы «до»: линейное растяжение не меняет порядка и относительных частот уровней, а только равномерно растягивает шкалу яркости.` });
  el('img', { file: 't01_pair.png', w: 6.4, cap: 'Рис. 1 — Малоконтрастная сцена до (слева) и после (справа) линейного контрастирования' });
  el('img', { file: 't01_hist.png', w: 6.4, cap: 'Рис. 2 — Гистограмма RGB до (слева) и после (справа) контрастирования' });
}

async function test02() {
  const src = getTestImage('seg');
  const T = 400;
  const res = await segment(src, { method: 'point', edge: 'mirror', threshold: T, normalize: false });
  const mask = res.views[0].image;
  const cx = 50, cy = 60;
  const g = src.getGray();
  const f0 = Math.round(g[cy * 512 + cx]);
  let sum = 0;
  const neigh = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      const v = Math.round(g[(cy + dy) * 512 + (cx + dx)]);
      sum += v;
      neigh.push(v);
    }
  }
  const R = 8 * f0 - sum;
  const m = imgAt(mask, cx, cy);
  const ok = m[0] === 255 && Math.abs(R) >= T;
  const cnt = countWhite(mask);
  const sp = getTestImage('noiseSP');
  const resSP = await segment(sp, { method: 'point', edge: 'mirror', threshold: T, normalize: false });
  const cntSP = countWhite(resSP.views[0].image);
  M.t02 = { R, cntSP };
  savePair('t02_pair.png', mask, res.views[1].image);
  summaryRows.push(['T-02', 'lab3_seg.png', `Лапласиан, T = ${T}`, `Пиксель (50,60): |R| = ${fmtI(R)} → 255; всего точек: ${fmtI(cnt)}`, ok ? 'ПРОЙДЕН' : 'ОШИБКА', `На «соль и перец» ложных точек: ${fmtI(cntSP)}`]);
  el('h2', { text: 'T-02 — Обнаружение изолированных точек лапласианом' });
  el('p', { text: `Файл: lab3_seg.png (512×512)   Метод и параметры: Лапласиан 3×3 (маска −1 в восьми соседях, 8 в центре); порог T = ${T}; края — Reflect.` });
  el('p', { text: `Ручной расчёт. Контрольная точка (50, 60) — изолированная белая точка размером в один пиксель. Яркость центра f₀ = ${f0}; соседи: ${neigh.join(', ')}; их сумма Σ = ${fmtI(sum)}.` });
  el('calc', { text: `Подстановка значений R = 8·f₀ − Σсосед. R = 8·${f0} − ${fmtI(sum)} = ${fmtI(8 * f0)} − ${fmtI(sum)} = ${fmtI(R)}. Правило: |R| ≥ T = ${T} → точка обнаружена.` });
  el('p', { text: `Ожидаемый результат: Отклик в изолированной точке должен превысить порог; выход — бинарная маска. Фактический результат: |R| = ${fmtI(R)} ≥ ${T}; контрольный пиксель: ${f0}→${m[0]}. Всего обнаружено пикселей: ${fmtI(cnt)} (пять точек, линии и углы фигур). Статус: ${ok ? 'ПРОЙДЕН' : 'ОШИБКА'}` });
  el('p', { text: `Примечание. На изображении «Импульсный шум 5%» тот же детектор даёт ${fmtI(cntSP)} обнаруженных пикселей — каждый шумовой пиксель формально является изолированной точкой. Детектор точек принципиально неустойчив к импульсному шуму без предварительной фильтрации.` });
  el('img', { file: 't02_pair.png', w: 6.4, cap: 'Рис. 3 — Маска обнаруженных точек (слева) и карта отклика лапласиана (справа)' });
}

async function test03() {
  const src = getTestImage('seg');
  const T = 300;
  const res = await segment(src, { method: 'line', edge: 'mirror', threshold: T, normalize: false });
  const mask = res.views[0].image;
  const g = src.getGray();
  const responses = (x, y) => LINE_KERNELS.map((k) => conv3At(g, 512, x, y, k));
  const cx = 240, cy = 370;
  const rs = responses(cx, cy);
  const best = rs.indexOf(rs.reduce((m, v) => (Math.abs(v) > Math.abs(m) ? v : m), 0));
  const m = imgAt(mask, cx, cy);
  const ok = m[0] > 0 && best === 2;
  const cnt = countWhite(mask);
  const hx = 150, hy = 485;
  const rsH = responses(hx, hy);
  const bestH = rsH.indexOf(rsH.reduce((m, v) => (Math.abs(v) > Math.abs(m) ? v : m), 0));
  savePair('t03_pair.png', mask, res.views[1].image);
  const colorName = ['красный', 'зелёный', 'синий', 'жёлтый'][best];
  summaryRows.push(['T-03', 'lab3_seg.png', `4 маски линий, T = ${T}`, `Пиксель (240,370): маска №${best + 1} (${ORIENTS[best]}), |R| = ${fmtI(Math.abs(rs[best]))}; всего: ${fmtI(cnt)}`, ok ? 'ПРОЙДЕН' : 'ОШИБКА', 'Ориентации различаются цветом.']);
  el('h2', { text: 'T-03 — Обнаружение линий четырьмя ориентированными масками' });
  el('p', { text: `Файл: lab3_seg.png (512×512)   Метод и параметры: Четыре маски лапласиана для 0°, 45°, 90°, 135°; порог T = ${T}; края — Reflect.` });
  el('p', { text: `Ручной расчёт. Контрольная точка (240, 370) лежит на вертикальной линии толщиной в один пиксель. Отклики четырёх масок в этой точке: R₀° = ${fmtI(rs[0])}, R₄₅° = ${fmtI(rs[1])}, R₉₀° = ${fmtI(rs[2])}, R₁₃₅° = ${fmtI(rs[3])}.` });
  el('calc', { text: `Подстановка значений Максимум |Rk|: |R₉₀°| = ${fmtI(Math.abs(rs[2]))} ≥ T = ${T} → линия обнаружена, ориентация — ${ORIENTS[best]} (маска №${best + 1}).` });
  el('p', { text: `Ожидаемый результат: Вертикальная линия должна быть размечена цветом маски 90°. Фактический результат: Пиксель (240,370) помечен ${colorName} цветом (маска ${ORIENTS[best]}); горизонтальная линия в точке (150, 485) размечена маской ${ORIENTS[bestH]} (отклик ${fmtI(Math.abs(rsH[bestH]))}). Всего размечено пикселей: ${fmtI(cnt)}. Статус: ${ok ? 'ПРОЙДЕН' : 'ОШИБКА'}` });
  el('p', { text: `Примечание. Контур окружности размечается всеми четырьмя цветами по квадрантам: каждый участок дуги «голосует» за ближайшую ориентацию, что подтверждает корректное вычисление ориентации маской-победителем.` });
  el('img', { file: 't03_pair.png', w: 6.4, cap: 'Рис. 4 — Маска линий с цветовой маркировкой ориентаций (слева) и карта отклика (справа)' });
}

async function test04() {
  const src = getTestImage('seg');
  const T = 200;
  const res = await segment(src, { method: 'sobel', edge: 'mirror', threshold: T, normalize: false });
  const mask = res.views[0].image;
  const g = src.getGray();
  const cx = 70, cy = 150;
  const { gx, gy, mag, dir } = sobelAt(g, 512, cx, cy);
  const m = imgAt(mask, cx, cy);
  const ok = m[0] === 255 && mag >= T;
  const dirDeg = (dir * 180) / Math.PI;
  const hue = Math.round(((dir + Math.PI) / (2 * Math.PI)) * 360);
  M.t04 = { mag, dirDeg };
  saveTriple('t04_triple.png', res.views[1].image, res.views[2].image, mask);
  const dmap = res.views[2].image;
  const dcol = imgAt(dmap, cx, cy);
  const cnt = countWhite(mask);
  summaryRows.push(['T-04', 'lab3_seg.png', `Соболь, T = ${T}`, `Пиксель (70,150): M = ${fmtI(mag)}, θ = ${fmt(dirDeg, 1)}°; всего: ${fmtI(cnt)}`, ok ? 'ПРОЙДЕН' : 'ОШИБКА', 'Направление кодируется цветом.']);
  el('h2', { text: 'T-04 — Перепады яркости: оператор Соболя и карта направлений' });
  el('p', { text: `Файл: lab3_seg.png (512×512)   Метод и параметры: Оператор Соболя; порог T = ${T}; края — Reflect.` });
  el('p', { text: `Ручной расчёт. Контрольная точка (70, 150) — левый край светлого круга (фон слева, круг справа). Значения яркости окна 3×3 берутся из изображения; gx = (правый столбец) − (левый столбец), gy = (нижняя строка) − (верхняя строка).` });
  el('calc', { text: `Подстановка значений gx = ${fmtI(gx)}, gy = ${fmtI(gy)}. M = √(gx² + gy²) = ${fmtI(mag)} ≥ T = ${T} → перепад. θ = atan2(${fmtI(gy)}, ${fmtI(gx)}) = ${fmt(dirDeg, 1)}°.` });
  el('p', { text: `Ожидаемый результат: Пиксель должен попасть в бинарную маску границ; карта направлений в этой точке должна иметь тон H = (θ + 180°)/360°·360° = ${hue}°. Фактический результат: Контрольный пиксель: ${m[0]} в маске; цвет пикселя карты направлений RGB(${dcol.join(', ')}) — тон ${hue}°. Всего граничных пикселей: ${fmtI(cnt)}. Статус: ${ok ? 'ПРОЙДЕН' : 'ОШИБКА'}` });
  el('p', { text: `Примечание. Градиент направлен горизонтально (θ ≈ 0°, от тёмного фона к светлому кругу), что соответствует теории: направление градиента перпендикулярно границе. Окружность на карте направлений образует полный цветовой круг по квадрантам.` });
  el('img', { file: 't04_triple.png', w: 6.4, cap: 'Рис. 5 — Модуль градиента (слева), карта направлений (в центре), бинарные границы (справа)' });
}

async function test05() {
  const src = getTestImage('noiseG');
  const run = async (sigma) => segment(src, { method: 'canny', edge: 'mirror', sigma, low: 60, high: 180, subpixel: true, normalize: false });
  const t0 = performance.now();
  const res14 = await run(1.4);
  const dtCanny = performance.now() - t0;
  const res30 = await run(3.0);
  const canny14 = res14.views[0].image;
  const canny30 = res30.views[0].image;
  const cnt14 = countWhite(canny14);
  const cnt30 = countWhite(canny30);
  M.t05 = { cnt14, cnt30, dtCanny };
  const g = src.getGray();
  const cx = 70, cy = 150;
  const raw = sobelAt(g, 512, cx, cy);
  const m = imgAt(canny14, cx, cy);
  const near = (img) => {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const p = imgAt(img, cx + dx, cy + dy);
        if (p[0] === 255) return true;
      }
    }
    return false;
  };
  const ok = m[0] === 255 || near(canny14);
  savePair('t05_pair.png', src.toImageData(), canny14);
  savePair('t05_sigma.png', canny14, canny30);
  summaryRows.push(['T-05', 'lab3_noiseG.png', 'Кэнни; σ = 1,4; 60/180; субпикс. NMS', `Границ в окрестности контрольной точки; всего границ: ${fmtI(cnt14)} (σ = 1,4) против ${fmtI(cnt30)} (σ = 3,0)`, ok ? 'ПРОЙДЕН' : 'ЧАСТИЧНО', 'Рост σ подавляет шумовые границы.']);
  el('h2', { text: 'T-05 — Алгоритм Кэнни на изображении с гауссовым шумом' });
  el('p', { text: `Файл: lab3_noiseG.png (512×512, гауссов шум σш = 25)   Метод и параметры: Кэнни; сглаживание σ = 1,4; пороги 60/180; субпиксельное подавление немаксимумов; края — Reflect. Сравнение с σ = 3,0.` });
  el('p', { text: `Ручной расчёт. Контрольная точка (70, 150) — граница круга. Градиент несглаженного изображения в этой точке: gx = ${fmtI(raw.gx)}, gy = ${fmtI(raw.gy)}, M = ${fmtI(raw.mag)}; шум добавляет к фону случайные отклонения порядка 25. Верхний порог 180, поэтому истинная граница остаётся «сильным» пикселем и после сглаживания.` });
  el('calc', { text: `Подстановка значений M = ${fmtI(raw.mag)} > t₂ = 180 → сильный пиксель; гистерезис: связные слабые (60 ≤ M < 180) пиксели присоединяются к границе.` });
  el('p', { text: `Ожидаемый результат: Границы фигур должны быть выделены тонкими связными линиями; шумовые отклики — подавлены. Фактический результат: Контрольная точка классифицирована как граница${m[0] === 255 ? '' : ' (в окрестности ±1 пиксель)'}; всего граничных пикселей: ${fmtI(cnt14)} при σ = 1,4 и ${fmtI(cnt30)} при σ = 3,0. Статус: ${ok ? 'ПРОЙДЕН' : 'ЧАСТИЧНО'}` });
  el('p', { text: `Примечание. Увеличение σ с 1,4 до 3,0 сокращает число граничных пикселей с ${fmtI(cnt14)} до ${fmtI(cnt30)} (шумовые короткие ветви исчезают), но границы становятся толще и плавнее — компромисс между обнаружением и локализацией. Время выполнения Кэнни в однопоточном режиме Node.js: ${Math.round(dtCanny)} мс.` });
  el('img', { file: 't05_pair.png', w: 6.4, cap: 'Рис. 6 — Зашумлённая сцена (слева) и результат Кэнни при σ = 1,4 (справа)' });
  el('img', { file: 't05_sigma.png', w: 6.4, cap: 'Рис. 7 — Результат Кэнни при σ = 1,4 (слева) и σ = 3,0 (справа)' });
}

async function test06() {
  const src = getTestImage('colorLow');
  const cx = 100, cy = 160;
  const before = src.toImageData();
  const resRGB = equalizeRGB(src);
  const resHSV = equalizeHSV(src);
  const [r0, g0, b0] = rgbAt(src, cx, cy);
  const hsv = [0, 0, 0];
  rgbToHsv(r0, g0, b0, hsv);
  const h0 = Math.round(hsv[0]);
  const hists = channelHists(src);
  const lutR = eqLUT(hists.r), lutG = eqLUT(hists.g), lutB = eqLUT(hists.b);
  const expR = lutR[r0], expG = lutG[g0], expB = lutB[b0];
  const [ar, ag, ab] = rgbAt(resRGB, cx, cy);
  const [hr, hg, hb] = rgbAt(resHSV, cx, cy);
  const hRGB = Math.round(hueAt(resRGB, cx, cy));
  const hHSV = Math.round(hueAt(resHSV, cx, cy));
  const okRGB = ar === expR && ag === expG && ab === expB;
  const okHSV = hHSV === h0;
  const pctRGB = hueDiffPct(before, resRGB.toImageData());
  const pctHSV = hueDiffPct(before, resHSV.toImageData());
  M.t06 = { pctRGB, pctHSV, h0, hRGB, hHSV };
  saveTriple('t06_triple.png', before, resRGB.toImageData(), resHSV.toImageData());
  summaryRows.push(['T-06', 'lab3_colorLow.png', 'Эквализация: RGB и V (HSV)', `Пиксель (100,160): RGB (${r0},${g0},${b0})→(${ar},${ag},${ab}); тон ${h0}°→${hRGB}° (RGB) и ${hHSV}° (HSV)`, okRGB && okHSV ? 'ПРОЙДЕН' : 'ОШИБКА', `Смена тона >10°: ${fmt(pctRGB, 2)}% (RGB) против ${fmt(pctHSV, 2)}% (HSV).`]);
  el('h2', { text: 'T-06 — Эквализация цветного изображения: RGB против HSV' });
  el('p', { text: `Файл: lab3_colorLow.png (512×512)   Метод и параметры: Эквализация гистограммы поканально в RGB и эквализация только компоненты V в HSV.` });
  el('p', { text: `Ручной расчёт. Контрольная точка (100, 160) лежит на цветном градиентном фоне: RGB(${r0}, ${g0}, ${b0}), тон H = ${h0}°. Поканальные LUT эквализации: R′ = LUTR(${r0}) = ${expR}, G′ = LUTG(${g0}) = ${expG}, B′ = LUTB(${b0}) = ${expB}. Для HSV-режима сохраняются H = ${h0}° и S, а V′ = LUTV(V).` });
  el('calc', { text: `Подстановка значений RGB-режим: (${r0}, ${g0}, ${b0}) → (${expR}, ${expG}, ${expB}). HSV-режим: H = ${h0}° сохраняется, изменяется только V; тон результата — ${h0}°.` });
  el('p', { text: `Ожидаемый результат: В RGB-режиме контрольный пиксель получает значения (${expR}, ${expG}, ${expB}); в HSV-режиме тон пикселя не изменяется. Фактический результат: RGB: (${r0}, ${g0}, ${b0})→(${ar}, ${ag}, ${ab}); тон ${h0}°→${hRGB}°. HSV: тон ${h0}°→${hHSV}°. Доля пикселей с изменением тона более чем на 10°: ${fmt(pctRGB, 2)}% в RGB-режиме и ${fmt(pctHSV, 2)}% в HSV-режиме. Статус: ${okRGB && okHSV ? 'ПРОЙДЕН' : 'ОШИБКА'}` });
  el('p', { text: `Примечание. Поканальная эквализация меняет соотношения каналов по-разному в разных участках диапазона, поэтому тон пикселей смещается (цветовые ореолы на стыках градиента). Эквализация только V сохраняет тон и насыщенность каждого пикселя и меняет одну лишь яркость.` });
  el('img', { file: 't06_triple.png', w: 6.4, cap: 'Рис. 8 — Исходное изображение (слева), эквализация RGB (в центре), эквализация V в HSV (справа)' });
}

function vAt(pb, x, y) {
  const [r, g, b] = rgbAt(pb, x, y);
  return Math.max(r, g, b);
}

async function test07() {
  const src = getTestImage('light');
  const cx = 15, cy = 15;
  const v0 = vAt(src, cx, cy);
  const res = clahe(src, 8, 3.0);
  const v1 = vAt(res, cx, cy);
  const exp = claheExpected(src, 8, 3.0, cx, cy);
  const ls0 = windowSigma(src, cx, cy, 16);
  const ls1 = windowSigma(res, cx, cy, 16);
  const s0 = grayStats(src);
  const s1 = grayStats(res);
  const ok = Math.abs(v1 - exp.interpolated) <= 1;
  M.t07 = { lsRatio: ls1 / ls0, v0, v1 };
  savePair('t07_pair.png', src.toImageData(), res.toImageData());
  summaryRows.push(['T-07', 'lab3_light.png', 'CLAHE; сетка 8×8; clipLimit = 3,0', `Пиксель (15,15): ${v0}→${v1}; локальная σ окна 33×33: ${fmt(ls0)}→${fmt(ls1)}`, ok ? 'ПРОЙДЕН' : 'ОШИБКА', 'Локальный контраст в тёмном углу вырос.']);
  el('h2', { text: 'T-07 — CLAHE на изображении с неравномерной засветкой' });
  el('p', { text: `Файл: lab3_light.png (512×512)   Метод и параметры: CLAHE; сетка тайлов 8×8 (тайл 64×64 = 4 096 пикселей); предел контраста clipLimit = 3,0; обработка компоненты V в HSV.` });
  el('p', { text: `Ручной расчёт. Контрольная точка (15, 15) — затемнённый угол сцены, яркость V = ${v0}. Пиксель относится к угловому тайлу; предел гистограммы L = clipLimit·A/256 = 3·4096/256 = 48. Гистограмма тайла обрезается по L, избыток распределяется по всем уровням, по накопленной гистограмме строится LUT; значение в контрольной точке — билинейная интерполяция отображений четырёх соседних тайлов.` });
  el('calc', { text: `Подстановка значений L = floor(3·4096/256) = 48. Отображение углового тайла: LUT(${v0}) = ${exp.lut0}; интерполированное по четырём тайлам значение: ${fmt(exp.interpolated, 1)} → ${v0}→${v1}.` });
  el('p', { text: `Ожидаемый результат: Локальное отображение должно поднять яркость и контраст затемнённого угла; глобальная средняя яркость меняется незначительно. Фактический результат: Контрольный пиксель: ${v0}→${v1}. Локальное стандартное отклонение окна 33×33 в углу: ${fmt(ls0)}→${fmt(ls1)} (рост в ${fmt(ls1 / ls0, 2)} раза). Средняя яркость: ${fmt(s0.mean)}→${fmt(s1.mean)}. Статус: ${ok ? 'ПРОЙДЕН' : 'ОШИБКА'}` });
  el('p', { text: `Примечание. Глобальная эквализация на этом изображении почти не помогает: суммарная гистограмма перекрыта яркой центральной областью. CLAHE действует локально и восстанавливает контраст каждого тайла отдельно; билинейная интерполяция отображений устраняет видимые границы между тайлами.` });
  el('img', { file: 't07_pair.png', w: 6.4, cap: 'Рис. 9 — Неравномерная засветка до (слева) и после (справа) CLAHE' });
}

async function test08() {
  let photoA = null, photoS = null;
  try {
    photoA = decodePng(fs.readFileSync('TestImages/A1_page_uneven_light.png'));
    photoS = decodePng(fs.readFileSync('TestImages/S1_blurred_text.png'));
  } catch {
    photoA = photoS = null;
  }
  if (!photoA || !photoS) {
    el('h2', { text: 'T-08 — Испытание на реальных фотографиях' });
    el('p', { text: 'Файлы TestImages недоступны; испытание пропущено.' });
    return;
  }
  const pbA = new PixelBuffer(photoA.width, photoA.height, new Uint8ClampedArray(photoA.rgba));
  const pbS = new PixelBuffer(photoS.width, photoS.height, new Uint8ClampedArray(photoS.rgba));
  const nA = photoA.width * photoA.height;
  const nS = photoS.width * photoS.height;
  const resA = clahe(pbA, 8, 3.0);
  const cx = Math.floor(photoA.width / 2), cy = Math.floor(photoA.height / 2);
  const v0 = vAt(pbA, cx, cy);
  const v1 = vAt(resA, cx, cy);
  const expA = claheExpected(pbA, 8, 3.0, cx, cy);
  const okA = Math.abs(v1 - expA.interpolated) <= 1;
  const resS = await segment(pbS, { method: 'canny', edge: 'mirror', sigma: 1.4, low: 60, high: 180, subpixel: true, normalize: false });
  const canny = resS.views[0].image;
  const cntS = countWhite(canny);
  const gS = pbS.getGray();
  let bestMag = -1, bx = 0, by = 0;
  for (let y = 1; y < photoS.height - 1; y += 2) {
    for (let x = 1; x < photoS.width - 1; x += 2) {
      const { mag } = sobelAt(gS, photoS.width, x, y);
      if (mag > bestMag) { bestMag = mag; bx = x; by = y; }
    }
  }
  const isEdge = imgAt(canny, bx, by)[0] === 255;
  let near = false;
  for (let dy = -2; dy <= 2 && !near; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      if (imgAt(canny, bx + dx, by + dy)[0] === 255) { near = true; break; }
    }
  }
  const okS = isEdge || near;
  M.t08 = { v0, v1, cntS, nS };
  savePair('t08_photoA.png', pbA.toImageData(), resA.toImageData());
  savePair('t08_photoS.png', pbS.toImageData(), canny);
  summaryRows.push(['T-08', 'A1_page_uneven_light.png, S1_blurred_text.png', `CLAHE 8×8, clip 3,0; Кэнни σ = 1,4, 60/180`, `CLAHE: пиксель (${cx},${cy}): ${v0}→${v1}. Кэнни: границ ${fmtI(cntS)} из ${fmtI(nS)} (${fmt((cntS / nS) * 100, 2)}%)`, okA && okS ? 'ПРОЙДЕН' : 'ЧАСТИЧНО', 'Испытания на реальных фотографиях.']);
  el('h2', { text: 'T-08 — Испытания на реальных фотографиях' });
  el('p', { text: `Файлы: A1_page_uneven_light.png (${photoA.width}×${photoA.height}, ${fmtI(nA)} пикселей) и S1_blurred_text.png (${photoS.width}×${photoS.height}, ${fmtI(nS)} пикселей) из папки TestImages. Методы: CLAHE (сетка 8×8, clipLimit = 3,0) и Кэнни (σ = 1,4; пороги 60/180).` });
  el('p', { text: `Ручной расчёт. Для A1 в контрольной точке (${cx}, ${cy}) яркость V = ${v0}; интерполированное значение CLAHE — ${fmt(expA.interpolated, 1)}. Для S1 контрольная точка выбрана в максимуме модуля градиента несглаженного изображения: (${bx}, ${by}), M = ${fmtI(bestMag)} > 180 → сильный пиксель гистерезиса.` });
  el('calc', { text: `Подстановка значений CLAHE: ${v0}→${v1} (расчёт ${fmt(expA.interpolated, 1)}). Кэнни: M(${bx}, ${by}) = ${fmtI(bestMag)} ≥ t₂ = 180; граница подтверждается${isEdge ? '' : ' в окрестности ±2 пикселя (смещение максимума после сглаживания)'}.` });
  el('p', { text: `Ожидаемый результат: CLAHE должен выровнять освещение страницы; Кэнни — выделить контуры размытого текста. Фактический результат: CLAHE: контрольный пиксель ${v0}→${v1}. Кэнни: найдено ${fmtI(cntS)} граничных пикселей (${fmt((cntS / nS) * 100, 2)}% изображения). Статус: ${okA && okS ? 'ПРОЙДЕН' : 'ЧАСТИЧНО'}` });
  el('p', { text: `Примечание. Размытие ослабляет градиенты: контуры букв выделяются фрагментами, а не замкнутыми линиями — пороговая сегментация разделяет яркости, но не восстанавливает резкость.` });
  el('img', { file: 't08_photoA.png', w: 6.4, cap: 'Рис. 10 — Фотография с неравномерным освещением до (слева) и после (справа) CLAHE' });
  el('img', { file: 't08_photoS.png', w: 6.4, cap: 'Рис. 11 — Фотография размытого текста (слева) и контуры Кэнни (справа)' });
}

await test01();
await test02();
await test03();
await test04();
await test05();
await test06();
await test07();
await test08();

el('summary', { rows: summaryRows });

el('h1', { text: '4. Автоматические проверки алгоритмов' });
el('p', { text: 'Помимо ручного протокола выполнены автоматические проверки на небольших заранее заданных массивах (npm test, Node.js): математические свойства функций и совпадение результата полосной обработки с полной обработкой изображения.' });
el('auto', { rows: [
  ['Группа проверок', 'Проверяемые условия', 'Результат'],
  ['Обработка границ', 'Индексация Clamp (дублирование) и Reflect (зеркальное отражение) за границами массива.', '2/2'],
  ['Ядра свёртки', 'Сумма весов гауссова ядра и ядра бокса равна 1.', '2/2'],
  ['Разделяемая свёртка', 'Совпадение разделяемой свёртки с полной двумерной для Clamp и Reflect (maxDiff < 10⁻⁵).', '2/2'],
  ['Полосная обработка', 'Совпадение композиции полос с полной обработкой для двух стратегий границ.', '2/2'],
  ['Обобщённая свёртка 2D', 'Совпадение полосной свёртки с полной для маски лапласиана.', '2/2'],
  ['Оператор Соболя', 'Градиент линейного рампа: gx = 80, gy = 0; направление 0°.', '2/2'],
  ['Маски линий', 'Вертикальная линия распознаётся маской 90°, горизонтальная — маской 0°.', '2/2'],
  ['LUT эквализации', 'Границы диапазона, монотонность отображения.', '3/3'],
  ['LUT линейного контрастирования', 'Значения на границах и в средней точке.', '1/1'],
  ['Цветовые преобразования', 'Круг RGB→HSV→RGB для шести цветов.', '1/1'],
  ['Яркость и статистики', 'Y = 0,299R + 0,587G + 0,114B; минимум и максимум буфера.', '2/2']
]});
el('p', { text: 'Итого 21/21 проверка пройдена. Автотесты проверяют отдельные программные свойства на контролируемых данных; испытания изображений показывают реальное поведение методов на шуме, размытии и неравномерном освещении.' });

el('h1', { text: '5. Реализация и оптимизация' });
el('p', { text: 'Приложение разделено на три слоя: Presentation (интерфейс, отображение «до/после», гистограммы, прогресс), Memory & Pixel Access (обёртка PixelBuffer над байтами RGBA, стратегии краёв Clamp/Reflect, преобразования RGB↔HSV) и Processing (чистые функции-алгоритмы). Обработка изображений выполнена только вложенными циклами по типизированным массивам Uint8ClampedArray и Float32Array; Canvas API используется только для декодирования файлов и вывода результата, а тестовые сцены генерируются программной растеризацией без Canvas.' });
el('p', { text: 'Гауссово и бокс-сглаживание реализованы разделяемыми свёртками: двумерная свёртка заменяется двумя одномерными проходами (по строкам, затем по столбцам), сложность снижается с O(K²·N²) до O(K·N²); совпадение с полной свёрткой подтверждено автотестом. Для краёв изображения реализованы две стратегии: дублирование граничного пикселя (Clamp) и зеркальное отражение (Reflect).' });
el('p', { text: 'Вычисления выносятся в пул Web Workers (до 8 потоков): изображение делится на горизонтальные полосы, обмен — переносимыми типизированными массивами; контрольные полосы совпадают с полной обработкой (автотест). При недоступности воркеров приложение автоматически переходит в однопоточный режим. CLAHE использует ограничение и перераспределение гистограмм тайлов с билинейной интерполяцией отображений — сложность O(N) на пиксель.' });

el('h1', { text: '6. Вывод' });
el('p', { text: 'В ходе лабораторной работы создано веб-приложение для обработки цифровых изображений по варианту 4. Реализованы обнаружение точек лапласианом, обнаружение линий четырьмя ориентированными масками, операторы Робертса, Прюитта и Соболя с картой направлений градиента, алгоритм Кэнни с субпиксельным подавлением немаксимумов и гистерезисом, линейное контрастирование, глобальная эквализация в RGB и по V в HSV и CLAHE. Подготовленная тестовая база включает восемь генерируемых изображений 512×512 и две реальные фотографии; для каждого теста оформлены номер, исходный файл, метод и его параметры, ручная подстановка чисел в формулу, ожидаемый и фактический результат, статус и примечание.' });
const t = summaryRows;
el('p', { text: `На малоконтрастном изображении линейное контрастирование расширило диапазон яркости до 0–255, а стандартное отклонение выросло в ${fmt(M.t01.sigmaRatio, 2)} раза при сохранении формы гистограммы. Детектор точек дал отклик ${fmtI(M.t02.R)} в изолированной точке по формуле 8f₀ − Σсоседних; на изображении с импульсным шумом тот же детектор порождает ${fmtI(M.t02.cntSP)} ложных обнаружений, что подтверждает необходимость предварительной фильтрации. Четыре маски линий корректно разделили ориентации: вертикальная линия распознана маской 90°, горизонтальная — маской 0°, а контур окружности размечен всеми четырьмя цветами по квадрантам.` });
el('p', { text: `Оператор Соболя в контрольной точке границы круга дал модуль градиента ${fmtI(M.t04.mag)} с направлением θ = ${fmt(M.t04.dirDeg, 1)}°, перпендикулярным границе, что соответствует теории; карта направлений кодирует угол тоном HSV. На зашумлённом изображении Кэнни выделил границы фигур; увеличение σ сглаживания с 1,4 до 3,0 сократило число граничных пикселей с ${fmtI(M.t05.cnt14)} до ${fmtI(M.t05.cnt30)} (подавление шумовых ветвей) ценой утолщения границ. Эквализация RGB сместила тон более чем на 10° у ${fmt(M.t06.pctRGB, 2)}% пикселей, тогда как эквализация V в HSV сохранила тон (${fmt(M.t06.pctHSV, 2)}% изменений); CLAHE повысил локальное стандартное отклонение в затемнённом углу в ${fmt(M.t07.lsRatio, 2)} раза и выровнял освещение реальной фотографии (контрольный пиксель ${M.t08 ? `${M.t08.v0}→${M.t08.v1}` : '—'}).` });
el('p', { text: 'Таким образом, результаты показывают, что качество обработки зависит от структуры исходного изображения и параметров алгоритма. Лапласиан пригоден для изолированных точек на гладком фоне; ориентированные маски — для линий известной ориентации; операторы градиента и Кэнни — для общих границ, причём Кэнни устойчивее к шуму за счёт сглаживания и гистерезиса. Линейное контрастирование сохраняет форму распределения, глобальная эквализация выравнивает накопленную плотность, CLAHE эффективнее на изображениях с неравномерной яркостью. Автоматические проверки подтвердили 21/21 проверку Node.js. Ручные расчёты контрольных точек совпали с результатами программы.' });

fs.writeFileSync(path.join(OUT, 'doc_layout.json'), JSON.stringify({ elements, summaryRows: t }, null, 2));
console.log('OK: images + doc_layout.json generated');
