import { segment } from './algo/segmentation.js';
import { PixelBuffer } from './core/pixel.js';
import { channelHists, linearContrast, equalizeRGB, equalizeHSV, clahe } from './algo/histogram.js';
import { TEST_IMAGES, getTestImage, downloadPB } from './algo/testImages.js';
import { getMode } from './algo/parallel.js';

const $ = (id) => document.getElementById(id);

let srcPB = null;
let currentViews = [];
let busy = false;
let testsBuilt = false;

const srcCanvas = $('srcCanvas');
const dstCanvas = $('dstCanvas');

function drawImage(canvas, image) {
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext('2d').putImageData(image, 0, 0);
}

function setSource(pb, name) {
  srcPB = pb;
  drawImage(srcCanvas, pb.toImageData());
  dstCanvas.width = pb.width;
  dstCanvas.height = pb.height;
  dstCanvas.getContext('2d').clearRect(0, 0, dstCanvas.width, dstCanvas.height);
  $('imgMeta').textContent = `${name} — ${pb.width} × ${pb.height}`;
}

function loadFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const maxDim = 1024;
    const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const cctx = c.getContext('2d');
    cctx.drawImage(img, 0, 0, w, h);
    const data = cctx.getImageData(0, 0, w, h);
    setSource(PixelBuffer.fromImageData(data), file.name);
    URL.revokeObjectURL(url);
    setStatus(`Загружено: ${file.name} (${img.width}×${img.height}${scale < 1 ? `, уменьшено до ${w}×${h}` : ''})`);
  };
  img.src = url;
}

function setStatus(text) {
  $('status').textContent = text;
}

function setProgress(p) {
  $('progressWrap').hidden = false;
  $('progressBar').style.width = `${Math.round(p * 100)}%`;
}

function hideProgress() {
  $('progressWrap').hidden = true;
  $('progressBar').style.width = '0%';
}

async function withBusy(fn) {
  if (busy) return;
  if (!srcPB) {
    setStatus('Сначала загрузите или выберите тестовое изображение');
    return;
  }
  busy = true;
  $('segApply').disabled = true;
  $('histApply').disabled = true;
  try {
    await fn();
  } catch (err) {
    setStatus(`Ошибка: ${err.message || err}`);
  } finally {
    busy = false;
    $('segApply').disabled = false;
    $('histApply').disabled = false;
    hideProgress();
  }
}

function fillViews(views) {
  currentViews = views;
  const sel = $('viewSelect');
  sel.innerHTML = '';
  for (const v of views) {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.label;
    sel.appendChild(opt);
  }
  $('viewRow').hidden = views.length <= 1;
  if (views.length) drawImage(dstCanvas, views[0].image);
}

async function applySegmentation() {
  await withBusy(async () => {
    const method = $('segMethod').value;
    const params = {
      method,
      edge: $('edgeMode').value,
      threshold: +$('segThreshold').value,
      sigma: +$('cannySigma').value / 10,
      low: +$('cannyLow').value,
      high: +$('cannyHigh').value,
      subpixel: $('cannySubpix').checked,
      normalize: $('autoNormalize').checked
    };
    if (method === 'canny' && params.low > params.high) {
      [params.low, params.high] = [params.high, params.low];
    }
    $('histRow').hidden = true;
    setStatus('Обработка…');
    const t0 = performance.now();
    const res = await segment(srcPB, params, setProgress);
    const dt = performance.now() - t0;
    fillViews(res.views);
    setStatus(`Готово за ${dt.toFixed(0)} мс (${getMode()}); ${res.info}`);
  });
}

function drawHistChart(canvas, hists) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#171d29';
  ctx.fillRect(0, 0, W, H);
  let max = 0;
  for (const h of [hists.r, hists.g, hists.b]) {
    for (let i = 0; i < 256; i++) if (h[i] > max) max = h[i];
  }
  if (max <= 0) return;
  ctx.strokeStyle = '#242e42';
  ctx.beginPath();
  for (let i = 1; i < 4; i++) {
    ctx.moveTo(0, (H / 4) * i);
    ctx.lineTo(W, (H / 4) * i);
  }
  ctx.stroke();
  const colors = ['rgba(255,90,90,0.65)', 'rgba(90,230,110,0.65)', 'rgba(100,140,255,0.65)'];
  const bw = W / 256;
  [hists.r, hists.g, hists.b].forEach((h, c) => {
    ctx.fillStyle = colors[c];
    for (let i = 0; i < 256; i++) {
      const bh = (h[i] / max) * (H - 6);
      if (bh > 0) ctx.fillRect(i * bw, H - bh, Math.max(1, bw), bh);
    }
  });
  ctx.fillStyle = '#8fa0bd';
  ctx.font = '11px sans-serif';
  ctx.fillText('0', 4, H - 4);
  ctx.fillText('255', W - 20, H - 4);
  ctx.fillText('частота', 6, 13);
}

async function applyHistogram() {
  await withBusy(async () => {
    const mode = $('histMode').value;
    $('viewRow').hidden = true;
    setStatus('Обработка…');
    setProgress(0.4);
    await new Promise((r) => setTimeout(r, 10));
    const t0 = performance.now();
    let result;
    let label;
    if (mode === 'linear') {
      let a = +$('linA').value, b = +$('linB').value;
      if ($('linAuto').checked) {
        const d = srcPB.data;
        let mn = 255, mx = 0;
        for (let p = 0; p < d.length; p += 4) {
          for (let c = 0; c < 3; c++) {
            const v = d[p + c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        }
        a = mn;
        b = mx;
      }
      if (b <= a) b = a + 1;
      result = linearContrast(srcPB, a, b);
      label = `линейное контрастирование, a = ${a}, b = ${b}`;
    } else if (mode === 'eqrgb') {
      result = equalizeRGB(srcPB);
      label = 'эквализация гистограммы (RGB, поканально)';
    } else if (mode === 'eqhsv') {
      result = equalizeHSV(srcPB);
      label = 'эквализация гистограммы (V в HSV)';
    } else {
      const tiles = +$('claheTiles').value;
      const clip = +$('claheClip').value / 10;
      result = clahe(srcPB, tiles, clip);
      label = `CLAHE, сетка ${tiles}×${tiles}, предел контраста ${clip.toFixed(1)}`;
    }
    const dt = performance.now() - t0;
    const before = channelHists(srcPB);
    const after = channelHists(result);
    drawImage(dstCanvas, result.toImageData());
    drawHistChart($('histBefore'), before);
    drawHistChart($('histAfter'), after);
    $('histRow').hidden = false;
    setStatus(`Готово за ${dt.toFixed(0)} мс (${getMode()}); ${label}`);
  });
}

function switchTab(name) {
  for (const b of $('tabs').querySelectorAll('button')) {
    b.classList.toggle('active', b.dataset.tab === name);
  }
  const isWork = name === 'seg' || name === 'hist';
  $('sidebar').hidden = !isWork;
  $('workspace').hidden = !isWork;
  $('segPanel').hidden = name !== 'seg';
  $('histPanel').hidden = name !== 'hist';
  $('testsTab').hidden = name !== 'tests';
  $('docsTab').hidden = name !== 'docs';
  if (name === 'seg') $('histRow').hidden = true;
  if (isWork && name === 'hist') $('viewRow').hidden = true;
  if (name === 'tests' && !testsBuilt) buildTests();
}

async function buildTests() {
  testsBuilt = true;
  const grid = $('testGrid');
  for (const t of TEST_IMAGES) {
    const card = document.createElement('div');
    card.className = 'test-card';
    const h3 = document.createElement('h3');
    h3.textContent = t.title;
    const canvas = document.createElement('canvas');
    const p = document.createElement('p');
    p.textContent = t.desc;
    const btns = document.createElement('div');
    btns.className = 'btns';
    const bLoad = document.createElement('button');
    bLoad.textContent = 'Загрузить';
    const bPng = document.createElement('button');
    bPng.textContent = 'Скачать PNG';
    bPng.disabled = true;
    btns.append(bLoad, bPng);
    card.append(h3, canvas, p, btns);
    grid.appendChild(card);
    bLoad.onclick = () => {
      const pb = getTestImage(t.id);
      setSource(pb, t.title);
      switchTab('seg');
      setStatus(`Загружено тестовое изображение: ${t.title}`);
    };
    setStatus(`Генерация базы: ${t.title}…`);
    await new Promise((r) => setTimeout(r, 10));
    const pb = getTestImage(t.id);
    drawImage(canvas, pb.toImageData());
    bPng.disabled = false;
    bPng.onclick = () => downloadPB(pb, `lab3_${t.id}.png`);
  }
  setStatus('База тестовых изображений сгенерирована');
}

function initControls() {
  const methodDefaults = { point: 400, line: 300, roberts: 100, prewitt: 200, sobel: 200, canny: 200 };
  const onMethodChange = () => {
    const m = $('segMethod').value;
    $('cannyCtrl').hidden = m !== 'canny';
    $('thresholdCtrl').hidden = m === 'canny';
    $('segThreshold').value = methodDefaults[m];
    $('thrOut').value = methodDefaults[m];
  };
  $('segMethod').addEventListener('change', onMethodChange);
  onMethodChange();

  const bind = (slider, out, fmt = (v) => v) => {
    slider.addEventListener('input', () => { out.value = fmt(+slider.value); });
  };
  bind($('segThreshold'), $('thrOut'));
  bind($('cannySigma'), $('sigmaOut'), (v) => (v / 10).toFixed(1));
  bind($('cannyLow'), $('lowOut'));
  bind($('cannyHigh'), $('highOut'));
  bind($('linA'), $('aOut'));
  bind($('linB'), $('bOut'));
  bind($('claheClip'), $('clipOut'), (v) => (v / 10).toFixed(1));

  const onHistMode = () => {
    $('linearCtrl').hidden = $('histMode').value !== 'linear';
    $('claheCtrl').hidden = $('histMode').value !== 'clahe';
    const auto = $('linAuto').checked;
    $('linA').disabled = auto;
    $('linB').disabled = auto;
  };
  $('histMode').addEventListener('change', onHistMode);
  $('linAuto').addEventListener('change', onHistMode);
  onHistMode();

  $('segApply').addEventListener('click', applySegmentation);
  $('histApply').addEventListener('click', applyHistogram);

  $('viewSelect').addEventListener('change', () => {
    const v = currentViews.find((x) => x.id === $('viewSelect').value);
    if (v) drawImage(dstCanvas, v.image);
  });

  for (const b of $('tabs').querySelectorAll('button')) {
    b.addEventListener('click', () => switchTab(b.dataset.tab));
  }

  $('fileInput').addEventListener('change', (e) => {
    if (e.target.files.length) loadFile(e.target.files[0]);
    e.target.value = '';
  });

  srcCanvas.addEventListener('dragover', (e) => e.preventDefault());
  srcCanvas.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
  });

  $('genAll').addEventListener('click', async () => {
    testsBuilt = false;
    $('testGrid').innerHTML = '';
    await buildTests();
  });

  $('downloadAll').addEventListener('click', async () => {
    for (const t of TEST_IMAGES) {
      downloadPB(getTestImage(t.id), `lab3_${t.id}.png`);
      await new Promise((r) => setTimeout(r, 250));
    }
  });

  $('modeInfo').textContent = `режим вычислений: ${getMode()}`;
}

function init() {
  initControls();
  setSource(getTestImage('seg'), 'Синтетическая сцена (база)');
  setStatus('Готово к работе. Загружено тестовое изображение из базы.');
}

init();
