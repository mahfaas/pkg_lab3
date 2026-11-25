import { edgeIndex, EDGE_CLAMP, EDGE_MIRROR, PixelBuffer, rgbToHsv, hsvToRgb } from '../js/core/pixel.js';
import { convolve2D, convolveSeparable, gaussianKernel1D, boxKernel1D } from '../js/core/convolution.js';
import { hPassStripe, vPassStripe, conv2Stripe, gradStripe, linesStripe } from '../js/algo/stripes.js';
import { eqLUT, linearLUT } from '../js/algo/histogram.js';

let failures = 0;
function check(name, cond) {
  if (cond) console.log(`ok    ${name}`);
  else { console.error(`FAIL  ${name}`); failures++; }
}
function close(a, b, eps = 1e-3) {
  return Math.abs(a - b) <= eps;
}

check('edgeIndex clamp', edgeIndex(-5, 10, EDGE_CLAMP) === 0 && edgeIndex(15, 10, EDGE_CLAMP) === 10);
check('edgeIndex mirror', edgeIndex(-1, 10, EDGE_MIRROR) === 1 && edgeIndex(-2, 10, EDGE_MIRROR) === 2 && edgeIndex(12, 10, EDGE_MIRROR) === 8 && edgeIndex(11, 10, EDGE_MIRROR) === 9);

const gk = gaussianKernel1D(1.4);
const ksum = gk.kernel.reduce((a, b) => a + b, 0);
check(`gaussian kernel sums to 1 (${ksum.toFixed(6)})`, close(ksum, 1));

const W = 37, H = 23;
const src = new Float32Array(W * H);
for (let i = 0; i < src.length; i++) src[i] = (Math.sin(i * 0.3) + 1) * 100 + (i % 7);

function full2DFrom1D(k, r) {
  const k2 = new Float32Array((2 * r + 1) * (2 * r + 1));
  for (let i = 0; i <= 2 * r; i++) for (let j = 0; j <= 2 * r; j++) k2[i * (2 * r + 1) + j] = k[i] * k[j];
  return k2;
}

for (const edge of [EDGE_CLAMP, EDGE_MIRROR]) {
  const sep = convolveSeparable(src, W, H, gk.kernel, gk.radius, edge);
  const full = convolve2D(src, W, H, full2DFrom1D(gk.kernel, gk.radius), 2 * gk.radius + 1, edge);
  let maxDiff = 0;
  for (let i = 0; i < src.length; i++) maxDiff = Math.max(maxDiff, Math.abs(sep[i] - full[i]));
  check(`separable == full 2D gaussian (${edge}), maxDiff=${maxDiff.toFixed(5)}`, maxDiff < 0.01);

  const stripes = [];
  const rows = 5;
  for (let y = 0; y < H; y += rows) stripes.push([y, Math.min(y + rows, H)]);
  const tmp = new Float32Array(W * H);
  const dst = new Float32Array(W * H);
  for (const [y0, y1] of stripes) tmp.set(hPassStripe(src, W, y0, y1, gk.kernel, gk.radius, edge), y0 * W);
  for (const [y0, y1] of stripes) dst.set(vPassStripe(tmp, W, H, y0, y1, gk.kernel, gk.radius, edge), y0 * W);
  maxDiff2: {
    let d = 0;
    for (let i = 0; i < src.length; i++) d = Math.max(d, Math.abs(dst[i] - full[i]));
    check(`stripe composition == full 2D (${edge}), maxDiff=${d.toFixed(5)}`, d < 0.01);
  }
}

const pointK = new Float32Array([-1, -1, -1, -1, 8, -1, -1, -1, -1]);
for (const edge of [EDGE_CLAMP, EDGE_MIRROR]) {
  const full = convolve2D(src, W, H, pointK, 3, edge);
  const st = new Float32Array(W * H);
  for (let y = 0; y < H; y += 7) st.set(conv2Stripe(src, W, H, y, Math.min(y + 7, H), pointK, 3, edge), y * W);
  let d = 0;
  for (let i = 0; i < src.length; i++) d = Math.max(d, Math.abs(st[i] - full[i]));
  check(`conv2 stripe == full (${edge})`, d < 1e-4);
}

const ramp = new Float32Array(9 * 9);
for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) ramp[y * 9 + x] = x * 10;
const g = gradStripe(ramp, 9, 9, 3, 6, 'sobel', EDGE_CLAMP);
let gxOk = true;
for (let r = 0; r < 3; r++) {
  for (let x = 1; x < 8; x++) {
    const i = r * 9 + x;
    if (Math.abs(g.gx[i] - 80) > 1e-6 || Math.abs(g.gy[i]) > 1e-6) gxOk = false;
  }
}
check('sobel gradient of x-ramp (interior): gx=80, gy=0', gxOk);
check('ramp direction is 0 deg', Math.abs(g.dir[1 * 9 + 4]) < 1e-6);

const vline = new Float32Array(15 * 15);
for (let y = 0; y < 15; y++) for (let x = 0; x < 15; x++) vline[y * 15 + x] = x === 7 ? 255 : 20;
const ls = linesStripe(vline, 15, 15, 0, 15, EDGE_CLAMP);
check('vertical line detected as kernel #2 (90°)', ls.dirIdx[7 * 15 + 7] === 2 && ls.resp[7 * 15 + 7] > 500);

const hline = new Float32Array(15 * 15);
for (let y = 0; y < 15; y++) for (let x = 0; x < 15; x++) hline[y * 15 + x] = y === 7 ? 255 : 20;
const ls2 = linesStripe(hline, 15, 15, 0, 15, EDGE_CLAMP);
check('horizontal line detected as kernel #0 (0°)', ls2.dirIdx[7 * 15 + 7] === 0 && ls2.resp[7 * 15 + 7] > 500);

const hist = new Uint32Array(256);
hist[100] = 50;
hist[150] = 50;
const lut = eqLUT(hist);
check('eqLUT maps min used level to 0', lut[100] === 0);
check('eqLUT maps max used level to 255', lut[150] === 255);
let mono = true;
for (let i = 1; i < 256; i++) if (lut[i] < lut[i - 1]) mono = false;
check('eqLUT monotonic', mono);

const lin = linearLUT(50, 200);
check('linearLUT endpoints', lin[0] === 0 && lin[50] === 0 && lin[200] === 255 && lin[255] === 255 && lin[125] === Math.round(((125 - 50) * 255) / 150));

const hsv = [0, 0, 0];
const rgb = [0, 0, 0];
let hsvRound = true;
for (const [r, gg, b] of [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 40], [200, 50, 150], [17, 200, 90]]) {
  rgbToHsv(r, gg, b, hsv);
  hsvToRgb(hsv[0], hsv[1], hsv[2], rgb);
  if (!close(rgb[0], r, 1) || !close(rgb[1], gg, 1) || !close(rgb[2], b, 1)) hsvRound = false;
}
check('rgb->hsv->rgb roundtrip', hsvRound);

const pb = new PixelBuffer(2, 1);
pb.data.set([10, 20, 30, 255, 250, 200, 100, 255]);
const gray = pb.getGray();
check('luminance Y = 0.299R+0.587G+0.114B', close(gray[0], 0.299 * 10 + 0.587 * 20 + 0.114 * 30) && close(gray[1], 0.299 * 250 + 0.587 * 200 + 0.114 * 100));
const mnmx = pb.minMax();
check('minMax', mnmx[0] === 10 && mnmx[1] === 250);

const bk = boxKernel1D(2);
check('box kernel sums to 1', close(bk.kernel.reduce((a, b) => a + b, 0), 1) && bk.kernel.length === 5);

if (failures > 0) {
  console.error(`\n${failures} checks FAILED`);
  process.exit(1);
}
console.log('\nAll checks passed');
