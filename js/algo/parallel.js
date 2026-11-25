import { hPassStripe, vPassStripe, conv2Stripe, gradStripe, linesStripe } from './stripes.js';

const RESULT_KEYS = {
  hpass: ['out'],
  vpass: ['out'],
  conv2: ['out'],
  grad: ['gx', 'gy', 'mag', 'dir'],
  lines: ['resp', 'dirIdx']
};

const SYNC_HANDLERS = {
  hpass: (p) => ({ out: hPassStripe(p.src, p.w, p.y0, p.y1, p.kernel, p.radius, p.edge) }),
  vpass: (p) => ({ out: vPassStripe(p.src, p.w, p.h, p.y0, p.y1, p.kernel, p.radius, p.edge) }),
  conv2: (p) => ({ out: conv2Stripe(p.src, p.w, p.h, p.y0, p.y1, p.kernel, p.ksize, p.edge) }),
  grad: (p) => gradStripe(p.src, p.w, p.h, p.y0, p.y1, p.op, p.edge),
  lines: (p) => linesStripe(p.src, p.w, p.h, p.y0, p.y1, p.edge)
};

class Pool {
  constructor(size) {
    this.size = size;
    this.idle = [];
    this.queue = [];
    this.pending = new Map();
    this.nextId = 1;
    const url = new URL('../workers/worker.js', import.meta.url);
    for (let i = 0; i < size; i++) {
      this.idle.push(new Worker(url, { type: 'module' }));
    }
    this.workers = this.idle.slice();
    for (const w of this.workers) {
      w.onmessage = (e) => this._onMessage(e);
      w.onerror = () => this._failAll();
    }
  }

  _onMessage(e) {
    const job = this.pending.get(e.data.id);
    if (!job) return;
    this.pending.delete(e.data.id);
    this.idle.push(job.worker);
    if (e.data.error) job.reject(new Error(e.data.error));
    else job.resolve(e.data);
    this._pump();
  }

  _failAll() {
    for (const job of this.pending.values()) job.reject(new Error('worker failure'));
    this.pending.clear();
    this.destroy();
  }

  destroy() {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.idle = [];
    this.queue = [];
  }

  run(msg) {
    return new Promise((resolve, reject) => {
      const m = { ...msg, id: this.nextId++ };
      this.queue.push({ resolve, reject, msg: m });
      this._pump();
    });
  }

  _pump() {
    while (this.queue.length && this.idle.length) {
      const worker = this.idle.pop();
      const job = this.queue.shift();
      this.pending.set(job.msg.id, { resolve: job.resolve, reject: job.reject, worker });
      worker.postMessage(job.msg);
    }
  }
}

let pool = null;
let mode = 'init';

function initPool() {
  if (mode !== 'init') return;
  try {
    const n = Math.min(8, navigator.hardwareConcurrency || 4);
    pool = new Pool(n);
    mode = 'workers';
  } catch {
    mode = 'sync';
  }
}

function destroyPool() {
  if (pool) pool.destroy();
  pool = null;
  mode = 'sync';
}

export function getMode() {
  if (mode === 'init') initPool();
  return mode === 'workers' ? `многопоточно, потоков: ${pool.size}` : 'однопоточно (воркеры недоступны)';
}

function splitRows(h, n) {
  const stripes = [];
  const base = Math.floor(h / n);
  let extra = h % n;
  let y = 0;
  for (let i = 0; i < n && y < h; i++) {
    const rows = base + (extra > 0 ? 1 : 0);
    if (extra > 0) extra--;
    stripes.push([y, y + rows]);
    y += rows;
  }
  return stripes;
}

async function runSync(op, src, w, h, params, edge, onProgress) {
  const keys = RESULT_KEYS[op];
  const out = {};
  for (const k of keys) {
    out[k] = k === 'dirIdx' ? new Uint8Array(w * h) : new Float32Array(w * h);
  }
  const stripes = splitRows(h, 8);
  for (let i = 0; i < stripes.length; i++) {
    const [y0, y1] = stripes[i];
    const res = SYNC_HANDLERS[op]({ ...params, w, h, y0, y1, edge, src });
    for (const k of keys) out[k].set(res[k], y0 * w);
    if (onProgress) onProgress((i + 1) / stripes.length);
    await new Promise((r) => setTimeout(r, 0));
  }
  return out;
}

async function runParallel(op, src, w, h, params, edge, onProgress) {
  const keys = RESULT_KEYS[op];
  const out = {};
  for (const k of keys) {
    out[k] = k === 'dirIdx' ? new Uint8Array(w * h) : new Float32Array(w * h);
  }
  const stripes = splitRows(h, pool.size * 2);
  let done = 0;
  await Promise.all(stripes.map(([y0, y1]) =>
    pool.run({ op, w, h, y0, y1, edge, params, src }).then((r) => {
      for (const k of keys) out[k].set(r.res[k], y0 * w);
      done++;
      if (onProgress) onProgress(done / stripes.length);
    })
  ));
  return out;
}

export async function runOp(op, src, w, h, params, edge, onProgress) {
  if (mode === 'init') initPool();
  if (mode === 'workers') {
    try {
      return await runParallel(op, src, w, h, params, edge, onProgress);
    } catch (err) {
      destroyPool();
      return runSync(op, src, w, h, params, edge, onProgress);
    }
  }
  return runSync(op, src, w, h, params, edge, onProgress);
}

export async function separableConv(src, w, h, kernel, radius, edge, onProgress) {
  const tmp = (await runOp('hpass', src, w, h, { kernel, radius }, edge,
    (p) => onProgress && onProgress(0.5 * p))).out;
  return (await runOp('vpass', tmp, w, h, { kernel, radius }, edge,
    (p) => onProgress && onProgress(0.5 + 0.5 * p))).out;
}
