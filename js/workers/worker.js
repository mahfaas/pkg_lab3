import { hPassStripe, vPassStripe, conv2Stripe, gradStripe, linesStripe } from '../algo/stripes.js';

const handlers = {
  hpass: (p) => ({ out: hPassStripe(p.src, p.w, p.y0, p.y1, p.kernel, p.radius, p.edge) }),
  vpass: (p) => ({ out: vPassStripe(p.src, p.w, p.h, p.y0, p.y1, p.kernel, p.radius, p.edge) }),
  conv2: (p) => ({ out: conv2Stripe(p.src, p.w, p.h, p.y0, p.y1, p.kernel, p.ksize, p.edge) }),
  grad: (p) => gradStripe(p.src, p.w, p.h, p.y0, p.y1, p.op, p.edge),
  lines: (p) => linesStripe(p.src, p.w, p.h, p.y0, p.y1, p.edge)
};

if (typeof self !== 'undefined' && typeof window === 'undefined') {
  self.onmessage = (e) => {
    const { id, op, w, h, y0, y1, edge, params } = e.data;
    try {
      const res = handlers[op]({ ...params, w, h, y0, y1, edge });
      const buffers = Object.values(res).map((a) => a.buffer);
      self.postMessage({ id, y0, y1, res }, buffers);
    } catch (err) {
      self.postMessage({ id, error: String(err && err.message ? err.message : err) });
    }
  };
}
