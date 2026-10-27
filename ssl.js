// A tiny contrastive (SimCLR / NT-Xent) model that trains live in the browser.
// 8 hidden concepts in R^16 -> MLP 16→32→2 -> unit circle. No labels are used for
// training; the colors are revealed to the viewer only. When it converges, the
// cluster centroids become the staves of a Vegvísir.
(function () {
  'use strict';

  const D = 16, H = 32, K = 8, N = 256, B = 64, V = 2 * B;
  const NP = H * D + H + 2 * H + 2;            // W1, b1, W2, b2 packed in one array
  const oW1 = 0, oB1 = H * D, oW2 = oB1 + H, oB2 = oW2 + 2 * H;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function createModel(seed) {
    let rand = mulberry32(seed);
    const gauss = () => {
      let u = 0;
      while (!u) u = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
    };

    // ---- data: N samples around K hidden prototypes ----
    const X = new Float64Array(N * D), Y = new Uint8Array(N);
    const protos = [];
    for (let k = 0; k < K; k++) {
      const p = new Float64Array(D);
      for (let d = 0; d < D; d++) p[d] = gauss();
      protos.push(p);
    }
    for (let i = 0; i < N; i++) {
      const y = i % K;
      Y[i] = y;
      for (let d = 0; d < D; d++) X[i * D + d] = protos[y][d] + 0.45 * gauss();
    }

    // ---- params + Adam state ----
    const P = new Float64Array(NP), G = new Float64Array(NP);
    const M1 = new Float64Array(NP), M2 = new Float64Array(NP);
    const m = { step: 0, tau: 0.2, lr: 3e-3, X, Y, P, G, rand, gauss };

    function init() {
      for (let j = 0; j < H * D; j++) P[oW1 + j] = gauss() * Math.sqrt(1 / D);
      for (let j = 0; j < 2 * H; j++) P[oW2 + j] = gauss() * Math.sqrt(1 / H);
      P.fill(0, oB1, oB1 + H);
      P.fill(0, oB2, oB2 + 2);
      M1.fill(0); M2.fill(0);
      m.step = 0;
    }
    init();
    m.init = init;

    // forward one input (x at xs[xo]) -> h at hs[ho], unit u at us[uo]; returns ||z||
    function forward(xs, xo, hs, ho, us, uo) {
      for (let a = 0; a < H; a++) {
        let s = P[oB1 + a];
        const w = oW1 + a * D;
        for (let d = 0; d < D; d++) s += P[w + d] * xs[xo + d];
        hs[ho + a] = Math.tanh(s);
      }
      let z0 = P[oB2], z1 = P[oB2 + 1];
      for (let a = 0; a < H; a++) {
        z0 += P[oW2 + a] * hs[ho + a];
        z1 += P[oW2 + H + a] * hs[ho + a];
      }
      const n = Math.hypot(z0, z1) + 1e-9;
      us[uo] = z0 / n; us[uo + 1] = z1 / n;
      return n;
    }
    m.forward = forward;

    // two augmented views: random scale, jitter, feature masking
    function augment(i, dst, o) {
      const s = 0.8 + 0.4 * rand();
      for (let d = 0; d < D; d++) {
        dst[o + d] = rand() < 0.25 ? 0 : X[i * D + d] * s + 0.3 * gauss();
      }
    }
    m.augment = augment;

    const xs = new Float64Array(V * D), hs = new Float64Array(V * H);
    const us = new Float64Array(V * 2), ns = new Float64Array(V), gu = new Float64Array(V * 2);
    const S = new Float64Array(V * V);

    // one NT-Xent step; views (2k, 2k+1) are a positive pair
    m.train = function () {
      for (let k = 0; k < B; k++) {
        const i = (rand() * N) | 0;
        augment(i, xs, 2 * k * D);
        augment(i, xs, (2 * k + 1) * D);
      }
      for (let v = 0; v < V; v++) ns[v] = forward(xs, v * D, hs, v * H, us, v * 2);

      const it = 1 / m.tau;
      let loss = 0, align = 0;
      gu.fill(0);
      for (let i = 0; i < V; i++) {
        let mx = -Infinity;
        for (let j = 0; j < V; j++) {
          const s = j === i ? -Infinity : (us[2 * i] * us[2 * j] + us[2 * i + 1] * us[2 * j + 1]) * it;
          S[i * V + j] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j < V; j++) {
          const e = j === i ? 0 : Math.exp(S[i * V + j] - mx);
          S[i * V + j] = e;
          sum += e;
        }
        const p = i ^ 1;
        loss += mx + Math.log(sum) - (us[2 * i] * us[2 * p] + us[2 * i + 1] * us[2 * p + 1]) * it;
        // dL_i/du_i = (Σ P_ij u_j - u_p)/τ ;  dL_i/du_j = (P_ij - [j=p]) u_i/τ
        for (let j = 0; j < V; j++) {
          if (j === i) continue;
          const c = (S[i * V + j] / sum - (j === p ? 1 : 0)) * it;
          gu[2 * i] += c * us[2 * j]; gu[2 * i + 1] += c * us[2 * j + 1];
          gu[2 * j] += c * us[2 * i]; gu[2 * j + 1] += c * us[2 * i + 1];
        }
        if (i < p) {
          const dx = us[2 * i] - us[2 * p], dy = us[2 * i + 1] - us[2 * p + 1];
          align += dx * dx + dy * dy;
        }
      }

      // backprop through normalization, W2, tanh, W1
      G.fill(0);
      const dh = new Float64Array(H);
      for (let v = 0; v < V; v++) {
        const ux = us[2 * v], uy = us[2 * v + 1];
        const gx = gu[2 * v] / V, gy = gu[2 * v + 1] / V;
        const dot = ux * gx + uy * gy;
        const dz0 = (gx - ux * dot) / ns[v], dz1 = (gy - uy * dot) / ns[v];
        G[oB2] += dz0; G[oB2 + 1] += dz1;
        for (let a = 0; a < H; a++) {
          const h = hs[v * H + a];
          G[oW2 + a] += dz0 * h;
          G[oW2 + H + a] += dz1 * h;
          dh[a] = (P[oW2 + a] * dz0 + P[oW2 + H + a] * dz1) * (1 - h * h);
        }
        for (let a = 0; a < H; a++) {
          G[oB1 + a] += dh[a];
          const w = oW1 + a * D, x = v * D;
          for (let d = 0; d < D; d++) G[w + d] += dh[a] * xs[x + d];
        }
      }

      // Adam
      m.step++;
      const b1 = 0.9, b2 = 0.999;
      const c1 = 1 - Math.pow(b1, m.step), c2 = 1 - Math.pow(b2, m.step);
      for (let j = 0; j < NP; j++) {
        M1[j] = b1 * M1[j] + (1 - b1) * G[j];
        M2[j] = b2 * M2[j] + (1 - b2) * G[j] * G[j];
        P[j] -= m.lr * (M1[j] / c1) / (Math.sqrt(M2[j] / c2) + 1e-8);
      }

      let unif = 0, cnt = 0;
      for (let i = 0; i < V; i += 2) {
        for (let j = i + 2; j < V; j += 2) {
          const dx = us[2 * i] - us[2 * j], dy = us[2 * i + 1] - us[2 * j + 1];
          unif += Math.exp(-2 * (dx * dx + dy * dy));
          cnt++;
        }
      }
      return { loss: loss / V, align: align / B, unif: Math.log(unif / cnt) };
    };

    // embed all clean samples into E (N*2)
    const hTmp = new Float64Array(H);
    m.embed = function (E) {
      for (let i = 0; i < N; i++) forward(X, i * D, hTmp, 0, E, i * 2);
    };

    // per-concept centroids (viewer-side only) + convergence score in [0,1]
    m.stats = function (E) {
      const cx = new Float64Array(K), cy = new Float64Array(K), cn = new Float64Array(K);
      for (let i = 0; i < N; i++) {
        cx[Y[i]] += E[2 * i]; cy[Y[i]] += E[2 * i + 1]; cn[Y[i]]++;
      }
      const cents = [];
      let tight = 0;
      for (let k = 0; k < K; k++) {
        const x = cx[k] / cn[k], y = cy[k] / cn[k];
        const r = Math.hypot(x, y);
        tight += r / K;
        cents.push({ k, angle: Math.atan2(y, x), r });
      }
      const sorted = cents.map((c) => c.angle).sort((a, b) => a - b);
      let minGap = Infinity;
      for (let k = 0; k < K; k++) {
        const g = k === K - 1 ? sorted[0] + 2 * Math.PI - sorted[k] : sorted[k + 1] - sorted[k];
        if (g < minGap) minGap = g;
      }
      const gap = minGap / ((2 * Math.PI) / K);
      const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
      return { cents, tight, gap, score: ss(0.7, 0.95, tight) * ss(0.2, 0.6, gap) };
    };

    m.N = N; m.K = K; m.D = D; m.H = H;
    return m;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createModel };
    return;
  }
  window.SSL = { createModel };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initUI);
  else initUI();

  // ------------------------------------------------------------------ UI
  function initUI() {
    const hero = document.getElementById('ssl-canvas');
    if (!hero) return;
    const spark = document.getElementById('ssl-spark');
    const tip = document.getElementById('ssl-tip');
    const $ = (id) => document.getElementById(id);
    const TAU = Math.PI * 2;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const model = createModel(25);
    const { N, K, D } = model;
    const E = new Float64Array(N * 2);           // current embedding (target)
    const S = new Float64Array(N * 2);           // displayed embedding (smoothed)
    const off = new Float64Array(N);             // fixed radial jitter so the ring reads as a band
    for (let i = 0; i < N; i++) off[i] = (model.rand() - 0.5) * 0.14;

    let lossEma = null, alignEma = 0, unifEma = 0, best = Infinity, stall = 0;
    let status = 'training', convergedAt = 0, hist = [], histEvery = 5;
    let stats = null, hover = null, dirty = true, frame = 0, raf = 0, heroVisible = true;
    let colors = {};

    function readTheme() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n) => cs.getPropertyValue(n).trim();
      colors = { ink: v('--ink'), muted: v('--muted'), rule: v('--rule'), accent: v('--accent'), panel: v('--panel'), c: [] };
      for (let k = 0; k < K; k++) colors.c.push(v('--c' + k));
      dirty = true;
    }

    function reset() {
      model.init();
      lossEma = null; best = Infinity; stall = 0; hist = []; histEvery = 5;
      setStatus('training');
      model.embed(E);
      // keep S as-is so the old structure visibly shatters into the new random init
    }

    function setStatus(s) {
      status = s;
      if (s === 'converged') convergedAt = performance.now();
      document.documentElement.dataset.run = s;
      for (const el of document.querySelectorAll('[data-status]')) el.textContent = s;
    }

    function trainStep() {
      const r = model.train();
      lossEma = lossEma == null ? r.loss : 0.97 * lossEma + 0.03 * r.loss;
      alignEma = 0.95 * alignEma + 0.05 * r.align;
      unifEma = 0.95 * unifEma + 0.05 * r.unif;
      // keep the whole run in ≤160 samples: when full, halve resolution
      if (model.step % histEvery === 0) {
        hist.push(lossEma);
        if (hist.length >= 160) {
          hist = hist.filter((_, i) => i % 2 === 0);
          histEvery *= 2;
        }
      }
      model.embed(E);
      if (model.step % 200 === 0) {
        if (best - lossEma < 0.01) stall++;
        else stall = 0;
        best = Math.min(best, lossEma);
      }
      const done = (model.step >= 1200 && stall >= 3 && stats && stats.score > 0.9) || model.step >= 5000;
      if (done) setStatus('converged');
    }

    // smooth displayed positions toward the target; returns how much they moved
    function smooth(k) {
      let moved = 0;
      for (let i = 0; i < N; i++) {
        let x = S[2 * i] + (E[2 * i] - S[2 * i]) * k;
        let y = S[2 * i + 1] + (E[2 * i + 1] - S[2 * i + 1]) * k;
        const n = Math.hypot(x, y) || 1;
        x /= n; y /= n;
        moved += Math.abs(x - S[2 * i]) + Math.abs(y - S[2 * i + 1]);
        S[2 * i] = x; S[2 * i + 1] = y;
      }
      return moved;
    }

    // ---- canvas helpers ----
    function fit(c) {
      const r = c.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      const ctx = c.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w: r.width, h: r.height };
    }

    const ease = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);

    // stave end-glyphs, drawn in a frame where the stave runs along +x and ends at x
    const GLYPHS = [
      (c, x, s) => { // trident
        c.moveTo(x, -0.7 * s); c.lineTo(x, 0.7 * s);
        c.moveTo(x, -0.7 * s); c.lineTo(x + 0.6 * s, -0.7 * s);
        c.moveTo(x, 0.7 * s); c.lineTo(x + 0.6 * s, 0.7 * s);
        c.moveTo(x, 0); c.lineTo(x + 0.8 * s, 0);
      },
      (c, x, s) => { c.moveTo(x + 0.9 * s, 0); c.arc(x + 0.45 * s, 0, 0.45 * s, 0, TAU); }, // ring
      (c, x, s) => { // double crossbar
        c.moveTo(x, -0.6 * s); c.lineTo(x, 0.6 * s);
        c.moveTo(x - 0.5 * s, -0.45 * s); c.lineTo(x - 0.5 * s, 0.45 * s);
      },
      (c, x, s) => { // open fork
        c.moveTo(x, 0); c.lineTo(x + 0.8 * s, -0.6 * s);
        c.moveTo(x, 0); c.lineTo(x + 0.8 * s, 0.6 * s);
      },
      (c, x, s) => { // arrowhead
        c.moveTo(x, 0); c.lineTo(x + 0.5 * s, 0);
        c.moveTo(x + 0.5 * s, 0); c.lineTo(x - 0.2 * s, -0.6 * s);
        c.moveTo(x + 0.5 * s, 0); c.lineTo(x - 0.2 * s, 0.6 * s);
      },
      (c, x, s) => { // crossbar with rings
        c.moveTo(x, -0.55 * s); c.lineTo(x, 0.55 * s);
        c.moveTo(x + 0.2 * s, -0.75 * s); c.arc(x, -0.75 * s, 0.2 * s, 0, TAU);
        c.moveTo(x + 0.2 * s, 0.75 * s); c.arc(x, 0.75 * s, 0.2 * s, 0, TAU);
      },
      (c, x, s) => { // hooked bar
        c.moveTo(x - 0.5 * s, -0.7 * s); c.lineTo(x, -0.7 * s); c.lineTo(x, 0.7 * s); c.lineTo(x - 0.5 * s, 0.7 * s);
      },
      (c, x, s) => { // diamond
        c.moveTo(x, 0); c.lineTo(x + 0.5 * s, -0.45 * s); c.lineTo(x + s, 0); c.lineTo(x + 0.5 * s, 0.45 * s); c.closePath();
        c.moveTo(x - 0.6 * s, -0.4 * s); c.lineTo(x - 0.6 * s, 0.4 * s);
      },
    ];

    function drawVegvisir(c, cx, cy, R, a, lw) {
      if (!stats || a < 0.01) return;
      const r0 = R * 0.11, r1 = R * 0.74, g = R * 0.08;
      c.save();
      c.translate(cx, cy);
      c.strokeStyle = colors.ink;
      c.lineWidth = lw;
      c.lineCap = 'round';
      c.lineJoin = 'round';
      c.globalAlpha = 0.85 * a;
      c.beginPath(); c.arc(0, 0, r0, 0, TAU); c.stroke();
      c.beginPath(); c.arc(0, 0, r0 * 0.38, 0, TAU); c.stroke();
      const end = r0 + (r1 - r0) * ease(a * 1.25);
      for (const ce of stats.cents) {
        c.save();
        c.rotate(ce.angle);
        c.beginPath();
        c.moveTo(r0, 0); c.lineTo(end, 0);
        if (ce.k % 2) { const m = r0 + (end - r0) * 0.55; c.moveTo(m, -g * 0.4); c.lineTo(m, g * 0.4); }
        c.stroke();
        const ga = Math.max(0, (a - 0.6) / 0.4);
        if (ga > 0) {
          c.globalAlpha = 0.85 * a * ga;
          c.beginPath();
          GLYPHS[ce.k](c, end, g);
          c.stroke();
        }
        c.restore();
      }
      c.restore();
    }

    function pos(i, cx, cy, R) {
      const r = R * (1 + off[i]);
      return [cx + S[2 * i] * r, cy - S[2 * i + 1] * r];
    }

    function drawScene(c, w, h, cx, cy, R, dot, lw, withHover) {
      c.clearRect(0, 0, w, h);
      c.strokeStyle = colors.rule;
      c.lineWidth = 1;
      c.beginPath(); c.arc(cx, cy, R, 0, TAU); c.stroke();
      for (let t = 0; t < 72; t++) { // compass ticks
        const a = (t / 72) * TAU, l = t % 9 === 0 ? 7 : 3;
        c.beginPath();
        c.moveTo(cx + Math.cos(a) * (R * 1.12), cy + Math.sin(a) * (R * 1.12));
        c.lineTo(cx + Math.cos(a) * (R * 1.12 + l), cy + Math.sin(a) * (R * 1.12 + l));
        c.stroke();
      }
      drawVegvisir(c, cx, cy, R, stats ? stats.score : 0, lw);
      for (let i = 0; i < N; i++) {
        const [x, y] = pos(i, cx, cy, R);
        c.fillStyle = colors.c[model.Y[i]];
        c.globalAlpha = hover && withHover && hover.i !== i ? 0.35 : 0.9;
        c.beginPath(); c.arc(x, y, dot, 0, TAU); c.fill();
      }
      c.globalAlpha = 1;
      if (withHover && hover) drawHover(c, cx, cy, R, dot);
    }

    // ---- hover: show a point's two augmented views ----
    const vx = new Float64Array(2 * D), vh = new Float64Array(model.H), vu = new Float64Array(4);
    function drawHover(c, cx, cy, R, dot) {
      model.forward(vx, 0, vh, 0, vu, 0);
      model.forward(vx, D, vh, 0, vu, 2);
      const [px, py] = pos(hover.i, cx, cy, R);
      const col = colors.c[model.Y[hover.i]];
      const vr = R * 1.24;
      const a = [cx + vu[0] * vr, cy - vu[1] * vr], b = [cx + vu[2] * vr, cy - vu[3] * vr];
      c.strokeStyle = col;
      c.lineWidth = 1.2;
      c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(px, py); c.lineTo(a[0], a[1]); c.moveTo(px, py); c.lineTo(b[0], b[1]); c.stroke();
      c.setLineDash([]);
      for (const p of [a, b]) { c.beginPath(); c.arc(p[0], p[1], dot + 2.5, 0, TAU); c.stroke(); }
      c.fillStyle = col;
      c.beginPath(); c.arc(px, py, dot + 1.5, 0, TAU); c.fill();
      const cos = vu[0] * vu[2] + vu[1] * vu[3];
      const sub = (n) => String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[d]);
      tip.textContent = `x${sub(hover.i)} · view₁ ↔ view₂ · cos ${cos.toFixed(2)}`;
    }

    hero.addEventListener('pointermove', (e) => {
      const r = hero.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const { cx, cy, R } = heroGeom(r.width, r.height);
      let bi = -1, bd = 14 * 14;
      for (let i = 0; i < N; i++) {
        const [x, y] = pos(i, cx, cy, R);
        const d = (x - mx) * (x - mx) + (y - my) * (y - my);
        if (d < bd) { bd = d; bi = i; }
      }
      if (bi < 0) {
        if (hover) { hover = null; tip.hidden = true; dirty = true; }
        return;
      }
      if (!hover || hover.i !== bi) {
        hover = { i: bi };
        model.augment(bi, vx, 0);
        model.augment(bi, vx, D);
      }
      tip.hidden = false;
      tip.style.left = Math.min(mx + 12, r.width - 190) + 'px';
      tip.style.top = my + 14 + 'px';
      dirty = true;
      if (reduce) render();
    });
    hero.addEventListener('pointerleave', () => {
      hover = null; tip.hidden = true; dirty = true;
      if (reduce) render();
    });

    const heroGeom = (w, h) => ({ cx: w / 2, cy: h / 2, R: Math.min(w, h) * 0.34 });

    function drawSpark() {
      if (!spark) return;
      const { ctx: c, w, h } = fit(spark);
      c.clearRect(0, 0, w, h);
      if (hist.length < 2) return;
      let lo = Infinity, hi = -Infinity;
      for (const v of hist) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const span = hi - lo || 1;
      c.strokeStyle = colors.accent;
      c.lineWidth = 1.5;
      c.beginPath();
      hist.forEach((v, i) => {
        const x = (i / (hist.length - 1)) * w, y = 3 + (1 - (v - lo) / span) * (h - 6);
        i ? c.lineTo(x, y) : c.moveTo(x, y);
      });
      c.stroke();
    }

    function drawMetrics() {
      $('m-step').textContent = model.step.toLocaleString();
      $('m-loss').textContent = lossEma == null ? '—' : lossEma.toFixed(3);
      $('m-align').textContent = alignEma.toFixed(3);
      $('m-unif').textContent = unifEma.toFixed(3);
      $('m-tau').textContent = model.tau.toFixed(2);
      for (const el of document.querySelectorAll('[data-step]')) el.textContent = model.step.toLocaleString();
      for (const el of document.querySelectorAll('[data-loss]')) el.textContent = lossEma == null ? '—' : lossEma.toFixed(2);
    }

    function render() {
      stats = model.stats(S);
      if (heroVisible) {
        const { ctx, w, h } = fit(hero);
        const g = heroGeom(w, h);
        drawScene(ctx, w, h, g.cx, g.cy, g.R, 2.3, 1.4, true);
        drawSpark();
      }
      drawMetrics();
      dirty = false;
    }

    function tick() {
      raf = requestAnimationFrame(tick);
      frame++;
      if (status === 'training') trainStep();
      else if (performance.now() - convergedAt > 20000) reset();
      const moved = smooth(0.12);
      if (moved > 1e-3 || status === 'training' || dirty) render();
    }

    function trainToConvergence() {
      reset();
      while (status === 'training') { stats = model.stats(E); trainStep(); }
      S.set(E);
      render();
    }

    // ---- controls ----
    $('ssl-reinit').addEventListener('click', () => (reduce ? trainToConvergence() : reset()));
    const tauIn = $('ssl-tau');
    tauIn.addEventListener('input', () => {
      model.tau = parseFloat(tauIn.value);
      if (reduce) { drawMetrics(); return; }
      if (status === 'converged') { best = Infinity; stall = 0; setStatus('training'); }
    });
    if (reduce) tauIn.addEventListener('change', trainToConvergence);

    // ---- lifecycle ----
    const start = () => { if (!raf && !reduce) raf = requestAnimationFrame(tick); };
    const stop = () => { cancelAnimationFrame(raf); raf = 0; };
    document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((es) => { heroVisible = es[0].isIntersecting; dirty = true; }).observe(hero);
    }
    window.addEventListener('resize', () => { dirty = true; if (reduce) render(); });
    document.addEventListener('themechange', () => { readTheme(); if (reduce) render(); });

    readTheme();
    model.embed(E);
    S.set(E);
    if (reduce) trainToConvergence();
    else start();
  }
})();
