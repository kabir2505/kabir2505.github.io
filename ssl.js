// A tiny contrastive (SimCLR / NT-Xent) model that trains live in the browser.
// 8 hidden concepts in R^16 -> MLP 16→32→2 -> unit circle. No labels are used for
// training; the colors are revealed to the viewer only. When it converges, the
// cluster centroids become the points of a (JoJo-flavoured) star.
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
    m.aug = { jitter: 0.3, mask: 0.25 };            // augmentation strength (Emerald Splash raises it)
    m.active = new Uint8Array(N).fill(1);          // samples still in the data (The Hand erases concepts)
    let pool = Array.from({ length: N }, (_, i) => i);
    const refreshPool = () => { pool = []; for (let i = 0; i < N; i++) if (m.active[i]) pool.push(i); };
    m.erase = (k) => { for (let i = 0; i < N; i++) if (Y[i] === k) m.active[i] = 0; refreshPool(); };
    m.restoreAll = () => { m.active.fill(1); refreshPool(); };
    m.focus = null;                                // Star Platinum: when set, batches come only from these points
    m.concepts = () => { const seen = new Set(); for (let i = 0; i < N; i++) if (m.active[i]) seen.add(Y[i]); return [...seen]; };

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
    m.getParams = () => Float64Array.from(P);
    m.setParams = (q) => { P.set(q); M1.fill(0); M2.fill(0); m.step = 0; };
    // full training state, for Crazy Diamond's restore
    m.saveState = () => ({ P: Float64Array.from(P), M1: Float64Array.from(M1), M2: Float64Array.from(M2),
      step: m.step, tau: m.tau, aug: { ...m.aug }, active: Uint8Array.from(m.active) });
    m.loadState = (st) => {
      P.set(st.P); M1.set(st.M1); M2.set(st.M2);
      m.step = st.step; m.tau = st.tau; m.aug = { ...st.aug };
      m.active.set(st.active); refreshPool();
    };

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
      const s = 0.8 + 0.4 * rand(), { jitter, mask } = m.aug;
      for (let d = 0; d < D; d++) {
        dst[o + d] = rand() < mask ? 0 : X[i * D + d] * s + jitter * gauss();
      }
    }
    m.augment = augment;

    const xs = new Float64Array(V * D), hs = new Float64Array(V * H);
    const us = new Float64Array(V * 2), ns = new Float64Array(V), gu = new Float64Array(V * 2);
    const S = new Float64Array(V * V);

    // one NT-Xent step; views (2k, 2k+1) are a positive pair
    m.train = function () {
      for (let k = 0; k < B; k++) {
        const src = m.focus || pool;
        const i = src[(rand() * src.length) | 0];
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

    // fixed augmented view pairs for some points, so agreement can be measured before and after training
    m.views = (idxs, draws = 2) => {
      const data = new Float64Array(idxs.length * draws * 2 * D);
      idxs.forEach((i, n) => { for (let v = 0; v < draws * 2; v++) augment(i, data, (n * draws * 2 + v) * D); });
      return { idxs, draws, data };
    };
    // mean cosine between each point's paired views under the current weights
    const hA = new Float64Array(H), uA = new Float64Array(4);
    m.agreeOn = ({ idxs, draws, data }) => {
      const per = new Float64Array(idxs.length);
      let mean = 0;
      for (let n = 0; n < idxs.length; n++) {
        let c = 0;
        for (let d = 0; d < draws; d++) {
          const o = (n * draws * 2 + d * 2) * D;
          forward(data, o, hA, 0, uA, 0);
          forward(data, o + D, hA, 0, uA, 2);
          c += uA[0] * uA[2] + uA[1] * uA[3];
        }
        per[n] = c / draws;
        mean += per[n] / idxs.length;
      }
      return { per, mean };
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
        if (!m.active[i]) continue;
        cx[Y[i]] += E[2 * i]; cy[Y[i]] += E[2 * i + 1]; cn[Y[i]]++;
      }
      const cents = [];
      let tight = 0;
      for (let k = 0; k < K; k++) {
        if (!cn[k]) continue;                      // concept erased
        const x = cx[k] / cn[k], y = cy[k] / cn[k];
        cents.push({ k, angle: Math.atan2(y, x), r: Math.hypot(x, y) });
      }
      const n = cents.length;
      for (const c of cents) tight += c.r / n;
      const sorted = cents.map((c) => c.angle).sort((a, b) => a - b);
      let minGap = Infinity;
      for (let k = 0; k < n; k++) {
        const g = k === n - 1 ? sorted[0] + 2 * Math.PI - sorted[k] : sorted[k + 1] - sorted[k];
        if (g < minGap) minGap = g;
      }
      const gap = minGap / ((2 * Math.PI) / n);
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
    const logEl = document.getElementById('train-log');
    const tbc = document.getElementById('tbc');
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
    let status = 'training', convergedAt = 0, frozenUntil = 0;
    let stopStart = -1e9, rippleUntil = 0, stopToken = 0;  // ZA WARUDO + Hamon effects
    let heavenUntil = 0, rewinding = null, P0 = null, trail = [];  // Made in Heaven, Bites the Dust
    let echoesUntil = 0, echoesTau = 0.2, emeraldUntil = 0, fxToken = 0;  // Echoes Act 3, Emerald Splash
    const erasedAt = {};                            // The Hand: concept -> when it was erased (for the fade)
    let lastErase = -1e9, cp = null;                // cp: checkpoint taken before the last Stand (Crazy Diamond)
    let rushing = null, rushTargets = null;         // Star Platinum's ORA ORA rush
    const AUG = { jitter: 0.3, mask: 0.25 };
    const wrap = hero.closest('.canvas-wrap');
    let stats = null, hover = null, dirty = true, frame = 0, raf = 0, heroVisible = true;
    let colors = {};

    function readTheme() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n) => cs.getPropertyValue(n).trim();
      colors = { ink: v('--ink'), paper: v('--paper'), muted: v('--muted'), rule: v('--rule'), accent: v('--accent'), panel: v('--panel'), gold: v('--gold'), magenta: v('--magenta'), c: [] };
      for (let k = 0; k < K; k++) colors.c.push(v('--c' + k));
      dirty = true;
    }

    // append a line to the train pane's log stream, keeping the last few
    function log(text, cls) {
      if (!logEl) return;
      const line = document.createElement('div');
      if (cls) line.className = cls;
      line.textContent = text;
      logEl.append(line);
      while (logEl.childElementCount > 14) logEl.firstElementChild.remove();
    }

    // ZA WARUDO: freeze training and motion for `ms`, invert the canvas, then resume
    function timeStop(ms, then) {
      const token = ++stopToken;
      const now = performance.now();
      frozenUntil = now + ms;
      stopStart = now;
      convergedAt += ms;                               // frozen time doesn't count toward the auto re-init
      if (wrap && !reduce) wrap.classList.add('timestop');
      log('⏸ ZA WARUDO — time has stopped', 'warn');
      dirty = true;
      setTimeout(() => {
        if (token !== stopToken) return;              // a reset happened meanwhile
        if (wrap) wrap.classList.remove('timestop');
        log('▶ time resumes', 'warn');
        if (then) then();
        dirty = true;
      }, ms);
    }

    function reset() {
      model.init();
      lossEma = null; best = Infinity; stall = 0;
      log('↻ re-initializing weights…', 'warn');
      if (tbc) tbc.classList.remove('show');
      cancelEffects();
      model.restoreAll();
      for (const k in erasedAt) delete erasedAt[k];
      setStatus('training');
      model.embed(E);
      remember();
      // keep S as-is so the old structure visibly shatters into the new random init
    }

    // stop every running Stand effect (time-stop, rewind, Made in Heaven, Echoes, Emerald)
    function cancelEffects() {
      stopToken++; fxToken++;                          // pending timeouts become no-ops
      frozenUntil = 0; rippleUntil = 0; heavenUntil = 0;
      if (rewinding) { clearInterval(rewinding.timer); rewinding = null; }
      if (rushing) { clearInterval(rushing); rushing = null; rushTargets = null; model.focus = null; }
      if (echoesUntil) { model.tau = echoesTau; echoesUntil = 0; }
      if (emeraldUntil) { model.aug = { ...AUG }; emeraldUntil = 0; }
      if (wrap) wrap.classList.remove('timestop', 'kingcrimson', 'bitesdust', 'echoes', 'emerald', 'crazydiamond', 'starplatinum');
    }

    // the run's starting weights + a coarse trajectory of the embedding, for Bites the Dust
    function remember() {
      P0 = model.getParams();
      trail = [Float32Array.from(E)];
    }

    function setStatus(s) {
      const was = status;
      status = s;
      if (s === 'converged') {
        convergedAt = performance.now();
        if (was !== 'converged') {
          // ZA WARUDO: time stops, then the checkpoint, "To Be Continued" and Hamon ripples
          const finish = () => {
            log(`✓ checkpoint saved: star.pt (step ${model.step.toLocaleString()})`, 'ok');
            if (tbc) tbc.classList.add('show');
            rippleUntil = performance.now() + 10000;
            dirty = true;
          };
          if (reduce) finish();
          else timeStop(1800, finish);
          document.dispatchEvent(new CustomEvent('ssl:converged', { detail: { step: model.step, loss: lossEma } }));
        }
      }
      document.documentElement.dataset.run = s;
      for (const el of document.querySelectorAll('[data-status]')) el.textContent = s;
    }

    function trainStep() {
      const r = model.train();
      lossEma = lossEma == null ? r.loss : 0.97 * lossEma + 0.03 * r.loss;
      alignEma = 0.95 * alignEma + 0.05 * r.align;
      unifEma = 0.95 * unifEma + 0.05 * r.unif;
      if (model.step % 50 === 0) {
        log(`step ${String(model.step).padStart(4)}  loss ${lossEma.toFixed(3)}  align ${alignEma.toFixed(3)}`);
      }
      model.embed(E);
      if (model.step % 25 === 0) {
        trail.push(Float32Array.from(E));
        if (trail.length > 240) trail = trail.filter((_, i) => i % 2 === 0);
      }
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

    // JoJo: the learned cluster centroids become the points of a star (an {8/3} star polygon),
    // drawn in as the model converges, with a small Joestar-style star in the middle.
    function drawStar(c, cx, cy, R, a, lw) {
      if (!stats || a < 0.01) return;
      const angles = stats.cents.map((ce) => ce.angle).sort((x, y) => x - y);
      const n = angles.length, r = R * 0.8;
      const pt = (i) => [cx + Math.cos(angles[i % n]) * r, cy - Math.sin(angles[i % n]) * r];
      const step = starStep(n);                    // {8/3}, {7/3}, {5/2}… or a polygon
      const p = ease(a * 1.2) * n;                 // how many of the n edges are drawn so far
      c.save();
      c.strokeStyle = colors.ink;
      c.lineWidth = lw;
      c.lineJoin = 'round';
      c.globalAlpha = 0.75 * a;
      c.beginPath();
      let [x0, y0] = pt(0);
      c.moveTo(x0, y0);
      for (let e = 0; e < Math.ceil(p); e++) {
        const [x1, y1] = pt((e + 1) * step);
        const f = Math.min(1, p - e);
        c.lineTo(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f);
        [x0, y0] = [x1, y1];
      }
      c.stroke();
      const ga = Math.max(0, (a - 0.6) / 0.4);
      if (ga > 0) {
        const rs = R * 0.13;
        c.globalAlpha = ga;
        c.fillStyle = colors.gold;
        c.beginPath();
        for (let k = 0; k < 10; k++) {
          const rr = k % 2 ? rs * 0.42 : rs, th = -Math.PI / 2 + (k * Math.PI) / 5;
          k ? c.lineTo(cx + Math.cos(th) * rr, cy + Math.sin(th) * rr) : c.moveTo(cx + Math.cos(th) * rr, cy + Math.sin(th) * rr);
        }
        c.closePath();
        c.fill();
      }
      c.restore();
    }

    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    function starStep(n) {
      for (let st = Math.floor((n - 1) / 2); st >= 2; st--) if (gcd(n, st) === 1) return st;
      return 1;
    }

    // ゴゴゴ: menacing glyphs drifting around the ring while the model is still training
    const MENACE = Array.from({ length: 7 }, (_, i) => ({ a: (i / 7) * TAU + 0.4, seed: i * 0.137 }));
    function drawMenacing(c, cx, cy, R, intensity) {
      if (intensity < 0.02 || reduce) return;
      const t = performance.now() * 0.00035;
      c.save();
      c.fillStyle = colors.magenta;
      c.font = `700 ${Math.round(R * 0.17)}px "Hiragino Sans", "Noto Sans JP", sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      for (const m of MENACE) {
        const ph = (t + m.seed) % 1;
        const r = R * (1.22 + 0.06 * Math.sin(m.seed * 40));
        c.globalAlpha = intensity * 0.55 * Math.sin(Math.PI * ph);
        c.save();
        c.translate(cx + Math.cos(m.a) * r, cy - Math.sin(m.a) * r - ph * R * 0.18);
        c.rotate(-0.15);
        c.fillText('ゴ', 0, 0);
        c.restore();
      }
      c.restore();
    }

    function pos(i, cx, cy, R) {
      const r = R * (1 + off[i]);
      return [cx + S[2 * i] * r, cy - S[2 * i + 1] * r];
    }

    // the shockwave that sweeps out when time stops
    function drawShockwave(c, cx, cy, R) {
      const t = (performance.now() - stopStart) / 900;
      if (reduce || t < 0 || t >= 1) return;
      c.save();
      c.strokeStyle = colors.ink;
      c.globalAlpha = 1 - t;
      c.lineWidth = 6 - 5 * t;
      c.beginPath(); c.arc(cx, cy, R * (0.1 + 1.6 * ease(t)), 0, TAU); c.stroke();
      c.restore();
    }

    // Hamon: sunlight-yellow ripples pulsing out from the centre star after convergence
    function drawRipples(c, cx, cy, R) {
      const now = performance.now();
      if (reduce || now >= rippleUntil) return;
      const fade = Math.min(1, (rippleUntil - now) / 1500);
      c.save();
      c.strokeStyle = colors.gold;
      c.lineWidth = 2;
      for (let i = 0; i < 3; i++) {
        const ph = (now / 1200 + i / 3) % 1;
        c.globalAlpha = (1 - ph) * 0.6 * fade;
        c.beginPath(); c.arc(cx, cy, R * 0.13 + ph * R * 0.8, 0, TAU); c.stroke();
      }
      c.restore();
    }

    // Emerald Splash: green shards glinting around the ring
    function drawEmerald(c, cx, cy, R) {
      if (reduce || performance.now() >= emeraldUntil) return;
      c.save();
      c.fillStyle = '#1fbf6a';
      for (let k = 0; k < 14; k++) {
        const a = Math.random() * TAU, r = R * (1.05 + Math.random() * 0.3), s = 2 + Math.random() * 3;
        const x = cx + Math.cos(a) * r, y = cy - Math.sin(a) * r;
        c.globalAlpha = 0.5 + Math.random() * 0.5;
        c.beginPath(); c.moveTo(x, y - s); c.lineTo(x + s * 0.6, y); c.lineTo(x, y + s); c.lineTo(x - s * 0.6, y); c.closePath(); c.fill();
      }
      c.restore();
    }

    // Star Platinum: rings on the targeted points, punch streaks and ORA bursts while the rush lasts
    function drawRush(c, cx, cy, R, dot) {
      if (reduce || !rushTargets) return;
      const t = [...rushTargets];
      c.save();
      c.strokeStyle = '#7b5cff';
      c.lineWidth = 1.2;
      for (const i of t) {
        const [x, y] = pos(i, cx, cy, R);
        c.globalAlpha = 0.7;
        c.beginPath(); c.arc(x, y, dot + 3, 0, TAU); c.stroke();
      }
      c.lineWidth = 2.2;
      for (let k = 0; k < 6; k++) {                    // punches flying in
        const [x, y] = pos(t[(Math.random() * t.length) | 0], cx, cy, R);
        const a = Math.atan2(y - cy, x - cx) + (Math.random() - 0.5) * 0.6;
        const r0 = R * (1.3 + Math.random() * 0.2), r1 = R * 1.08;
        c.globalAlpha = 0.35 + Math.random() * 0.5;
        c.beginPath();
        c.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
        c.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
        c.stroke();
      }
      c.fillStyle = '#7b5cff';
      c.font = `900 italic ${Math.round(R * 0.13)}px ${getComputedStyle(document.body).fontFamily}`;
      c.textAlign = 'center';
      for (let k = 0; k < 3; k++) {                    // ORA!
        const [x, y] = pos(t[(Math.random() * t.length) | 0], cx, cy, R);
        c.save();
        c.globalAlpha = 0.5 + Math.random() * 0.5;
        c.translate(x + (Math.random() - 0.5) * 18, y - 8 - Math.random() * 14);
        c.rotate((Math.random() - 0.5) * 0.5);
        c.fillText('ORA', 0, 0);
        c.restore();
      }
      c.restore();
    }

    const effectsActive = (now) => !reduce && (!!rushing || now - stopStart < 900 || now < rippleUntil || now < emeraldUntil || now - lastErase < 600);

    function drawScene(c, w, h, cx, cy, R, dot, lw, withHover) {
      if (!reduce && performance.now() < heavenUntil) {  // Made in Heaven: leave afterimages
        c.save();
        c.globalAlpha = 0.3;
        c.fillStyle = colors.paper;
        c.fillRect(0, 0, w, h);
        c.restore();
      } else {
        c.clearRect(0, 0, w, h);
      }
      c.strokeStyle = colors.rule;
      c.lineWidth = 1;
      c.beginPath(); c.arc(cx, cy, R, 0, TAU); c.stroke();
      const score = stats ? stats.score : 0;
      drawMenacing(c, cx, cy, R, status === 'training' ? 1 - 0.8 * score : 0);
      drawStar(c, cx, cy, R, score, lw);
      drawRipples(c, cx, cy, R);
      drawEmerald(c, cx, cy, R);
      const now = performance.now();
      const heavy = now < echoesUntil ? 1.5 : 1;  // 3 FREEZE: everything gets heavier
      for (let i = 0; i < N; i++) {
        let size = dot * heavy;
        if (!model.active[i]) {                    // The Hand: erased points shrink away
          const t = (now - (erasedAt[model.Y[i]] || -1e9)) / 500;
          if (reduce || t >= 1) continue;
          size *= 1 - t;
        }
        const [x, y] = pos(i, cx, cy, R);
        c.fillStyle = colors.c[model.Y[i]];
        c.globalAlpha = hover && withHover && hover.i !== i ? 0.35 : 0.9;
        c.beginPath(); c.arc(x, y, size, 0, TAU); c.fill();
      }
      c.globalAlpha = 1;
      drawRush(c, cx, cy, R, dot);
      drawShockwave(c, cx, cy, R);
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
      const cosEl = tip.querySelector('.hd-cos');
      if (cosEl) cosEl.textContent = `view₁ ↔ view₂ · cos ${cos.toFixed(2)}`;
    }

    // Heaven's Door: hovering a point opens it like a book — its raw 16-d input, as pages
    const sub = (n) => String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[d]);
    function openBook(i) {
      const xs = Array.from(model.X.subarray(i * D, (i + 1) * D));
      const mx = Math.max(...xs.map(Math.abs)) || 1;
      const col = colors.c[model.Y[i]];
      const bars = xs.map((v, d) => {
        const h = (Math.abs(v) / mx) * 15;
        return `<rect x="${d * 10 + 1}" y="${v >= 0 ? 17 - h : 17}" width="8" height="${Math.max(0.8, h)}" fill="${col}"/>`;
      }).join('');
      tip.innerHTML = `<div class="hd-title">📖 Heaven's Door — x${sub(i)}</div>` +
        `<svg class="hd-bars" viewBox="0 0 160 34" width="160" height="34" aria-hidden="true">` +
        `<line x1="0" y1="17" x2="160" y2="17" stroke="currentColor" stroke-opacity="0.25"/>${bars}</svg>` +
        `<div class="hd-meta">16-d input · concept hidden from the model</div><div class="hd-cos"></div>`;
    }

    hero.addEventListener('pointermove', (e) => {
      const r = hero.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const { cx, cy, R } = heroGeom(r.width, r.height);
      let bi = -1, bd = 14 * 14;
      for (let i = 0; i < N; i++) {
        if (!model.active[i]) continue;
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
        openBook(bi);
      }
      tip.hidden = false;
      tip.style.left = Math.max(4, Math.min(mx + 12, r.width - 214)) + 'px';
      tip.style.top = my + 14 + 'px';
      dirty = true;
      if (reduce) render();
    });
    hero.addEventListener('pointerleave', () => {
      hover = null; tip.hidden = true; dirty = true;
      if (reduce) render();
    });

    const heroGeom = (w, h) => ({ cx: w / 2, cy: h / 2, R: Math.min(w, h) * 0.34 });

    function drawMetrics() {
      $('m-step').textContent = model.step.toLocaleString();
      $('m-loss').textContent = lossEma == null ? '—' : lossEma.toFixed(3);
      $('m-align').textContent = alignEma.toFixed(3);
      $('m-unif').textContent = unifEma.toFixed(3);
      for (const el of document.querySelectorAll('[data-tau]')) el.textContent = model.tau.toFixed(2);
      for (const el of document.querySelectorAll('[data-step]')) el.textContent = model.step.toLocaleString();
      for (const el of document.querySelectorAll('[data-loss]')) el.textContent = lossEma == null ? '—' : lossEma.toFixed(2);
    }

    function render() {
      stats = model.stats(S);
      if (heroVisible) {
        const { ctx, w, h } = fit(hero);
        const g = heroGeom(w, h);
        drawScene(ctx, w, h, g.cx, g.cy, g.R, 2.3, 1.4, true);
      }
      drawMetrics();
      dirty = false;
    }

    function tick() {
      raf = requestAnimationFrame(tick);
      frame++;
      const now = performance.now();
      const frozen = now < frozenUntil || !!rewinding || !!rushing;  // ZA WARUDO, a rewind, or Star Platinum's rush (it trains itself)
      const heaven = now < heavenUntil;                // Made in Heaven: time accelerates
      let moved = 0;
      if (!frozen) {
        for (let k = heaven ? 8 : 1; k > 0 && status === 'training'; k--) trainStep();
        if (status !== 'training' && now - convergedAt > 20000) reset();
        moved = smooth(heaven ? 0.35 : 0.12);
      }
      if (moved > 1e-3 || (!frozen && status === 'training') || dirty || heaven || effectsActive(now)) render();
    }

    function trainToConvergence() {
      reset();
      while (status === 'training') { stats = model.stats(E); trainStep(); }
      S.set(E);
      render();
    }

    // ---- controls ----
    const reinit = () => (reduce ? trainToConvergence() : reset());
    function setTau(v) {
      model.tau = Math.min(1, Math.max(0.05, v));
      log(`τ ← ${model.tau.toFixed(2)}`, 'warn');
      if (reduce) { trainToConvergence(); return; }
      if (status === 'converged') { best = Infinity; stall = 0; setStatus('training'); }
      drawMetrics();
    }
    for (const b of document.querySelectorAll('[data-reinit]')) b.addEventListener('click', reinit);
    for (const b of document.querySelectorAll('[data-tau-step]')) {
      b.addEventListener('click', () => setTau(model.tau + parseFloat(b.dataset.tauStep)));
    }
    // ---- time Stands (typed into the prompt; see script.js) ----
    const busy = () => {
      const now = performance.now();
      return !!rewinding || !!rushing || now < frozenUntil || now < echoesUntil || now < emeraldUntil;
    };
    // remember the run right before a Stand acts, so Crazy Diamond can restore it
    function checkpoint(name) {
      cp = { name, st: model.saveState(), E: Float64Array.from(E), lossEma, best, stall, status, P0, trail: trail.slice() };
    }
    const resume = () => { if (status !== 'training') { best = Infinity; stall = 0; setStatus('training'); } };
    const flash = (cls, ms) => {
      if (!wrap || reduce) return;
      wrap.classList.add(cls);
      setTimeout(() => wrap.classList.remove(cls), ms);
    };

    // King Crimson: the next ~300 steps happen, but the time in between is erased
    function kingCrimson(n = 300) {
      if (busy()) return false;
      checkpoint('King Crimson');
      if (status !== 'training') reset();
      const a = model.step;
      for (let k = 0; k < n && status === 'training'; k++) { stats = model.stats(E); trainStep(); }
      S.set(E);
      log(`⏭ King Crimson — ${model.step - a} steps erased (step ${a} → ${model.step})`, 'warn');
      flash('kingcrimson', 600);
      render();
      return true;
    }

    // Made in Heaven: time accelerates x8 for a few seconds
    function madeInHeaven(ms = 6000) {
      if (busy()) return false;
      checkpoint('Made in Heaven');
      if (reduce) { trainToConvergence(); log('⏩ Made in Heaven — time accelerated to the end', 'warn'); return true; }
      if (status !== 'training') reset();
      heavenUntil = performance.now() + ms;
      log('⏩ Made in Heaven — time accelerates ×8', 'warn');
      const until = heavenUntil;
      setTimeout(() => { if (heavenUntil === until) { log('▶ time flows normally', 'warn'); dirty = true; } }, ms);
      return true;
    }

    // Bites the Dust: play the trajectory backwards to step 0, then restart from the same initial weights
    function bitesTheDust(ms = 2500) {
      if (busy() || !P0) return false;
      checkpoint('Bites the Dust');
      const back = () => {
        model.setParams(P0);
        lossEma = null; best = Infinity; stall = 0; heavenUntil = 0; rippleUntil = 0;
        if (tbc) tbc.classList.remove('show');
        model.embed(E);
        trail = [Float32Array.from(E)];
        setStatus('training');
        log('↺ back to step 0 — the same run begins again', 'warn');
        dirty = true;
      };
      log('⏪ Bites the Dust — rewinding…', 'warn');
      if (reduce) { back(); trainToConvergenceFrom(); return true; }
      const frames = trail.slice().reverse();
      const t0 = performance.now();
      wrap && wrap.classList.add('bitesdust');
      rewinding = {
        timer: setInterval(() => {
          const t = Math.min(1, (performance.now() - t0) / ms);
          E.set(frames[Math.min(frames.length - 1, Math.floor(t * frames.length))]);
          smooth(0.4);
          render();
          if (t >= 1) {
            clearInterval(rewinding.timer);
            rewinding = null;
            if (wrap) wrap.classList.remove('bitesdust');
            back();
          }
        }, 16),
      };
      return true;
    }

    // The Hand: erase one hidden concept from space; the rest re-spread to fill the gap
    function theHand() {
      if (busy()) return 'busy';
      const left = model.concepts();
      if (left.length <= 5) return 'limit';
      checkpoint('The Hand');
      const k = left[(Math.random() * left.length) | 0];
      model.erase(k);
      erasedAt[k] = lastErase = performance.now();
      if (hover && model.Y[hover.i] === k) { hover = null; tip.hidden = true; }
      log(`✋ ZA HANDO! — a concept was erased from space (${left.length - 1} left)`, 'warn');
      resume();
      if (reduce) trainToConvergenceFrom();
      dirty = true;
      return left.length - 1;
    }

    // Crazy Diamond: restore the run to how it was right before the last Stand
    function crazyDiamond() {
      if (!cp) return false;
      const c = cp;
      cp = null;
      cancelEffects();
      model.loadState(c.st);
      E.set(c.E);
      lossEma = c.lossEma; best = c.best; stall = c.stall; P0 = c.P0; trail = c.trail;
      for (const k in erasedAt) if (model.concepts().includes(+k)) delete erasedAt[k];
      status = c.status;                                // restore quietly, without re-triggering ZA WARUDO
      document.documentElement.dataset.run = status;
      for (const el of document.querySelectorAll('[data-status]')) el.textContent = status;
      if (tbc) tbc.classList.toggle('show', status === 'converged');
      if (status === 'converged') convergedAt = performance.now();
      log(`💎 DORARARA — restored to step ${model.step} (before ${c.name})`, 'ok');
      if (wrap && !reduce) { wrap.classList.add('crazydiamond'); setTimeout(() => wrap.classList.remove('crazydiamond'), 700); }
      dirty = true;
      render();
      return c.name;
    }

    // a timed tweak to the training setup that undoes itself (Echoes Act 3, Emerald Splash)
    function timedTweak(ms, cls, apply, undo) {
      const token = ++fxToken;
      apply();
      resume();
      if (reduce) { trainToConvergenceFrom(); undo(); return; }
      if (wrap) wrap.classList.add(cls);
      setTimeout(() => {
        if (token !== fxToken) return;
        undo();
        if (wrap) wrap.classList.remove(cls);
        dirty = true;
      }, ms);
    }

    // Echoes Act 3, "3 FREEZE": τ drops, everything gets heavier, clusters pull into tight knots
    function echoes(ms = 8000) {
      if (busy()) return false;
      checkpoint('Echoes Act 3');
      echoesTau = model.tau;
      timedTweak(ms, 'echoes', () => {
        model.tau = 0.08;
        echoesUntil = performance.now() + ms;
        log(`⬇ 3 FREEZE! — τ ${echoesTau.toFixed(2)} → 0.08`, 'warn');
      }, () => {
        model.tau = echoesTau;
        echoesUntil = 0;
        log(`τ back to ${echoesTau.toFixed(2)}`, 'warn');
        drawMetrics();
      });
      drawMetrics();
      return true;
    }

    // Hierophant Green, "Emerald Splash": a burst of much harsher augmentations
    function emeraldSplash(ms = 5000) {
      if (busy()) return false;
      checkpoint('Emerald Splash');
      timedTweak(ms, 'emerald', () => {
        model.aug = { jitter: 0.9, mask: 0.5 };
        emeraldUntil = performance.now() + ms;
        log('💚 EMERALD SPLASH! — augmentations ×3', 'warn');
      }, () => {
        model.aug = { ...AUG };
        emeraldUntil = 0;
        log('splash over — recovering', 'warn');
      });
      return true;
    }

    // Star Platinum: find the points the model handles worst (their two augmented views disagree most)
    // and rush them with a barrage of training steps drawn only from those points: hard-example mining
    function starPlatinum(steps = 200, k = 48) {
      if (busy()) return false;
      checkpoint('Star Platinum');
      const all = [];
      for (let i = 0; i < N; i++) if (model.active[i]) all.push(i);
      const { per } = model.agreeOn(model.views(all));
      const targets = all.map((i, n) => [per[n], i]).sort((a, b) => a[0] - b[0]).slice(0, k).map((x) => x[1]);
      const probe = model.views(targets);            // fixed views: before/after are measured the same way
      const before = model.agreeOn(probe).mean;
      resume();
      log(`👊 ORA ORA ORA — ${steps} steps on the ${k} hardest points`, 'warn');
      const finish = () => {
        model.focus = null; rushTargets = null; rushing = null;
        if (wrap) wrap.classList.remove('starplatinum');
        log(`⭐ ORA! — their agreement ${before.toFixed(3)} → ${model.agreeOn(probe).mean.toFixed(3)}`, 'ok');
        dirty = true;
      };
      model.focus = targets;
      if (reduce) {
        for (let n = 0; n < steps && status === 'training'; n++) trainStep();
        finish();
        S.set(E); render();
        return true;
      }
      rushTargets = new Set(targets);
      if (wrap) wrap.classList.add('starplatinum');
      let done = 0;
      rushing = setInterval(() => {
        for (let n = 0; n < 3 && done < steps; n++, done++) { stats = model.stats(E); trainStep(); }
        smooth(0.3);
        render();
        if (done >= steps || status !== 'training') { clearInterval(rushing); finish(); }
      }, 16);
      return true;
    }

    // reduced motion: finish a restored run without animating
    function trainToConvergenceFrom() {
      while (status === 'training') { stats = model.stats(E); trainStep(); }
      S.set(E);
      render();
    }

    // for the prompt in script.js
    window.SSL.api = {
      reinit, setTau,
      timeStop: (ms) => (busy() ? false : (checkpoint('ZA WARUDO'), timeStop(ms), true)),
      freeze: (ms) => timeStop(ms),
      starPlatinum, kingCrimson, madeInHeaven, bitesTheDust, theHand, crazyDiamond, echoes, emeraldSplash,
      state: () => ({ step: model.step, loss: lossEma, tau: model.tau, status }),
    };

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
    remember();
    if (reduce) trainToConvergence();
    else start();
  }
})();
