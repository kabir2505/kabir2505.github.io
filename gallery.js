// IJCAI trip gallery (trip.html). Renders from static/ijcai/gallery.json (built by
// tools/build_gallery.py). The first row of thumbnails loads first; the highlight clip
// (a light 480p preview) only starts downloading once those are in, and the full clip
// with sound loads only on demand. Other videos load nothing until played.
(function () {
  'use strict';
  const root = document.getElementById('gallery');
  if (!root) return;

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const saveData = !!(navigator.connection && navigator.connection.saveData);
  let items = [], shown = [], current = -1, filter = 'all';
  const EAGER = 6;                                 // thumbnails fetched first, at high priority

  const el = (tag, cls, attrs) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };

  // the gallery is the whole trip page, so fetch the manifest right away (trip.html preloads it)
  load();

  async function load() {
    try {
      items = await (await fetch('static/ijcai/gallery.json')).json();
    } catch (e) {
      root.querySelector('.g-status').textContent = 'could not load photos';
      return;
    }
    root.textContent = '';
    const feat = items.find((i) => i.featured);
    const fig = feat ? feature(feat) : null;
    if (fig) root.append(fig);
    root.append(filters(), el('div', 'g-grid'));
    renderGrid();
    if (fig) startWhenReady(fig.startPreview);
  }

  // wait for the first row of thumbnails (or ~2.5s) so the clip doesn't compete with them for bandwidth
  function startWhenReady(start) {
    const first = [...root.querySelectorAll('.g-thumb img')].slice(0, EAGER);
    const loaded = first.map((img) => (img.complete ? Promise.resolve() : new Promise((r) => { img.onload = img.onerror = r; })));
    Promise.race([Promise.all(loaded), new Promise((r) => setTimeout(r, 2500))]).then(start);
  }

  // ---- highlight ----
  function feature(item) {
    const fig = el('figure', 'g-feature');
    // no src yet: the poster shows immediately; the light preview is attached once the grid has loaded
    const v = el('video', '', { muted: '', loop: '', playsinline: '', preload: 'none', poster: item.poster });
    v.muted = true;
    let full = false;
    const useFull = () => {                       // swap to the full clip (with sound), keeping the position
      if (full) return;
      full = true;
      const t = v.currentTime;
      v.src = item.src;
      v.addEventListener('loadedmetadata', () => { v.currentTime = t; }, { once: true });
    };
    const badge = el('span', 'g-badge mono');
    badge.textContent = '★ highlight of the trip';
    const sound = el('button', 'g-btn mono', { type: 'button', 'aria-label': 'Toggle sound' });
    sound.textContent = 'sound off';
    sound.addEventListener('click', () => {
      if (v.muted) useFull();                     // the preview is silent
      v.muted = !v.muted;
      sound.textContent = v.muted ? 'sound off' : 'sound on';
      if (v.paused) v.play().catch(() => {});
    });
    const play = el('button', 'g-btn g-play mono', { type: 'button', 'aria-label': 'Play or pause' });
    const sync = () => (play.textContent = v.paused ? '▶ play' : '❚❚ pause');
    play.addEventListener('click', () => {
      if (!v.getAttribute('src')) v.src = item.preview || item.src;
      v.paused ? v.play().catch(() => {}) : v.pause();
    });
    v.addEventListener('play', sync);
    v.addEventListener('pause', sync);
    sync();
    const cap = el('figcaption');
    cap.textContent = item.caption;
    const controls = el('div', 'g-controls');
    controls.append(play, sound);
    fig.append(v, badge, controls, cap);

    // autoplay muted only while on screen (and only if the visitor hasn't asked for less)
    fig.startPreview = () => {
      if (reduce || saveData) return;
      if (!v.getAttribute('src')) v.src = item.preview || item.src;
      new IntersectionObserver((es) => {
        if (es[0].isIntersecting) v.play().catch(() => {});
        else v.pause();
      }, { threshold: 0.5 }).observe(v);
    };
    return fig;
  }

  // ---- filters + grid ----
  function filters() {
    const groups = ['all', ...new Set(items.map((i) => i.group))];
    const bar = el('div', 'g-filters mono', { role: 'tablist' });
    for (const g of groups) {
      const n = g === 'all' ? items.length : items.filter((i) => i.group === g).length;
      const b = el('button', 'g-chip', { type: 'button', role: 'tab', 'aria-selected': String(g === filter) });
      b.textContent = `${g} · ${n}`;
      b.addEventListener('click', () => {
        filter = g;
        for (const c of bar.children) c.setAttribute('aria-selected', String(c === b));
        renderGrid();
      });
      bar.append(b);
    }
    return bar;
  }

  function renderGrid() {
    const grid = root.querySelector('.g-grid');
    shown = items.filter((i) => filter === 'all' || i.group === filter);
    grid.textContent = '';
    shown.forEach((item, idx) => {
      const b = el('button', 'g-thumb' + (item.type === 'video' ? ' is-video' : ''), { type: 'button', 'aria-label': item.caption || 'photo' });
      const img = el('img', '', {
        src: item.thumb, alt: item.caption || '', decoding: 'async', width: item.w, height: item.h,
        ...(idx < EAGER ? { fetchpriority: 'high' } : { loading: 'lazy' }),
      });
      b.append(img);
      if (item.caption) {
        const label = el('span', 'g-label');
        label.textContent = item.caption;
        b.append(label);
      }
      b.addEventListener('click', () => open(idx));
      grid.append(b);
    });
  }

  // ---- lightbox ----
  const box = el('dialog', 'g-lightbox');
  box.innerHTML = '<div class="g-stage"></div><p class="g-cap"></p>' +
    '<button class="g-nav g-prev" type="button" aria-label="Previous">‹</button>' +
    '<button class="g-nav g-next" type="button" aria-label="Next">›</button>' +
    '<button class="g-close" type="button" aria-label="Close">✕</button>';
  document.body.append(box);
  const stage = box.querySelector('.g-stage');
  box.querySelector('.g-prev').addEventListener('click', () => step(-1));
  box.querySelector('.g-next').addEventListener('click', () => step(1));
  box.querySelector('.g-close').addEventListener('click', () => box.close());
  box.addEventListener('click', (e) => { if (e.target === box) box.close(); });
  box.addEventListener('close', () => { stage.textContent = ''; current = -1; });
  document.addEventListener('keydown', (e) => {
    if (!box.open) return;
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
  });

  function open(idx) {
    current = idx;
    const item = shown[idx];
    stage.textContent = '';
    let media;
    if (item.type === 'video') {
      media = el('video', '', { controls: '', playsinline: '', autoplay: '', poster: item.poster, src: item.src });
    } else {
      media = el('img', '', { src: item.thumb, alt: item.caption || '', width: item.w, height: item.h });
      const full = new Image();
      full.onload = () => { if (current === idx) media.src = item.full; };
      full.src = item.full;
    }
    stage.append(media);
    box.querySelector('.g-cap').textContent = `${item.caption || ''}  ·  ${idx + 1} / ${shown.length}`;
    if (!box.open) box.showModal();
  }

  function step(d) {
    if (current < 0) return;
    open((current + d + shown.length) % shown.length);
  }
})();
