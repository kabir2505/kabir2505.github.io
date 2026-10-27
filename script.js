document.querySelectorAll('a').forEach(link => {
  if (!/^(https?:|mailto:)/.test(link.getAttribute('href'))) return; // internal links stay in this tab
  link.setAttribute('target', '_blank');
  link.setAttribute('rel', 'noopener noreferrer');
});

// theme toggle: light by default, dark only when chosen (remembered)
const themeBtn = document.getElementById('theme-toggle');
if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
    document.dispatchEvent(new Event('themechange'));
  });
}

// index: highlight the section currently in view
const tocLinks = [...document.querySelectorAll('.toc-item')];
if (tocLinks.length && 'IntersectionObserver' in window) {
  const byId = new Map(tocLinks.map(a => [a.getAttribute('href').slice(1), a]));
  const visible = new Set();
  const spy = new IntersectionObserver(entries => {
    entries.forEach(e => (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id)));
    const first = [...byId.keys()].find(id => visible.has(id));
    if (first) tocLinks.forEach(a => a.classList.toggle('active', a === byId.get(first)));
  }, { rootMargin: '-40px 0px -55% 0px' });
  byId.forEach((_, id) => { const s = document.getElementById(id); if (s) spy.observe(s); });
}
