// Blog: posts are Markdown files in blog/posts/<slug>.md, indexed by blog/posts.json.
// blog.html lists them (#blog-list); post.html?slug=… renders one (#post).
// To add a post: drop a .md file in blog/posts/ and add an entry to blog/posts.json.
(function () {
  'use strict';

  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtDate = (iso) => new Date(iso + 'T00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

  // ---- tiny Markdown renderer (headings, paragraphs, lists, quotes, rules, inline emphasis/links/code) ----
  function inline(text) {
    const keep = [];                                // links and code are set aside so emphasis can't touch their insides
    const stash = (html) => `\u0000${keep.push(html) - 1}\u0000`;
    let s = esc(text);
    s = s.replace(/`([^`]+)`/g, (_, c) => stash(`<code>${c}</code>`));
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => stash(`<a href="${href}">${emph(label)}</a>`));
    s = emph(s);
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => keep[+i]);
  }
  function emph(s) {
    return s
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^\w])_([^_]+)_(?=[^\w]|$)/g, '$1<em>$2</em>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  }

  function markdown(md) {
    const lines = String(md).replace(/\r/g, '').split('\n');
    const out = [];
    let i = 0;
    const isBlank = (l) => !l.trim();
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) { i++; continue; }
      let m;
      if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
        // the posts use ####/#####, so they become h2/h3 on the page
        const level = Math.min(6, Math.max(2, m[1].length - 2));
        out.push(`<h${level}>${inline(m[2].trim())}</h${level}>`);
        i++;
      } else if (/^(-{3,}|\*{3,})\s*$/.test(line)) {
        out.push('<hr>');
        i++;
      } else if (/^>\s?/.test(line)) {
        const buf = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
        out.push(`<blockquote>${inline(buf.join(' '))}</blockquote>`);
      } else if (/^\s*(\d+)[.)]\s+/.test(line) || /^\s*[-*]\s+/.test(line)) {
        const ordered = /^\s*\d+[.)]\s+/.test(line);
        const marker = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*]\s+/;
        const items = [];
        while (i < lines.length && (marker.test(lines[i]) || (isBlank(lines[i]) && marker.test(lines[i + 1] || '')))) {
          if (!isBlank(lines[i])) items.push(lines[i].replace(marker, ''));
          i++;
        }
        const tag = ordered ? 'ol' : 'ul';
        out.push(`<${tag}>${items.map((t) => `<li>${inline(t)}</li>`).join('')}</${tag}>`);
      } else {
        const buf = [];
        while (i < lines.length && !isBlank(lines[i]) && !/^(#{1,6}\s|>|\s*\d+[.)]\s|\s*[-*]\s)/.test(lines[i])) buf.push(lines[i++].trim());
        out.push(`<p>${inline(buf.join(' '))}</p>`);
      }
    }
    return out.join('\n');
  }

  const minutes = (md) => Math.max(1, Math.ceil(String(md).split(/\s+/).filter(Boolean).length / 220));

  async function getIndex() {
    const res = await fetch('blog/posts.json');
    if (!res.ok) throw new Error('index');
    return res.json();
  }

  // ---- blog.html ----
  async function renderList(root) {
    try {
      const posts = await getIndex();
      root.innerHTML = posts.map((p) => `
        <article class="post-row">
          <div class="post-date">${esc(fmtDate(p.date))}</div>
          <div>
            <a class="post-title" href="post.html?slug=${encodeURIComponent(p.slug)}">${esc(p.title)}</a>
            <p class="post-dek">${esc(p.dek || '')}</p>
            <div class="post-tags">${(p.tags || []).map((t) => `<span class="chip">${esc(t)}</span>`).join('')}</div>
          </div>
        </article>`).join('') || '<p class="dim">no posts yet.</p>';
    } catch (e) {
      root.innerHTML = '<p class="err">ls: cannot open blog/: try reloading</p>';
    }
  }

  // ---- post.html ----
  async function renderPost(root) {
    const slug = new URLSearchParams(location.search).get('slug') || '';
    const set = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
    set('post-cmd', `<span class="arrow">❯</span> cat blog/${esc(slug || '?')}.md`);
    const missing = () => {
      set('post-title', 'Not found');
      root.innerHTML = `<p class="err">cat: blog/${esc(slug)}.md: No such file or directory</p><p><a href="blog.html">← all posts</a></p>`;
    };
    try {
      const posts = await getIndex();
      const at = posts.findIndex((p) => p.slug === slug);
      if (at < 0) return missing();
      const post = posts[at];
      const res = await fetch(`blog/posts/${encodeURIComponent(slug)}.md`);
      if (!res.ok) return missing();
      const md = await res.text();
      document.title = `${post.title} · Kabir J`;
      set('post-meta', `<span class="post-date">${esc(fmtDate(post.date))}</span>${post.place ? ` · ${esc(post.place)}` : ''} · ${minutes(md)} min read`);
      set('post-title', esc(post.title));
      set('post-dek', esc(post.dek || ''));
      root.innerHTML = markdown(md);
      root.querySelectorAll('a[href^="http"]').forEach((a) => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });
      const newer = posts[at - 1], older = posts[at + 1];
      set('post-nav', [
        '<a href="blog.html">← all posts</a>',
        older ? `<a href="post.html?slug=${encodeURIComponent(older.slug)}">older: ${esc(older.title)}</a>` : '',
        newer ? `<a href="post.html?slug=${encodeURIComponent(newer.slug)}">newer: ${esc(newer.title)}</a>` : '',
      ].filter(Boolean).join(''));
      const eof = document.getElementById('post-eof');
      if (eof) eof.hidden = false;
    } catch (e) {
      missing();
    }
  }

  // reading progress bar
  function progress(bar) {
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
    };
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  if (typeof module !== 'undefined' && module.exports) { module.exports = { markdown }; return; }

  const list = document.getElementById('blog-list');
  if (list) renderList(list);
  const post = document.getElementById('post');
  if (post) renderPost(post);
  const bar = document.getElementById('progress');
  if (bar) progress(bar);
})();
