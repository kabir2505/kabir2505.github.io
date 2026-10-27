document.querySelectorAll('a').forEach(link => {
  if (!/^(https?:|mailto:)/.test(link.getAttribute('href'))) return; // internal links stay in this tab
  link.setAttribute('target', '_blank');
  link.setAttribute('rel', 'noopener noreferrer');
});

// theme toggle: light by default, dark only when chosen (remembered)
const themeBtn = document.getElementById('theme-toggle');
const toggleTheme = () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
  document.dispatchEvent(new Event('themechange'));
};
if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

// tmux status-bar clock
const clock = document.getElementById('clock');
if (clock) {
  const tick = () => { clock.textContent = new Date().toTimeString().slice(0, 5); };
  tick();
  setInterval(tick, 30000);
}

// news: tag future items "upcoming" and ongoing ones "now", so the list never needs manual edits
const today = new Date();
today.setHours(0, 0, 0, 0);
document.querySelectorAll('.news li[data-date]').forEach(li => {
  const start = new Date(li.dataset.date + 'T00:00');
  const end = new Date((li.dataset.end || li.dataset.date) + 'T00:00');
  const label = start > today ? 'upcoming' : end >= today && li.dataset.end ? 'now' : '';
  if (!label) return;
  const tag = document.createElement('span');
  tag.className = 'tag ' + label;
  tag.textContent = label;
  li.querySelector('.what').append(' ', tag);
});

// terminal tabs: highlight the section currently under the title bar
const tabLinks = [...document.querySelectorAll('.tabs a')];
const tabSecs = tabLinks.map(a => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
if (tabSecs.length) {
  const update = () => {
    const line = document.querySelector('.titlebar').offsetHeight + 40;
    let current = tabSecs[0];
    for (const sec of tabSecs) if (sec.getBoundingClientRect().top <= line) current = sec;
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) current = tabSecs[tabSecs.length - 1];
    tabLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current.id));
  };
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
}

// ---- the prompt: a tiny shell over the page ----
const form = document.getElementById('repl');
if (form) {
  const input = document.getElementById('repl-in');
  const out = document.getElementById('repl-out');
  const api = () => (window.SSL && window.SSL.api) || null;
  const history = [];
  let hIdx = 0;

  // files shown by `ls`, and the section each one lives in
  const FILES = {
    'about.md': 'about', 'research_interests.yaml': 'interests', 'career.log': 'career',
    'news.log': 'news', 'publications.bib': 'publications', 'projects/': 'projects', 'papers_i_love.txt': 'papers',
    'trip/': 'misc', 'misc.yaml': 'misc', 'contact': 'contact',
  };
  const findFile = name => {
    const n = (name || '').replace(/^\.\//, '');
    return Object.keys(FILES).find(f => f === n || f.replace(/\/$/, '') === n.replace(/\/$/, '') || f.split('.')[0] === n);
  };

  // build output from strings and [text, href] link pairs
  const line = (...parts) => {
    const div = document.createElement('div');
    for (const p of parts) {
      if (Array.isArray(p)) {
        const a = document.createElement('a');
        a.textContent = p[0];
        a.href = p[1];
        div.append(a);
      } else {
        div.append(p);
      }
    }
    return div;
  };
  const block = (text, cls) => {
    const pre = document.createElement('div');
    pre.className = 'pre' + (cls ? ' ' + cls : '');
    pre.textContent = text;
    return pre;
  };

  const flash = id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    el.classList.remove('flash');
    void el.offsetWidth; // restart the animation
    el.classList.add('flash');
  };

  const SKILLS = ['torch', 'numpy', 'transformers', 'faiss', 'open_clip', 'scikit-learn', 'xgboost', 'pandas', 'fastapi'];

  const COMMANDS = {
    help: () => block([
      'available commands:',
      '  ls                 list files',
      '  cat <file>         jump to a file (try: cat career.log)',
      '  whoami             who is this',
      '  train [--tau x]    restart the model in pane 1, optionally with a new temperature',
      '  reinit             re-initialize the weights',
      '  trip               open the IJCAI-ECAI \'26 photos',
      '  theme              toggle light / dark',
      '  history, clear     the usual',
      '…and a few things you might try in a real terminal (or in a JoJo fight).',
    ].join('\n')),
    ls: () => {
      const div = document.createElement('div');
      div.className = 'files';
      for (const [f, id] of Object.entries(FILES)) {
        const a = document.createElement('a');
        a.textContent = f;
        a.href = '#' + id;
        if (f.endsWith('/')) a.className = 'dir';
        div.append(a);
      }
      return div;
    },
    cat: args => {
      if (!args[0]) return block('cat: missing file operand (try: cat about.md)', 'err');
      const f = findFile(args[0]);
      if (!f) return block(`cat: ${args[0]}: No such file or directory`, 'err');
      flash(FILES[f]);
      return block(`→ ${f}`, 'dim');
    },
    whoami: () => block('kabir — Research Intern @ Microsoft Research · Data Science @ IIT Madras'),
    pwd: () => block('/home/kabir/research'),
    date: () => block(new Date().toString()),
    train: args => {
      const a = api();
      if (!a) return block('python: train_simclr.py not found', 'err');
      const i = args.indexOf('--tau');
      if (i >= 0) {
        const tau = parseFloat(args[i + 1]);
        if (!(tau > 0)) return block('train: --tau expects a number, e.g. train --tau 0.1', 'err');
        a.setTau(tau);
      }
      a.reinit();
      flash('train');
      return block(`launching python train_simclr.py --tau ${a.state().tau.toFixed(2)} … watch pane 1:train`, 'ok');
    },
    reinit: () => {
      const a = api();
      if (!a) return block('nothing to re-initialize', 'err');
      a.reinit();
      return block('↻ weights re-initialized', 'ok');
    },
    trip: () => { setTimeout(() => { location.href = 'trip.html'; }, 400); return block('opening trip/ijcai26 …', 'ok'); },
    open: args => (findFile(args[0] || '') === 'trip/' ? COMMANDS.trip() : COMMANDS.cat(args)),
    cd: args => {
      if (findFile(args[0] || '') === 'trip/') return COMMANDS.trip();
      return block(args[0] && args[0] !== '~' && args[0] !== '..' ? `cd: no such file or directory: ${args[0]}` : 'already home', args[0] ? 'err' : 'dim');
    },
    theme: () => { toggleTheme(); return block(`theme → ${document.documentElement.dataset.theme}`, 'dim'); },
    history: () => block(history.map((h, i) => `${String(i + 1).padStart(4)}  ${h}`).join('\n') || ''),
    echo: (args, raw) => {
      const text = raw.replace(/^echo\s*/, '');
      if (text.trim() === '$EMAIL') return line(['kabir25.dl@gmail.com', 'mailto:kabir25.dl@gmail.com']);
      return block(text);
    },
    // easter eggs
    'nvidia-smi': () => {
      const st = api() ? api().state() : null;
      return block([
        new Date().toString().slice(0, 24),
        '+-----------------------------------------------------------------------+',
        '| NVIDIA-SMI  ---       Driver Version: none       CUDA Version: none   |',
        '+-----------------------------------------------------------------------+',
        'No devices were found',
        '',
        st ? `→ training on your CPU anyway (step ${st.step.toLocaleString()}${st.loss != null ? `, InfoNCE ${st.loss.toFixed(2)}` : ''})`
           : '→ training on your CPU anyway',
      ].join('\n'));
    },
    sudo: (args, raw) => /rm\s+-rf/.test(raw)
      ? block('nice try 🙂', 'ok')
      : block('kabir is not in the sudoers file. This incident will be reported.', 'err'),
    rm: () => block('rm: permission denied (the model is still training)', 'err'),
    vim: () => block('how do I exit this? (:q!)', 'dim'),
    htop: () => {
      const st = api() ? api().state() : { step: 0 };
      return block([
        '  PID USER      CPU%  MEM%  COMMAND',
        ` 4242 kabir      3.1   0.4  python train_simclr.py --tau ${api() ? st.tau.toFixed(2) : '0.20'}   # step ${st.step.toLocaleString()}`,
        ' 1337 kabir      0.2   0.1  zsh',
        '    1 root       0.0   0.0  coffee',
      ].join('\n'));
    },
    'pip': args => (args[0] === 'list'
      ? block(['Package          Version', '---------------- -------', ...SKILLS.map(s => s.padEnd(17) + 'latest')].join('\n'))
      : block('usage: pip list', 'dim')),
    python: () => block('Python 3.12 — the interesting script is already running in pane 1:train', 'dim'),
    exit: () => block('there is no escape; the model is still training.', 'dim'),
    // JoJo
    ora: () => block('ORA ORA ORA ORA ORA ORA ORA ORA!', 'ok'),
    muda: () => block('MUDA MUDA MUDA MUDA MUDA MUDA!', 'err'),
    yare: () => block('yare yare daze…', 'dim'),
    za: (args, raw) => (/^za\s+warudo$/i.test(raw) ? COMMANDS['the'](['world'], 'the world') : block('zsh: command not found: za', 'err')),
    the: args => {
      if ((args[0] || '').toLowerCase() !== 'world') return block(`zsh: command not found: the`, 'err');
      const a = api();
      if (a) a.freeze(5000);
      return block('ZA WARUDO! Time has stopped. (training paused for 5 seconds)', 'warn');
    },
  };
  COMMANDS.vi = COMMANDS.nano = COMMANDS.vim;
  COMMANDS.top = COMMANDS.htop;
  COMMANDS.python3 = COMMANDS.python;
  COMMANDS.clear = () => { out.textContent = ''; return null; };

  const run = raw => {
    const text = raw.trim();
    if (!text) return;
    history.push(text);
    hIdx = history.length;
    const [cmd, ...args] = text.split(/\s+/);
    const fn = COMMANDS[cmd.toLowerCase()];
    const result = fn ? fn(args, text) : block(`zsh: command not found: ${cmd}`, 'err');
    if (cmd === 'clear') return;
    const entry = document.createElement('div');
    entry.className = 'cmd typed';
    const ps1 = document.createElement('div');
    ps1.className = 'ps1';
    ps1.innerHTML = '<span class="arrow">❯</span> <span class="c"></span>';
    ps1.lastChild.textContent = text;
    entry.append(ps1);
    if (result) {
      result.classList.add('out');
      entry.append(result);
    }
    out.append(entry);
  };

  form.addEventListener('submit', e => {
    e.preventDefault();
    run(input.value);
    input.value = '';
    input.scrollIntoView({ block: 'nearest' });
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'ArrowUp' && history.length) {
      e.preventDefault();
      hIdx = Math.max(0, hIdx - 1);
      input.value = history[hIdx];
    } else if (e.key === 'ArrowDown' && history.length) {
      e.preventDefault();
      hIdx = Math.min(history.length, hIdx + 1);
      input.value = history[hIdx] || '';
    } else if (e.key === 'Tab') {
      // complete a command, or a file name after it
      e.preventDefault();
      const parts = input.value.split(' ');
      const last = parts[parts.length - 1];
      const pool = parts.length === 1 ? Object.keys(COMMANDS) : Object.keys(FILES);
      const hits = pool.filter(x => x.startsWith(last));
      if (hits.length === 1) {
        parts[parts.length - 1] = hits[0];
        input.value = parts.join(' ');
      }
    }
  });

  // clicking the prompt line focuses the input
  form.addEventListener('click', () => input.focus());
}
