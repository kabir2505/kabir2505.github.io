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
    'about.md': 'about', 'research_interests.yaml': 'about', 'career.log': 'career',
    'news.log': 'news', 'publications.bib': 'publications', 'projects/': 'projects', 'papers_i_love.txt': 'papers',
    'trip/': 'misc', 'misc.yaml': 'misc',
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
      '  speedwagon [stand] Speedwagon explains what each Stand does',
      '  history, clear     the usual',
      'stands:  star platinum · za warudo · king crimson · made in heaven · bites the dust',
      '         the hand · crazy diamond · echoes act 3 · emerald splash',
      '         (and hover a point in the training pane: Heaven\'s Door)',
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
    muda: () => block('MUDA MUDA MUDA MUDA MUDA MUDA!', 'err'),
    yare: () => block('yare yare daze…', 'dim'),
  };

  // time Stands: each one really does something to the training run in pane 1
  const BUSY = 'time is already being manipulated… wait for it to resume.';
  const STANDS = {
    'za warudo': a => (a.timeStop(5000) ? block('ZA WARUDO! Time has stopped. (training paused for 5 seconds)', 'warn') : block(BUSY, 'dim')),
    'king crimson': a => (a.kingCrimson() ? block('KING CRIMSON! The time in between has been erased.', 'warn') : block(BUSY, 'dim')),
    'made in heaven': a => (a.madeInHeaven() ? block('MADE IN HEAVEN! Time accelerates…', 'warn') : block(BUSY, 'dim')),
    'bites the dust': a => (a.bitesTheDust() ? block('KILLER QUEEN — BITES THE DUST! Rewinding to step 0…', 'warn') : block(BUSY, 'dim')),
  };
  STANDS['the hand'] = a => {
    const r = a.theHand();
    if (r === 'busy') return block(BUSY, 'dim');
    if (r === 'limit') return block('ZA HANDO… only 5 concepts left — erasing more would leave nothing to learn.', 'dim');
    return block(`ZA HANDO! A whole concept has been erased from space. ${r} remain — watch the rest re-spread.`, 'warn');
  };
  STANDS['crazy diamond'] = a => {
    const r = a.crazyDiamond();
    return r ? block(`DORARARARA! Restored to how it was before ${r}.`, 'ok') : block('Crazy Diamond: nothing to fix (try another Stand first).', 'dim');
  };
  STANDS['echoes act 3'] = a => (a.echoes() ? block('3 FREEZE! Everything feels heavier… τ drops to 0.08 for 8 seconds.', 'warn') : block(BUSY, 'dim'));
  STANDS['emerald splash'] = a => (a.emeraldSplash() ? block('EMERALD SPLASH! The augmentations hit three times harder for 5 seconds.', 'warn') : block(BUSY, 'dim'));
  STANDS['star platinum'] = a => (a.starPlatinum() ? block('STAR PLATINUM! ORA ORA ORA ORA ORA ORA ORA ORA ORA!', 'warn') : block(BUSY, 'dim'));
  STANDS['ora'] = STANDS['ora ora'] = STANDS['star finger'] = STANDS['star platinum'];
  STANDS['the world'] = STANDS['za warudo'];
  STANDS['killer queen'] = STANDS['bites the dust'];
  STANDS['za hando'] = STANDS['the hand'];
  STANDS['dora'] = STANDS['crazy diamond'];
  STANDS['3 freeze'] = STANDS['echoes'] = STANDS['echoes act 3'];
  STANDS['hierophant green'] = STANDS['emerald splash'];

  // Speedwagon explains: a Stand card for each Stand on this page
  const CARDS = [
    { key: 'star platinum', name: 'Star Platinum', user: 'Jotaro Kujo', part: 'Part 3 · Stardust Crusaders',
      series: 'Superhuman speed and pinpoint precision; finishes fights with its ORA ORA rush.',
      here: 'Finds the 48 points the model handles worst (their two augmented views disagree most) and rushes them with 200 extra training steps, then reports how much their agreement improved: hard-example mining.',
      summon: ['star platinum', 'ora ora', 'ora'] },
    { key: 'za warudo', name: 'The World', user: 'DIO', part: 'Part 3 · Stardust Crusaders',
      series: 'Stops time for everyone but its user.',
      here: 'Pauses training for 5 seconds. It also fires on its own when the model converges: colors invert, a shockwave sweeps out, then the star, Hamon ripples and "To Be Continued".',
      summon: ['za warudo', 'the world'] },
    { key: 'king crimson', name: 'King Crimson', user: 'Diavolo', part: 'Part 5 · Golden Wind',
      series: 'Erases a span of time; only the user experiences what happens in it.',
      here: 'About 300 training steps happen instantly. The points jump straight to where they would have ended up.',
      summon: ['king crimson'] },
    { key: 'made in heaven', name: 'Made in Heaven', user: 'Enrico Pucci', part: 'Part 6 · Stone Ocean',
      series: 'Accelerates time itself.',
      here: 'Training runs 8× faster for 6 seconds, leaving motion trails behind the points.',
      summon: ['made in heaven'] },
    { key: 'bites the dust', name: 'Killer Queen: Bites the Dust', user: 'Yoshikage Kira', part: 'Part 4 · Diamond is Unbreakable',
      series: 'Rewinds time to an earlier moment.',
      here: 'Plays the run backwards to step 0, then restarts it from its exact original weights.',
      summon: ['bites the dust', 'killer queen'] },
    { key: 'the hand', name: 'The Hand', user: 'Okuyasu Nijimura', part: 'Part 4 · Diamond is Unbreakable',
      series: 'Its right hand erases whatever it swipes through, space included.',
      here: 'Erases one hidden concept from the data. The remaining clusters re-spread to fill the gap: contrastive uniformity, live. Works down to 5 concepts.',
      summon: ['the hand', 'za hando'] },
    { key: 'crazy diamond', name: 'Crazy Diamond', user: 'Josuke Higashikata', part: 'Part 4 · Diamond is Unbreakable',
      series: 'Restores anything it touches to a previous state.',
      here: 'Restores the run to exactly how it was before the last Stand: weights, optimizer state, step, τ and erased concepts.',
      summon: ['crazy diamond', 'dora'] },
    { key: 'echoes act 3', name: 'Echoes Act 3', user: 'Koichi Hirose', part: 'Part 4 · Diamond is Unbreakable',
      series: '"3 FREEZE": makes whatever it hits extremely heavy.',
      here: 'Drops the temperature τ to 0.08 for 8 seconds, so clusters pull into tight, heavy knots. Then τ returns.',
      summon: ['echoes act 3', '3 freeze', 'echoes'] },
    { key: 'emerald splash', name: 'Hierophant Green', user: 'Noriaki Kakyoin', part: 'Part 3 · Stardust Crusaders',
      series: '"Emerald Splash": fires a barrage of emerald energy.',
      here: 'Augmentations hit 3× harder for 5 seconds. The loss spikes in the log, then recovers: a live robustness test.',
      summon: ['emerald splash', 'hierophant green'] },
    { key: "heaven's door", name: "Heaven's Door", user: 'Rohan Kishibe', part: 'Part 4 · Diamond is Unbreakable',
      series: 'Opens people up like books so their memories can be read.',
      here: 'Hover any point in the training pane: it opens like a book, showing its raw 16-dimensional input, the one the model sees with no label.',
      summon: ['(hover a point)'] },
  ];
  const norm = t => t.toLowerCase().replace(/[’']/g, "'").replace(/[-_\s]+/g, ' ').trim();
  const findCard = t => {
    const q = norm(t);
    return CARDS.find(c => norm(c.name) === q || c.summon.some(a => norm(a) === q) || norm(c.key) === q || (q === 'heavens door' && c.key === "heaven's door"));
  };
  const standCard = c => {
    const card = document.createElement('div');
    card.className = 'stand-card';
    const row = (label, text) => {
      const r = document.createElement('div');
      r.className = 'sc-row';
      const l = document.createElement('span');
      l.className = 'sc-label';
      l.textContent = label;
      const v = document.createElement('span');
      v.textContent = text;
      r.append(l, v);
      return r;
    };
    const head = document.createElement('div');
    head.className = 'sc-head';
    head.innerHTML = '<span class="sc-menace" aria-hidden="true">ゴゴゴ</span>';
    const title = document.createElement('div');
    title.className = 'sc-name';
    title.textContent = `「${c.name}」`;
    head.prepend(title);
    card.append(head,
      row('STAND USER', `${c.user} — ${c.part}`),
      row('IN THE SERIES', c.series),
      row('ON THIS PAGE', c.here),
      row('SUMMON', c.summon.join('  ·  ')));
    return card;
  };
  COMMANDS.speedwagon = args => {
    if (args.length) {
      const c = findCard(args.join(' '));
      return c ? standCard(c) : block(`Speedwagon: "I've never heard of a Stand called '${args.join(' ')}'!" (try: speedwagon)`, 'err');
    }
    return block([
      'SPEEDWAGON IS EXPLAINING! Here are the Stands on this page:',
      '',
      ...CARDS.map(c => `  ${c.name.padEnd(30)} ${c.summon[0]}`),
      '',
      'For a Stand card: speedwagon <stand>   e.g. speedwagon king crimson',
    ].join('\n'));
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
    const stand = STANDS[text.toLowerCase().replace(/[-_\s]+/g, ' ')];
    const fn = stand ? () => (api() ? stand(api()) : block('the Stand has no one to act on', 'err')) : COMMANDS[cmd.toLowerCase()];
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
