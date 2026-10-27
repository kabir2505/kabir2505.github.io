# kabirj_site

My personal website: **https://kabir2505.github.io**

It's styled as a zsh terminal (light and dark). Next to the content, a small neural network **trains live in your browser**: a 16→32→2 MLP learning a contrastive (SimCLR / InfoNCE) loss on unlabeled points, with backprop written by hand in plain JavaScript. No GPU, no server.

It never sees the colors; they're hidden concepts shown only to you. When it converges, time stops, the clusters form a star, Hamon ripples pulse, and "To Be Continued" rolls in.

## How the model works (SimCLR in your browser)

**Where it runs:** on each visitor's own CPU, inside their browser tab. GitHub Pages only serves static files; `ssl.js` downloads, and the tab trains its own copy of the model. The model is tiny (610 parameters, about 1 ms per step), so one step per screen refresh is plenty. It pauses when the tab is hidden and stops computing once it converges.

### 1. Data
- 8 hidden "concepts": prototype vectors $\mu_1,\dots,\mu_8 \in \mathbb{R}^{16}$, each $\sim \mathcal{N}(0, I)$.
- 256 samples, 32 per concept: $x = \mu_y + 0.45\,\varepsilon$, with $\varepsilon \sim \mathcal{N}(0, I)$.
- The label $y$ is **never shown to the model**. It only colors the dots for you.

### 2. Augmentations (two views per sample)
- Scale the whole vector by $s \sim U(0.8, 1.2)$.
- Then, per dimension: with probability 0.25 mask it to 0, otherwise use $s\,x_d + 0.3\,\mathcal{N}(0,1)$.

### 3. Encoder
A 16 → 32 → 2 MLP (610 parameters), normalized onto the unit circle $S^1$:

$$h = \tanh(W_1 x + b_1),\qquad z = W_2 h + b_2,\qquad u = \frac{z}{\lVert z\rVert}$$

### 4. Objective: NT-Xent / InfoNCE
Each step draws $B = 64$ samples and makes 2 views of each ($2B = 128$ embeddings). Views $2k$ and $2k{+}1$ are a positive pair; every other view in the batch is a negative. With $\tau = 0.2$:

$$\ell_i = -\log \frac{\exp\big(u_i^\top u_{p(i)}/\tau\big)}{\sum_{j \ne i} \exp\big(u_i^\top u_j/\tau\big)},\qquad \mathcal{L} = \frac{1}{2B}\sum_{i=1}^{2B} \ell_i$$

It's a 127-way "which one is my partner?" classification. The numerator pulls the two views together (**alignment**); the denominator pushes everything else apart (**uniformity**).

### 5. Training
- Backprop is written by hand in JavaScript, through the softmax, the normalization ($\partial u/\partial z = (I - uu^\top)/\lVert z\rVert$), $W_2$, $\tanh$ and $W_1$. Gradients were checked against finite differences (relative error about $10^{-6}$).
- Adam, learning rate $3\times10^{-3}$, $\beta = (0.9, 0.999)$, one step per animation frame.

### 6. Metrics in the training card
- **InfoNCE:** $\mathcal{L}$ above, smoothed.
- **align:** $\mathbb{E}\,\lVert u_a - u_b\rVert^2$ over positive pairs. Lower means the two views agree more.
- **uniform:** $\log \mathbb{E}\, e^{-2\lVert u_i - u_j\rVert^2}$ over different samples. More negative means the points are spread more evenly. (Both from Wang & Isola, 2020.)

### 7. Why a star, and why the loss stalls near 3.3
- Alignment pulls each concept's samples into a tight cluster; uniformity spreads the 8 clusters evenly around the circle. Connecting their centers gives the {8/3} star.
- In a batch of 128 views, each concept has about 16 views that look nearly identical, yet the loss treats them as negatives. These **false negatives** (class collision) cap how confident the model can be, so the loss bottoms out near $\log 16 \approx 2.8$ plus a bit, about 3.3 in practice.
- It counts as converged after step 1,200 once the loss stops improving (less than 0.01 over 600 steps) and the clusters are tight and evenly spaced, or at step 5,000.

### How it differs from real SimCLR
| Real SimCLR | Here |
|---|---|
| ResNet on images | 610-parameter MLP on synthetic 16-D points |
| Projection head; loss on $g(h)$, representation is $h$ | No separate head; the 2-D output is used directly so it can be plotted |
| 128-D sphere, batches of thousands, crops and color jitter | Unit circle, batch 64, noise / masking / scaling |

The core objective, NT-Xent with in-batch negatives and a temperature, is the same, just small enough to train live in a browser tab.

## Time Stands 🌟

Type a Stand's name into the **Try the terminal** box at the bottom of the page. Each one really does something to the training run:

Not sure what a Stand does? Ask Speedwagon: `speedwagon` lists every Stand, and `speedwagon king crimson` (or any Stand or alias) prints its Stand card: user, Part, what it does in the series, and what it does to the model here.

| Stand | What it does to the model | What you see |
|---|---|---|
| `star platinum` / `ora ora` / `ora` | Finds the 48 points the model handles worst (their two augmented views disagree most) and rushes them with 200 extra training steps: hard-example mining. Reports how much their agreement improved | Purple rings on the targets, punches flying in, ORA bursts. `⭐ ORA! — their agreement 0.938 → 0.958` |
| `za warudo` / `the world` | Training stops for 5s (also fires automatically when the model converges) | Inverted colors and a shockwave |
| `king crimson` | ~300 training steps happen instantly; the time in between is erased | A crimson flash, and the points jump to where they'd be. The log shows `⏭ King Crimson — 300 steps erased` |
| `made in heaven` | Training runs 8× faster for 6 seconds | Motion trails as points streak into place, then `▶ time flows normally` |
| `bites the dust` / `killer queen` | The run plays backwards to step 0, then restarts from its exact original weights | A pink Killer Queen tint while rewinding, then `↺ back to step 0 — the same run begins again` |
| `the hand` / `za hando` | Erases one hidden concept from the data; the remaining clusters re-spread to fill the gap (contrastive *uniformity*, live). Works down to 5 concepts | Erased points shrink away and the star redraws with fewer points. `✋ ZA HANDO! — a concept was erased from space` |
| `crazy diamond` / `dora` | Restores the run to exactly how it was before the last Stand: weights, optimizer state, step, τ, erased concepts | A brief diamond tint. `💎 DORARARA — restored to step N (before King Crimson)` |
| `echoes act 3` / `3 freeze` | Temperature τ drops to 0.08 for 8 seconds, so the clusters pull into tight, heavy knots | A green tint, with points drawn heavier. `⬇ 3 FREEZE! — τ 0.20 → 0.08` |
| `emerald splash` / `hierophant green` | Augmentations hit 3× harder for 5 seconds; the loss spikes, then recovers (a live robustness demo) | An emerald tint with green shards around the ring. `💚 EMERALD SPLASH! — augmentations ×3` |

**Heaven's Door:** hover any point in the training pane and it opens like a book, showing its raw 16-dimensional input (the one the model sees, with no label) and how close its two augmented views land.

- Commands aren't case-sensitive, and hyphens work (`KING-CRIMSON`).
- While a Stand is already active, others reply "time is already being manipulated…". Crazy Diamond can always step in to fix things.
- `↻ reinit` cancels everything.
- Visitors with reduced motion get the same results without the animation.

Other things to try: `help`, `ls`, `cat career.log`, `nvidia-smi`, `muda`, `yare`, `sudo rm -rf /`.

## Files

- `index.html`, `styles.css`, `script.js`: the page, terminal theme, prompt and commands
- `ssl.js`: the contrastive model (training, rendering, JoJo effects, time Stands)
- `trip.html`, `gallery.js`, `static/ijcai/`: the IJCAI-ECAI '26 trip gallery
- `tools/build_gallery.py`: compresses trip photos/videos into web-sized assets

Static site, no build step. Run it locally with `python3 -m http.server` and open `localhost:8000`.
