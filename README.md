# kabirj_site

My personal website: **https://kabir2505.github.io**

It's styled as a zsh terminal (light and dark). Next to the content, a small neural network **trains live in your browser**: a 16→32→2 MLP learning a contrastive (SimCLR / InfoNCE) loss on unlabeled points, with backprop written by hand in plain JavaScript. No GPU, no server.

It never sees the colors; they're hidden concepts shown only to you. When it converges, time stops, the clusters form a star, Hamon ripples pulse, and "To Be Continued" rolls in.

## Time Stands 🌟

Type a Stand's name into the **Try the terminal** box at the bottom of the page. Each one really does something to the training run:

Not sure what a Stand does? Ask Speedwagon: `speedwagon` lists every Stand, and `speedwagon king crimson` (or any Stand or alias) prints its Stand card: user, Part, what it does in the series, and what it does to the model here.

| Stand | What it does to the model | What you see |
|---|---|---|
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

Other things to try: `help`, `ls`, `cat career.log`, `nvidia-smi`, `ora`, `muda`, `yare`, `sudo rm -rf /`.

## Files

- `index.html`, `styles.css`, `script.js`: the page, terminal theme, prompt and commands
- `ssl.js`: the contrastive model (training, rendering, JoJo effects, time Stands)
- `trip.html`, `gallery.js`, `static/ijcai/`: the IJCAI-ECAI '26 trip gallery
- `tools/build_gallery.py`: compresses trip photos/videos into web-sized assets

Static site, no build step. Run it locally with `python3 -m http.server` and open `localhost:8000`.
