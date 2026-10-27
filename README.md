# kabirj_site

My personal website: **https://kabir2505.github.io**

It's styled as a zsh terminal (light and dark). Next to the content, a small neural network **trains live in your browser**: a 16→32→2 MLP learning a contrastive (SimCLR / InfoNCE) loss on unlabeled points, with backprop written by hand in plain JavaScript. No GPU, no server.

It never sees the colors; they're hidden concepts shown only to you. When it converges, time stops, the clusters form a star, Hamon ripples pulse, and "To Be Continued" rolls in.

## Time Stands 🌟

Type a Stand's name into the **Try the terminal** box at the bottom of the page. Each one really does something to the training run:

| Stand | What it does to the model | What you see |
|---|---|---|
| `za warudo` / `the world` | Training stops for 5s (also fires automatically when the model converges) | Inverted colors and a shockwave |
| `king crimson` | ~300 training steps happen instantly; the time in between is erased | A crimson flash, and the points jump to where they'd be. The log shows `⏭ King Crimson — 300 steps erased` |
| `made in heaven` | Training runs 8× faster for 6 seconds | Motion trails as points streak into place, then `▶ time flows normally` |
| `bites the dust` / `killer queen` | The run plays backwards to step 0, then restarts from its exact original weights | A pink Killer Queen tint while rewinding, then `↺ back to step 0 — the same run begins again` |

- Commands aren't case-sensitive, and hyphens work (`KING-CRIMSON`).
- While time is already being bent, other Stands reply "time is already being manipulated…".
- `↻ reinit` cancels everything.
- Visitors with reduced motion get the same results without the animation.

Other things to try: `help`, `ls`, `cat career.log`, `nvidia-smi`, `ora`, `muda`, `yare`, `sudo rm -rf /`.

## Files

- `index.html`, `styles.css`, `script.js`: the page, terminal theme, prompt and commands
- `ssl.js`: the contrastive model (training, rendering, JoJo effects, time Stands)
- `trip.html`, `gallery.js`, `static/ijcai/`: the IJCAI-ECAI '26 trip gallery
- `tools/build_gallery.py`: compresses trip photos/videos into web-sized assets

Static site, no build step. Run it locally with `python3 -m http.server` and open `localhost:8000`.
