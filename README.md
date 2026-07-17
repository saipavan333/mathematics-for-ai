# Mathematics for AI Engineers — In Depth, Plain English

A self-contained, runnable course that teaches the mathematics behind modern machine
learning **from intuition to code**. Every concept opens with a real AI/ML situation,
explains itself in plain English before any symbol, draws the picture before the algebra,
derives every step with no hand-waving, and ships **runnable NumPy code** next to the
formula — all in a single static web app with no build step.

## What's inside

- **13 core tracks (76 lessons + timed exams)** — Arithmetic & Algebra, Linear Algebra
  I–IV, Calculus I–II, Optimization, Probability I–II, Statistics, Information Theory,
  Numerical Methods.
- **Bridge tracks (being added)** — Matrix Calculus & Backprop, Bayesian Inference &
  Probabilistic ML, Diffusion & Generative Models, Neural-Network & Transformer Math.
- **5 capstones that actually run in the browser** — PCA from scratch on MNIST digits;
  Gradient Descent / Momentum / Adam; a Gaussian Bayes classifier; Cross-Entropy / KL /
  Mutual Information on a tiny language model; and a tiny reverse-mode autograd engine that
  trains a 2-layer net.
- Every lesson follows one standard: **AI Moment → Plain English → Intuition (with a
  diagram) → Formalism → Derivation (every step labelled) → Runnable Code → Diagram →
  Key Points → Common Mistakes → Quiz → Practice → Deep Dive.**

## How to use it

- **Online:** visit the deployed site (see **`DEPLOY.md`** to publish your own copy to
  GitHub Pages — free, and works on any device including your phone).
- **Locally, easiest:** double-click `index.html`.
- **Locally, most reliable:** serve the folder so the in-browser Python behaves:
  ```bash
  python -m http.server 8000     # from this folder, then open http://localhost:8000
  ```

Notes:
- **Math** renders with [KaTeX](https://katex.org); **code runs** in the browser with
  [Pyodide](https://pyodide.org). NumPy, Matplotlib, and scikit-learn load from a CDN on
  demand, so you need internet on first run.
- The first time you click **▶ Run**, Python boots (a few seconds); after that it's quick.
  Cells within a lesson share state, so later cells can use earlier variables.
- PyTorch can't run inside a browser, so ideas are shown as NumPy equivalents. All
  in-browser code uses NumPy / Matplotlib / scikit-learn.

## Navigating

- Left sidebar: every track; click a track to expand its lessons.
- **Next / Previous** buttons (or the **← / →** arrow keys) move through the curriculum and
  always land at the **top** of the new lesson.
- Search filters lessons; the top-right toggle switches light/dark; "Mark complete" stores
  progress in your browser.

## Project structure

```
index.html               # app shell — loads everything below
styles.css               # theme
app.js                   # lesson engine (nav, routing, quizzes, in-browser code runner)
curriculum.js            # the track/lesson map (sidebar + prev/next order)
content/
  track01.js … track13.js  # the 13 core tracks
  track14.js …             # bridge tracks (added over time)
  capstones.js             # the 5 runnable projects
DIAGRAM_GOLDEN_STANDARD.md # rules every hand-drawn diagram must satisfy
DEPLOY.md                  # step-by-step publishing guide
```

## Tech

Pure vanilla JavaScript — no framework, no bundler, no server. Everything is static files,
which is why it deploys to GitHub Pages (or any static host) in a couple of minutes.
Adding a track later = dropping in its `content/trackNN.js` and one line in `curriculum.js`.
