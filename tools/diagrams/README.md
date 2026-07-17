# Diagram generator (tracks 1–13)

Every lesson diagram in tracks 1–13 is computed from the real math (per
`../../DIAGRAM_GOLDEN_STANDARD.md`) and emitted as inline SVG.

- `dg.py` — framework: `Canvas`, `Frame` (data→pixel), palette, `caption()`.
- `t01.py … t13.py` — one function per lesson, registered with `@reg(track)`.
- `gen.py` — writes `../../content/diagrams.js` (assigns each SVG to
  `LESSON_CONTENT[id].diagram`).
- `render.py <tracks…>` — per-track PNG contact sheets for review.
- `montage.py` — all-63 grid for a final geometric sweep.

Regenerate:  `python3 gen.py`  (needs numpy; PNG review needs cairosvg + pillow).
Diagrams render on a fixed light plate via the `.section-body svg` rule in
`styles.css`, so their dark ink stays readable in light AND dark themes.
