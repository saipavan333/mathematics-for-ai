# HANDOFF — Mathematics for AI Engineers (compact context)

**What:** No-build single-page course teaching the math behind AI in plain English. For Pavan's brother. Repo `saipavan333/mathematics-for-ai`; live `https://saipavan333.github.io/mathematics-for-ai/`.

**Stack:** vanilla-JS SPA. `index.html` loads `curriculum.js` → `content/track01–17.js` + `capstones.js` + `nutshells.js` + `diagrams.js` + `widgets.js` → `assistant-data.js` → `app.js` → `features.js`. Styling: `styles.css` + `features.css`. KaTeX (math) + Pyodide (runnable Python) from CDN. State in `localStorage` under `mathai:`.

**Content:** 17 tracks + Capstones = 88 lessons + 18 exams. Each lesson: nutshell, prose, worked code, diagram, interactive widget, quiz, practice, deep-dive. Scan-verified: 0 math/schema/link/quiz defects.

**Features:** home roadmap, spaced-repetition `#review`, search, glossary `#glossary` (+inline tooltips), cheat sheets `#cheatsheets`, interview bank `#interview` (410 Qs), exam `#exam`, concept map `#map`, read-aloud, light/dark, WCAG-AA, print, SEO/OG, PWA offline. Feature routes wired via `window.MHUB` hooks in `app.js`.

**AI assistant:** retrieval + Cloudflare Worker (Gemini free tier) `gd-assistant-proxy.uekpavanharish.workers.dev`; Worker source at `../AI Engineer - Gradient Descent/worker/assistant-proxy.js`. `ALLOWED_ORIGINS` locked to the Pages origin.

**Deploy (Windows only):** `git rm -r ai-math` (once) → `git add -A && git commit -m "…" && git push`.

**Gotchas:** this Linux mount CANNOT push git or delete `ai-math/` ("Operation not permitted") — Pavan pushes from Windows. Files can torn-read in bash → trust `node --check`/Read. Never type `\n` in an Edit payload (lands literal). Gemini Flash "thinking" tokens share `maxOutputTokens` (keep high).

**Not built:** subscriptions/access-control, explainer videos, separate labs.

**Deep detail:** agent memory `maths-for-ai-course.md`.
