/* ============================================================
   Math for AI Engineers — lesson engine
   Pure vanilla JS. Loads last; reads window.CURRICULUM (map)
   and window.LESSON_CONTENT (authored lessons).
   ============================================================ */
(function () {
  "use strict";

  const CURRICULUM = window.CURRICULUM || [];
  const CONTENT    = window.LESSON_CONTENT || {};
  const LS = {
    get(k, d) { try { return JSON.parse(localStorage.getItem("mathai:" + k)) ?? d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("mathai:" + k, JSON.stringify(v)); } catch (e) {} }
  };

  /* ---- flat ordered list of lessons across all tracks ---- */
  const FLAT = [];
  CURRICULUM.forEach(tr => (tr.lessons || []).forEach(ls =>
    FLAT.push({ id: ls.id, title: ls.title, track: tr, lesson: ls })));
  const indexOf = id => FLAT.findIndex(x => x.id === id);

  /* ============================================================
     SECTION ORDER + LABELS (the fixed lesson standard)
     ============================================================ */
  const ORDER = [
    ["aiMoment",      "AI Moment",       "s-ai",   true ],
    ["plainEnglish",  "Plain English",   "s-plain",false],
    ["intuition",     "Intuition",       "s-intu", false],
    ["formalism",     "Formalism",       "s-form", false],
    ["derivation",    "Derivation",      "s-deriv",false],
    ["__code",        "Code",            "s-code", false],
    ["diagram",       "Diagram",         "s-intu", false],
    ["__widget",      "Try It",          "s-try",  false],
    ["__keyPoints",   "Key Points",      "s-key",  true ],
    ["__mistakes",    "Common Mistakes", "s-miss", true ],
    ["__quiz",        "Quiz",            "s-quiz", false],
    ["__practice",    "Practice",        "s-prac", false],
    ["deepDive",      "Deep Dive",       "s-deep", true ]
  ];

  /* ============================================================
     NAV (sidebar)
     ============================================================ */
  const navEl = document.getElementById("nav");

  function buildNav() {
    let collapsed = LS.get("collapsed", null);
    if (collapsed === null) collapsed = CURRICULUM.map(t => t.id).filter(id => id !== "1");
    const collapsedSet = new Set(collapsed);
    navEl.innerHTML = "";
    const done = new Set(LS.get("done", []));

    CURRICULUM.forEach(tr => {
      const wrap = document.createElement("div");
      wrap.className = "track" + (collapsedSet.has(tr.id) ? " collapsed" : "");
      wrap.dataset.track = tr.id;
      wrap.style.setProperty("--th", "var(--h" + tr.id + ")");

      const head = document.createElement("div");
      head.className = "track-head";
      head.innerHTML =
        `<span class="track-num">${tr.id}</span>` +
        `<span class="track-title">${tr.short || tr.title}</span>` +
        `<span class="chev">▾</span>`;
      head.addEventListener("click", () => {
        wrap.classList.toggle("collapsed");
        const c = [...document.querySelectorAll(".track.collapsed")].map(e => e.dataset.track);
        LS.set("collapsed", c);
      });
      wrap.appendChild(head);

      const ul = document.createElement("ul");
      ul.className = "lessons";
      (tr.lessons || []).forEach(ls => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#" + ls.id;
        a.dataset.id = ls.id;
        a.className = (done.has(ls.id) ? "done " : "") + (isMastered(ls.id) ? "mastered " : "") + (CONTENT[ls.id] ? "" : "stub");
        a.innerHTML =
          `<span class="dot"></span>` +
          `<span class="ll-id">${ls.id}</span>` +
          `<span class="ll-title">${ls.title}</span>`;
        li.appendChild(a);
        ul.appendChild(li);
      });
      wrap.appendChild(ul);
      navEl.appendChild(wrap);
    });
  }

  function highlightNav(id) {
    document.querySelectorAll(".lessons a").forEach(a => a.classList.toggle("active", a.dataset.id === id));
    const trk = FLAT[indexOf(id)] && FLAT[indexOf(id)].track.id;
    const w = navEl.querySelector(`.track[data-track="${trk}"]`);
    if (w && w.classList.contains("collapsed")) w.classList.remove("collapsed");
  }

  /* ============================================================
     RENDER a lesson
     ============================================================ */
  const contentEl = document.getElementById("content");
  const crumbsEl  = document.getElementById("crumbs");
  const navFootEl = document.getElementById("lessonNav");

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function sectionNode(cls, label, builder) {
    const callout = ["s-ai", "s-key", "s-miss", "s-deep"].includes(cls);
    const sec = el("div", "section " + cls + (callout ? " callout" : ""));
    sec.appendChild(el("div", "section-label", label));
    builder(sec);
    return sec;
  }

  function renderLesson(id) {
    const entry = FLAT[indexOf(id)];
    contentEl.innerHTML = "";
    if (!entry) { contentEl.appendChild(el("p", null, "Lesson not found.")); return; }

    const c = CONTENT[id];
    contentEl.dataset.track = entry.track.id;
    contentEl.style.setProperty("--th", "var(--h" + entry.track.id + ")");

    /* header */
    const head = el("div", "lesson-h");
    head.innerHTML = `<div class="eyebrow">Track ${entry.track.id} · ${entry.track.short || entry.track.title}</div>
                      <h2>${id} &nbsp;${entry.title}</h2>`;
    contentEl.appendChild(head);
    if (c && c.subtitle) contentEl.appendChild(el("div", "lesson-sub", c.subtitle));
    const nut = (window.NUTSHELLS || {})[id];
    if (nut && c && !c.exam) contentEl.appendChild(el("div", "nutshell", `<span class="nutshell-tag">In one breath</span>${nut}`));

    if (!c) {
      renderPlaceholder(entry);
    } else if (c.exam) {
      renderExam(c);
      renderCompleteRow(id);
    } else {
      ORDER.forEach(([key, label, cls]) => {
        if (key === "__code")      return renderCode(c.code, cls, label);
        if (key === "__keyPoints") return renderKeyPoints(c.keyPoints, cls, label);
        if (key === "__mistakes")  return renderMistakes(c.commonMistakes, cls, label);
        if (key === "__quiz")      return renderQuiz(c.quiz, cls, label, id);
        if (key === "__practice")  return renderPractice(c.practice, cls, label);
        if (key === "__widget")    return renderWidget(id, cls, label);
        if (c[key]) contentEl.appendChild(
          sectionNode(cls, label, s => s.appendChild(el("div", "section-body", c[key]))));
      });
      renderCompleteRow(id);
    }

    /* math + meta */
    typesetMath(contentEl, 0, function () { crossLink(contentEl); });
    crumbsEl.innerHTML = `<b>${entry.track.title}</b> &nbsp;›&nbsp; ${entry.title}`;
    document.title = id + " " + entry.title + " — Math for AI Engineers";
    buildLessonNav(id);
    highlightNav(id);
    LS.set("last", id);
    const seen = new Set(LS.get("seen", [])); seen.add(id); LS.set("seen", [...seen]);
    touchStreak();
    scrollTop();
  }

  /* ---- interactive widget ---- */
  function renderWidget(id, cls, label) {
    const build = (window.WIDGETS || {})[id];
    if (!build) return;
    contentEl.appendChild(sectionNode(cls, label, s => {
      const body = el("div", "section-body");
      const mount = el("div", "widget-mount");
      body.appendChild(mount);
      s.appendChild(body);
      try { build(mount); } catch (e) { mount.textContent = "(interactive demo failed to load)"; }
    }));
  }

  /* ---- CODE cells (runnable) ---- */
  function renderCode(cells, cls, label) {
    if (!cells || !cells.length) return;
    const sec = sectionNode(cls, label, () => {});
    cells.forEach((cell, i) => sec.appendChild(buildCodeCell(cell, i)));
    contentEl.appendChild(sec);
  }

  function buildCodeCell(cell, i) {
    const box = el("div", "codecell ui");
    const head = el("div", "cc-head");
    head.innerHTML = `<span class="cc-dot r"></span><span class="cc-dot y"></span><span class="cc-dot g"></span>
                      <span class="cc-label">${cell.label || "Python"}</span>`;
    const run = el("button", "cc-run", "▶ Run");
    head.appendChild(run);
    const pre = el("pre"); const code = el("code");
    code.textContent = (cell.src || "").replace(/^\n+/, "").replace(/\s+$/, "");
    pre.appendChild(code); box.append(head, pre);
    const out = el("div", "cc-out");
    out.innerHTML = `<div class="cc-out-label">Output</div>`;
    const outPre = el("pre"); out.appendChild(outPre);
    box.appendChild(out);
    run.addEventListener("click", () => runCell(code.textContent, run, out, outPre));
    return box;
  }

  /* ---- KEY POINTS ---- */
  function renderKeyPoints(points, cls, label) {
    if (!points || !points.length) return;
    contentEl.appendChild(sectionNode(cls, label, s => {
      const ul = el("ul", "keypoints");
      points.forEach(p => ul.appendChild(el("li", null, p)));
      s.appendChild(ul);
    }));
  }

  /* ---- COMMON MISTAKES ---- */
  function renderMistakes(items, cls, label) {
    if (!items || !items.length) return;
    contentEl.appendChild(sectionNode(cls, label, s => {
      items.forEach(m => {
        const d = el("div", "mistake");
        d.appendChild(el("div", "wrong", m.wrong));
        d.appendChild(el("div", "why", m.why));
        s.appendChild(d);
      });
    }));
  }

  /* ---- QUIZ ---- */
  function renderQuiz(qs, cls, label, lessonId) {
    if (!qs || !qs.length) return;
    contentEl.appendChild(sectionNode(cls, label, s => {
      qs.forEach((q, qi) => {
        const box = el("div", "quiz-q");
        box.appendChild(el("div", "qtext", `${qi + 1}. ${q.q}`));
        const exp = el("div", "quiz-exp");
        // shuffle display order so the correct option isn't always first
        const order = q.options.map((_, i) => i);
        for (let k = order.length - 1; k > 0; k--) {
          const j = Math.floor(Math.random() * (k + 1));
          [order[k], order[j]] = [order[j], order[k]];
        }
        order.forEach((origIdx) => {
          const b = el("button", "quiz-opt", q.options[origIdx]);
          b.addEventListener("click", () => {
            const btns = box.querySelectorAll(".quiz-opt");
            btns.forEach((btn, bi) => {
              btn.classList.remove("correct", "wrong");
              if (order[bi] === q.answer) btn.classList.add("correct");
            });
            if (origIdx !== q.answer) b.classList.add("wrong");
            exp.innerHTML = (origIdx === q.answer ? "<b>Correct.</b> " : "<b>Not quite.</b> ") + q.explain;
            exp.classList.add("show"); typesetMath(exp);
            if (lessonId) { recordQuiz(lessonId, qi, origIdx === q.answer); refreshStats(); }
          });
          box.appendChild(b);
        });
        box.appendChild(exp);
        s.appendChild(box);
      });
    }));
  }

  /* ---- PRACTICE ---- */
  function renderPractice(items, cls, label) {
    if (!items || !items.length) return;
    contentEl.appendChild(sectionNode(cls, label, s => {
      const wrap = el("div", "practice");
      items.forEach((p, i) => {
        const det = el("details");
        const lvl = p.level || "med";
        const lname = { easy: "Easy", med: "Medium", hard: "Hard" }[lvl] || "Medium";
        det.appendChild(el("summary", null,
          `Exercise ${i + 1}<span class="lvl ${lvl}">${lname}</span><div class="pr-prompt">${p.prompt}</div>`));
        det.appendChild(el("div", "sol", "<b>Solution.</b> " + p.solution));
        wrap.appendChild(det);
      });
      s.appendChild(wrap);
    }));
  }

  /* ---- TRACK EXAM ---- */
  function renderExam(c) {
    if (c.intro) contentEl.appendChild(sectionNode("s-ai", "How to Take This Exam",
      s => s.appendChild(el("div", null, c.intro))));
    contentEl.appendChild(sectionNode("s-prac", "Problems", s => {
      const wrap = el("div", "practice");
      (c.problems || []).forEach((p, i) => {
        const det = el("details");
        const lvl = p.level || "med";
        const lname = { easy: "Easy", med: "Medium", hard: "Hard" }[lvl] || "Medium";
        det.appendChild(el("summary", null,
          `Problem ${i + 1}<span class="lvl ${lvl}">${lname}</span><div class="pr-prompt">${p.prompt}</div>`));
        det.appendChild(el("div", "sol", "<b>Solution.</b> " + p.solution));
        wrap.appendChild(det);
      });
      s.appendChild(wrap);
    }));
    if (c.rubric) contentEl.appendChild(sectionNode("s-deep", "Self-Assessment Rubric",
      s => s.appendChild(el("div", null, c.rubric))));
  }

  /* ---- PLACEHOLDER (not yet authored) ---- */
  function renderPlaceholder(entry) {
    const box = el("div", "pending-box");
    box.innerHTML =
      `<div class="big">This lesson is mapped and on the build queue.</div>
       <p style="color:var(--ink-faint);font-size:14px">Full lesson — written to the same standard as the live ones —
       arrives when this track gets its dedicated build.</p>`;
    if (entry.lesson.covers && entry.lesson.covers.length) {
      const h = el("div", null, `<b style="font-size:13px;color:var(--ink-soft)">This lesson will cover</b>`);
      const ul = el("ul", "syllabus");
      entry.lesson.covers.forEach(t => ul.appendChild(el("li", null, t)));
      box.append(h, ul);
    }
    if (entry.lesson.anchor)
      box.appendChild(el("div", "tag-note", "<b>AI anchor:</b> " + entry.lesson.anchor));
    contentEl.appendChild(box);
  }

  /* ---- complete toggle ---- */
  function renderCompleteRow(id) {
    const done = new Set(LS.get("done", []));
    const row = el("div", "complete-row");
    const b = el("button", "complete-btn ui" + (done.has(id) ? " done" : ""),
      done.has(id) ? "✓ Completed" : "Mark this lesson complete");
    b.addEventListener("click", () => {
      const d = new Set(LS.get("done", []));
      d.has(id) ? d.delete(id) : d.add(id);
      LS.set("done", [...d]); buildNav(); highlightNav(id);
      b.classList.toggle("done"); b.textContent = d.has(id) ? "✓ Completed" : "Mark this lesson complete";
    });
    row.appendChild(b);
    contentEl.appendChild(row);
  }

  /* ---- prev / next ---- */
  function buildLessonNav(id) {
    const i = indexOf(id);
    const prev = FLAT[i - 1], next = FLAT[i + 1];
    navFootEl.innerHTML = "";
    const mk = (item, dir) => {
      const a = el("a", dir + (item ? "" : " disabled"));
      a.href = item ? "#" + item.id : "#";
      a.innerHTML = `<div class="dir">${dir === "prev" ? "‹ Previous" : "Next ›"}</div>
                     <div class="ttl">${item ? item.id + " " + item.title : "—"}</div>`;
      return a;
    };
    navFootEl.append(mk(prev, "prev"), mk(next, "next"));
  }

  /* ============================================================
     MATH
     ============================================================ */
  function typesetMath(root, _tries, done) {
    if (!window.renderMathInElement) {           // KaTeX still loading — retry, don't give up
      if ((_tries || 0) < 60) { setTimeout(() => typesetMath(root, (_tries || 0) + 1, done), 120); return; }
      if (done) { try { done(); } catch (e) {} } // KaTeX never loaded (offline) — still cross-link
      return;
    }
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$",  right: "$",  display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false, ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      });
    } catch (e) {}
    if (done) { try { done(); } catch (e) {} }
  }

  /* ---- cross-reference auto-linking: turn "14.5" / "Track 6" / "C.1" into clickable jumps ---- */
  const XREF_IDS = new Set(FLAT.map(x => x.id));
  const XREF_FIRST = (() => { const m = {}; CURRICULUM.forEach(tr => { const f = (tr.lessons || []).find(l => CONTENT[l.id]); if (f) m[tr.id] = f.id; }); return m; })();
  function crossLink(root) {
    if (!root) return;
    const cur = (location.hash || "").replace(/^#/, "");
    const rx = /(?:\b\d{1,2}\.\d{1,2}\b|\bC\.\d\b|\bTrack \d{1,2}\b)/;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        for (let p = n.parentNode; p && p !== root; p = p.parentNode) {
          const nm = (p.nodeName || "").toUpperCase();
          if (nm === "A" || nm === "CODE" || nm === "PRE" || nm === "BUTTON" || nm === "SVG" || nm === "TEXT" || nm === "TSPAN" || nm === "H2" || nm === "SCRIPT" || nm === "STYLE") return NodeFilter.FILTER_REJECT;
          if (p.classList && (p.classList.contains("katex") || p.classList.contains("codecell"))) return NodeFilter.FILTER_REJECT;
        }
        return rx.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    const nodes = []; let n; while ((n = walker.nextNode())) nodes.push(n);
    nodes.forEach(t => {
      const html = t.nodeValue
        .replace(/\b(\d{1,2}\.\d{1,2}|C\.\d)\b/g, m => (XREF_IDS.has(m) && m !== cur) ? `<a class="xref" href="#${m}">${m}</a>` : m)
        .replace(/\bTrack (\d{1,2})\b/g, (m, g) => (XREF_FIRST[g] && XREF_FIRST[g].split(".")[0] !== cur.split(".")[0]) ? `<a class="xref" href="#${XREF_FIRST[g]}">Track ${g}</a>` : m);
      if (html !== t.nodeValue) { const s = document.createElement("span"); s.innerHTML = html; t.parentNode.replaceChild(s, t); }
    });
  }

  /* ============================================================
     PYODIDE (lazy)
     ============================================================ */
  let pyReady = null, outBuf = [];
  const pyStatus = document.getElementById("py-status");
  const setStatus = (t) => { if (!t) { pyStatus.classList.remove("show"); } else { pyStatus.textContent = t; pyStatus.classList.add("show"); } };

  async function ensurePyodide() {
    if (pyReady) return pyReady;
    pyReady = (async () => {
      setStatus("Booting Python…");
      let tries = 0;
      while (typeof loadPyodide === "undefined" && tries++ < 100) await new Promise(r => setTimeout(r, 100));
      if (typeof loadPyodide === "undefined") throw new Error("Pyodide failed to load (check your internet connection).");
      const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/" });
      setStatus("Installing NumPy…");
      await py.loadPackage(["numpy"]);            // essential; matplotlib loads lazily per cell
      py.setStdout({ batched: s => outBuf.push(s) });
      py.setStderr({ batched: s => outBuf.push(s) });
      setStatus("");
      return py;
    })();
    return pyReady;
  }

  const CAPTURE = `
import io as _io, base64 as _b64, json as _json
__imgs=[]
try:
    import matplotlib.pyplot as _plt
    for _n in _plt.get_fignums():
        _f=_plt.figure(_n); _bio=_io.BytesIO()
        _f.savefig(_bio,format='png',bbox_inches='tight',dpi=110); _bio.seek(0)
        __imgs.append(_b64.b64encode(_bio.read()).decode())
    _plt.close('all')
except Exception:
    pass
_json.dumps(__imgs)`;

  async function runCell(code, btn, out, outPre) {
    btn.disabled = true; const old = btn.textContent; btn.textContent = "Running…";
    out.classList.add("show"); outPre.className = ""; outPre.textContent = "";
    try {
      const py = await ensurePyodide();
      outBuf = [];
      let result;
      try {
        try { setStatus("Loading libraries…"); await py.loadPackagesFromImports(code); } catch (e) {}
        /* set a headless backend only if matplotlib actually loaded — never fatal */
        try { await py.runPythonAsync("try:\n    import matplotlib\n    matplotlib.use('AGG')\nexcept Exception:\n    pass"); } catch (e) {}
        setStatus("");
        result = await py.runPythonAsync(code);
      } catch (err) {
        outPre.className = "err";
        outPre.textContent = (outBuf.join("\n") + "\n" + String(err.message || err)).trim();
        return;
      }
      let text = outBuf.join("\n");
      if (result !== undefined && result !== null && String(result) !== "undefined") {
        const rs = "" + result;
        if (rs.trim() !== "") text += (text ? "\n" : "") + rs;
      }
      outPre.textContent = text;
      // figures
      let imgs = [];
      try { imgs = JSON.parse(await py.runPythonAsync(CAPTURE)); } catch (e) {}
      imgs.forEach(b64 => {
        const im = document.createElement("img");
        im.src = "data:image/png;base64," + b64; out.appendChild(im);
      });
      if (!text && !imgs.length) outPre.textContent = "(ran — no text output)";
    } catch (e) {
      outPre.className = "err"; outPre.textContent = String(e.message || e);
    } finally {
      btn.disabled = false; btn.textContent = old;
    }
  }

  /* ============================================================
     LEARNING SYSTEM — progress · mastery · spaced repetition · streak
     ============================================================ */
  const DAY = 86400000;
  const BOX_DAYS = { 1: 1, 2: 2, 3: 4, 4: 8, 5: 16 };
  const todayStr = () => new Date().toISOString().slice(0, 10);
  const dateOffset = n => new Date(Date.now() + n * DAY).toISOString().slice(0, 10);

  const PHASES = [
    { name: "Foundations", blurb: "The algebra and linear algebra everything else stands on.", tracks: ["1","2","3","4","5"] },
    { name: "Calculus & Optimization", blurb: "Derivatives, gradients, and how models actually learn.", tracks: ["6","7","8"] },
    { name: "Probability & Statistics", blurb: "Reasoning under uncertainty; estimating from data.", tracks: ["9","10","11"] },
    { name: "Information & Computation", blurb: "Entropy, and doing all this numerically and stably.", tracks: ["12","13"] },
    { name: "The Bridge to ML", blurb: "Matrix calculus / backprop and Bayesian inference.", tracks: ["14","15"] },
    { name: "Modern Deep Learning", blurb: "Diffusion, generative models, and the transformer.", tracks: ["16","17"] },
    { name: "Capstone Projects", blurb: "Five things you build from scratch that actually run.", tracks: ["C"] }
  ];

  const trackById = id => CURRICULUM.find(t => t.id === id);
  const realLessons = tr => (tr.lessons || []).filter(l => CONTENT[l.id] && !CONTENT[l.id].exam);
  const doneSet = () => new Set(LS.get("done", []));
  function quizRec() { return LS.get("quiz", {}); }
  function isMastered(id) { const r = quizRec()[id]; return !!(r && r.best >= 0.8); }
  function isSeen(id) { return new Set(LS.get("seen", [])).has(id); }
  function lessonDone(id) { return isMastered(id) || doneSet().has(id); }

  function trackProgress(tr) {
    const ls = realLessons(tr); if (!ls.length) return { done: 0, total: 0, pct: 0 };
    const done = ls.filter(l => lessonDone(l.id)).length;
    return { done, total: ls.length, pct: Math.round(100 * done / ls.length) };
  }
  function overallProgress() {
    let done = 0, total = 0;
    CURRICULUM.forEach(tr => { const p = trackProgress(tr); done += p.done; total += p.total; });
    return { done, total, pct: total ? Math.round(100 * done / total) : 0 };
  }

  /* ---- streak ---- */
  function touchStreak() {
    const s = LS.get("streak", { last: null, days: 0 }); const t = todayStr();
    if (s.last === t) return s;
    s.days = (s.last === dateOffset(-1)) ? (s.days + 1) : 1;
    s.last = t; LS.set("streak", s); return s;
  }
  function streakDays() {
    const s = LS.get("streak", { last: null, days: 0 });
    return (s.last === todayStr() || s.last === dateOffset(-1)) ? s.days : 0;
  }

  /* ---- spaced repetition (Leitner boxes) ---- */
  function srs() { return LS.get("srs", {}); }
  function seedCard(cardId) { const s = srs(); if (s[cardId]) return; s[cardId] = { box: 1, due: Date.now() + DAY, lapses: 0 }; LS.set("srs", s); }
  function gradeCard(cardId, correct) {
    const s = srs(); const c = s[cardId] || { box: 1, due: 0, lapses: 0 };
    if (correct) c.box = Math.min(5, c.box + 1); else { c.box = 1; c.lapses = (c.lapses || 0) + 1; }
    c.due = Date.now() + BOX_DAYS[c.box] * DAY; s[cardId] = c; LS.set("srs", s);
  }
  function dueCards() { const s = srs(), now = Date.now(); return Object.keys(s).filter(id => s[id].due <= now); }
  function cardQuestion(cardId) {
    const m = /^q:(.+):(\d+)$/.exec(cardId); if (!m) return null;
    const c = CONTENT[m[1]]; if (!c || !c.quiz) return null;
    const q = c.quiz[+m[2]]; if (!q) return null;
    return { lessonId: m[1], qi: +m[2], q, title: (FLAT[indexOf(m[1])] || {}).title || m[1] };
  }
  function recordQuiz(lessonId, qi, correct) {
    seedCard("q:" + lessonId + ":" + qi);
    const solved = LS.get("solved", {}); const set = new Set(solved[lessonId] || []);
    correct ? set.add(qi) : set.delete(qi);
    solved[lessonId] = [...set]; LS.set("solved", solved);
    const total = (CONTENT[lessonId].quiz || []).length || 1;
    const qz = quizRec(); const prevBest = (qz[lessonId] && qz[lessonId].best) || 0;
    qz[lessonId] = { best: Math.max(prevBest, set.size / total), attempts: ((qz[lessonId] && qz[lessonId].attempts) || 0) + 1 };
    LS.set("quiz", qz);
  }

  /* ---- header + sidebar chrome (injected) ---- */
  function refreshStats() {
    const due = dueCards().length, prog = overallProgress(), streak = streakDays();
    const badge = document.getElementById("reviewBadge"); if (badge) badge.textContent = due ? due : "";
    const bar = document.getElementById("hdrProgress"); if (bar) bar.style.width = prog.pct + "%";
    const stk = document.getElementById("hdrStreak"); if (stk) stk.textContent = streak ? ("🔥 " + streak) : "";
    document.querySelectorAll(".lessons a").forEach(a => a.classList.toggle("mastered", isMastered(a.dataset.id)));
  }
  function injectChrome() {
    const brand = document.querySelector(".brand");
    if (brand && !document.getElementById("homeLink")) {
      const nav = el("div", "sidenav");
      nav.innerHTML = `<a id="homeLink" href="#home" class="sidenav-btn">&#9672; Home</a>` +
        `<a id="reviewLink" href="#review" class="sidenav-btn">&#8635; Review <span id="reviewBadge" class="rev-badge"></span></a>`;
      brand.parentNode.insertBefore(nav, brand.nextSibling);
    }
    const spacer = document.querySelector(".topbar .spacer");
    if (spacer && !document.getElementById("hdrStatsWrap")) {
      const s = el("div", "hdr-stats"); s.id = "hdrStatsWrap";
      s.innerHTML = `<span id="hdrStreak" class="hdr-streak"></span>` +
        `<span class="hdr-bar" title="overall progress"><span id="hdrProgress" class="hdr-bar-fill"></span></span>`;
      spacer.parentNode.insertBefore(s, spacer);
    }
  }

  /* ---- shared bits ---- */
  function clearMeta() { crumbsEl.innerHTML = ""; navFootEl.innerHTML = ""; document.querySelectorAll(".lessons a.active").forEach(a => a.classList.remove("active")); }
  function statCard(big, html, cls) { return el("div", "statcard " + cls, `<div class="statcard-big">${big}</div><div class="statcard-lbl">${html}</div>`); }
  function actionBtn(text, href, cls) { const a = el("a", "dash-btn " + (cls || ""), text); a.href = href; return a; }

  /* ============================================================
     HOME DASHBOARD
     ============================================================ */
  function renderDashboard() {
    contentEl.innerHTML = ""; contentEl.removeAttribute("data-track"); clearMeta();
    const prog = overallProgress(), due = dueCards().length, streak = streakDays(), last = LS.get("last", null);
    const wrap = el("div", "dash");

    /* hero: editorial headline + a progress ring (the signature moment) */
    const r = 54, C = 2 * Math.PI * r;
    const ringSVG =
      `<svg class="ring" viewBox="0 0 130 130" width="134" height="134" aria-hidden="true">
         <circle cx="65" cy="65" r="${r}" style="fill:none;stroke:var(--ring-track);stroke-width:11"/>
         <circle class="ring-fg" cx="65" cy="65" r="${r}" transform="rotate(-90 65 65)"
                 style="fill:none;stroke:var(--accent);stroke-width:11;stroke-linecap:round;stroke-dasharray:${C.toFixed(1)};stroke-dashoffset:${(C * (1 - prog.pct / 100)).toFixed(1)}"/>
         <text x="65" y="64" text-anchor="middle" style="fill:var(--ink);font-family:Georgia,serif;font-size:37px;font-weight:700">${prog.pct}<tspan style="fill:var(--ink-soft);font-size:17px">%</tspan></text>
         <text x="65" y="86" text-anchor="middle" style="fill:var(--ink-faint);font-family:-apple-system,sans-serif;font-size:11px;letter-spacing:.6px">MASTERED</text>
       </svg>`;
    const hero = el("div", "dash-hero");
    hero.innerHTML =
      `<canvas class="hero-bg" aria-hidden="true"></canvas>
       <div class="hero-txt">
         <div class="hero-eyebrow">${streak > 1 ? `Day ${streak} · welcome back &#128293;` : "Mathematics for AI Engineers"}</div>
         <h2>Master the math behind modern AI — from first principles to transformers.</h2>
         <p class="dash-sub">Plain English first. Run every line of code. Quiz yourself. Then let a few minutes of daily review make it permanent.</p>
       </div>
       <div class="hero-ring">${ringSVG}<div class="hero-ring-meta">${prog.done} / ${prog.total} lessons</div></div>`;
    wrap.appendChild(hero);

    const act = el("div", "dash-actions");
    if (due) act.appendChild(actionBtn(`&#8635;&nbsp; Review ${due} card${due > 1 ? "s" : ""}`, "#review", "primary"));
    if (last && CONTENT[last]) act.appendChild(actionBtn(`Continue &nbsp;·&nbsp; ${last} ${(FLAT[indexOf(last)] || {}).title || ""}`, "#" + last, due ? "" : "primary"));
    else { const f = (realLessons(CURRICULUM[0])[0] || {}).id; if (f) act.appendChild(actionBtn(`Start learning &nbsp;·&nbsp; ${f}`, "#" + f, "primary")); }
    wrap.appendChild(act);

    const stats = el("div", "dash-stats");
    stats.appendChild(statCard(due, `due to review<br><span>${due ? "clear these to lock it in" : "all caught up &#10003;"}</span>`, "s-quiz"));
    stats.appendChild(statCard(streak || 0, `day streak<br><span>${streak ? "keep it alive" : "study today to start"}</span>`, "s-ai"));
    stats.appendChild(statCard(prog.total - prog.done, `lessons to go<br><span>of ${prog.total} total</span>`, "s-key"));
    wrap.appendChild(stats);

    wrap.appendChild(el("div", "dash-how",
      `<div class="dash-how-t">How to learn here</div>
       <ol>
         <li><b>Read</b> the lesson top to bottom — plain English first, then the math, then every step of the derivation.</li>
         <li><b>Run</b> each code cell (press ▸ Run) and read the output — the formula becomes something real.</li>
         <li><b>Take the quiz.</b> It records what you got right and schedules those questions for spaced review.</li>
         <li><b>Try 2 practice problems</b> before opening the solution.</li>
         <li><b>Review daily.</b> Clear your due cards and it locks into long-term memory.</li>
       </ol>`));

    const road = el("div", "roadmap");
    road.appendChild(el("div", "roadmap-t", "Your roadmap"));
    PHASES.forEach((ph, pi) => {
      const trks = ph.tracks.map(trackById).filter(Boolean);
      let d = 0, t = 0; trks.forEach(tr => { const p = trackProgress(tr); d += p.done; t += p.total; });
      const pct = t ? Math.round(100 * d / t) : 0;
      const phEl = el("div", "phase");
      phEl.appendChild(el("div", "phase-h",
        `<span class="phase-n">${pi + 1}</span>
         <div class="phase-txt"><div class="phase-name">${ph.name}</div><div class="phase-blurb">${ph.blurb}</div></div>
         <div class="phase-pct">${pct}%</div>`));
      const grid = el("div", "phase-tracks");
      trks.forEach(tr => {
        const p = trackProgress(tr);
        const first = (realLessons(tr)[0] || tr.lessons[0] || {}).id;
        const card = el("a", "tcard" + (p.total && p.done === p.total ? " complete" : "")); card.href = "#" + first;
        card.dataset.track = tr.id; card.style.setProperty("--th", "var(--h" + tr.id + ")");
        card.innerHTML =
          `<div class="tcard-top"><span class="tcard-id">${tr.id}</span><span class="tcard-name">${tr.short || tr.title}</span></div>
           <div class="tbar"><div class="tbar-fill" style="width:${p.pct}%"></div></div>
           <div class="tcard-meta">${p.done}/${p.total} lessons${p.total && p.done === p.total ? " · done ✓" : ""}</div>`;
        grid.appendChild(card);
      });
      phEl.appendChild(grid); road.appendChild(phEl);
    });
    wrap.appendChild(road);

    contentEl.appendChild(wrap);
    const hb = wrap.querySelector(".hero-bg"); if (hb) heroAurora(hb);
    document.title = "Home — Math for AI Engineers";
    scrollTop();
  }

  /* generative constellation — full-screen click-through backdrop (Wondersmith signature) */
  function injectAppBackdrop() {
    if (document.querySelector(".app-bg")) return;
    const cv = document.createElement("canvas"); cv.className = "app-bg"; cv.setAttribute("aria-hidden", "true");
    document.body.insertBefore(cv, document.body.firstChild);
    const hues = ["#E5484D","#EF6C36","#D98324","#AF7B00","#5E9E12","#2FA84F","#12A594","#0EA5B7","#0C8CE0","#3B6FF0","#5B54F0","#7A4DE0","#9A3FD4","#B833C0","#C42EA6","#D92B7C","#E5486A","#5B6B84"];
    const ctx = cv.getContext("2d"); if (!ctx) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0, H = 0, pts = [];
    function size() {
      W = window.innerWidth; H = window.innerHeight;
      cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR); ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      const target = Math.max(60, Math.min(130, Math.round(W * H / 17000)));
      while (pts.length < target) pts.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .26, vy: (Math.random() - .5) * .26, c: hues[pts.length % hues.length] });
      pts.length = target;
    }
    size();
    const mouse = { x: -999, y: -999 };
    window.addEventListener("pointermove", e => { mouse.x = e.clientX; mouse.y = e.clientY; });
    const reduce = false; /* animations always on, by explicit request */
    let raf = 0;
    function frame() {
      ctx.clearRect(0, 0, W, H);
      for (const p of pts) {
        const dx = mouse.x - p.x, dy = mouse.y - p.y, d2 = dx * dx + dy * dy;
        if (d2 < 26000 && d2 > 1) { p.vx += dx / d2 * 7; p.vy += dy / d2 * 7; }
        p.vx *= .99; p.vy *= .99; p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1; if (p.y < 0 || p.y > H) p.vy *= -1;
        p.x = Math.max(0, Math.min(W, p.x)); p.y = Math.max(0, Math.min(H, p.y));
      }
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j], dd = Math.hypot(a.x - b.x, a.y - b.y);
        if (dd < 150) { ctx.strokeStyle = a.c; ctx.globalAlpha = (1 - dd / 150) * .34; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      }
      for (const p of pts) { ctx.fillStyle = p.c; ctx.globalAlpha = .92; ctx.beginPath(); ctx.arc(p.x, p.y, 2.9, 0, 7); ctx.fill(); }
      ctx.globalAlpha = 1;
      if (!reduce) raf = requestAnimationFrame(frame);
    }
    document.addEventListener("visibilitychange", () => { if (!document.hidden && !reduce && !raf) { raf = requestAnimationFrame(frame); } });
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(size, 180); });
    frame();
  }

  /* hero card gets its OWN technique — a slow aurora of drifting colour, distinct from the page constellation */
  function heroAurora(cv) {
    const ctx = cv.getContext("2d"); if (!ctx) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const cols = ["#6366f1", "#a855f7", "#ec4899", "#0ea5b7", "#3b6ff0", "#12a594"];
    const hexA = (h, a) => { const n = parseInt(h.slice(1), 16); return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")"; };
    let W = 0, H = 0, blobs = [];
    function size() { const r = cv.getBoundingClientRect(); W = r.width || 760; H = r.height || 200; cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); }
    size();
    for (let i = 0; i < 6; i++) blobs.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .16, vy: (Math.random() - .5) * .12, r: Math.max(W, H) * (0.34 + Math.random() * 0.24), c: cols[i % cols.length], ph: Math.random() * 7 });
    const reduce = false; /* animations always on, by explicit request */
    let raf = 0, t = 0;
    function frame() {
      if (!cv.isConnected) { raf = 0; return; }
      ctx.clearRect(0, 0, W, H); t += 0.006;
      for (const b of blobs) {
        b.x += b.vx + Math.sin(t + b.ph) * 0.14; b.y += b.vy + Math.cos(t * 0.8 + b.ph) * 0.11;
        if (b.x < -b.r) b.x = W + b.r; if (b.x > W + b.r) b.x = -b.r;
        if (b.y < -b.r) b.y = H + b.r; if (b.y > H + b.r) b.y = -b.r;
        const g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
        g.addColorStop(0, hexA(b.c, 0.15)); g.addColorStop(1, hexA(b.c, 0));
        ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      }
      if (!reduce) raf = requestAnimationFrame(frame);
    }
    document.addEventListener("visibilitychange", () => { if (!document.hidden && !reduce && cv.isConnected && !raf) raf = requestAnimationFrame(frame); });
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(size, 200); });
    frame();
  }

  /* ============================================================
     SPACED-REPETITION REVIEW
     ============================================================ */
  function renderReview() {
    contentEl.innerHTML = ""; contentEl.removeAttribute("data-track"); clearMeta();
    let queue = dueCards().map(cardQuestion).filter(Boolean);
    const studyAhead = queue.length === 0;
    if (studyAhead) queue = Object.keys(srs()).map(cardQuestion).filter(Boolean).sort(() => Math.random() - 0.5).slice(0, 10);
    const wrap = el("div", "review"); contentEl.appendChild(wrap);
    document.title = "Review — Math for AI Engineers";

    if (!queue.length) {
      wrap.appendChild(el("div", "review-empty",
        `<h2>Nothing to review yet</h2>
         <p>Take a lesson's quiz and its questions start appearing here for spaced review — resurfaced at growing intervals so they stick. Study a lesson or two first.</p>`));
      wrap.appendChild(actionBtn("Go to your roadmap", "#home", "primary"));
      scrollTop(); return;
    }
    queue.sort(() => Math.random() - 0.5);
    let idx = 0, correctCount = 0;
    const head = el("div", "review-head"), card = el("div", "review-card");
    wrap.append(head, card);

    function showCard() {
      const item = queue[idx];
      head.innerHTML =
        `<div class="review-progress">Review · card ${idx + 1} of ${queue.length}${studyAhead ? " · studying ahead" : ""}</div>
         <a class="review-from" href="#${item.lessonId}">from ${item.lessonId} — ${item.title}</a>`;
      card.innerHTML = "";
      card.appendChild(el("div", "review-q", item.q.q));
      const opts = el("div", "review-opts"); let answered = false;
      const order = item.q.options.map((_, i) => i);
      for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(Math.random() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
      order.forEach(origIdx => {
        const b = el("button", "review-opt", item.q.options[origIdx]);
        b.addEventListener("click", () => {
          if (answered) return; answered = true;
          const right = origIdx === item.q.answer; if (right) correctCount++;
          card.querySelectorAll(".review-opt").forEach((btn, bi) => { if (order[bi] === item.q.answer) btn.classList.add("correct"); });
          if (!right) b.classList.add("wrong");
          gradeCard("q:" + item.lessonId + ":" + item.qi, right);
          const exp = el("div", "review-exp " + (right ? "ok" : "no"), (right ? "<b>Correct.</b> " : "<b>Not quite.</b> ") + item.q.explain);
          card.appendChild(exp); typesetMath(exp);
          const next = el("button", "review-next", idx + 1 < queue.length ? "Next card ›" : "Finish review");
          next.addEventListener("click", () => { idx++; idx < queue.length ? showCard() : finish(); });
          card.appendChild(next);
        });
        opts.appendChild(b);
      });
      card.appendChild(opts); typesetMath(card); scrollTop();
    }
    function finish() {
      touchStreak(); refreshStats();
      head.innerHTML = ""; card.innerHTML = "";
      wrap.appendChild(el("div", "review-done",
        `<h2>Review complete 🎉</h2>
         <p>You got <b>${correctCount} of ${queue.length}</b> right. Correct cards move to a longer interval; missed ones return tomorrow.</p>`));
      const row = el("div", "dash-actions");
      row.appendChild(actionBtn("Back to home", "#home", "primary"));
      const remaining = dueCards().length; if (remaining) row.appendChild(actionBtn(`Review ${remaining} more`, "#review", ""));
      wrap.appendChild(row); scrollTop();
    }
    showCard();
  }

  /* ============================================================
     ROUTING + GLOBAL UI
     ============================================================ */
  function scrollTop() {
    window.scrollTo({ top: 0, behavior: "auto" });
    document.querySelector(".main").scrollTop = 0;
    document.documentElement.scrollTop = 0;
  }

  function go() {
    const raw = (location.hash || "").replace(/^#/, "");
    if (raw === "home")   { renderDashboard(); refreshStats(); if (window.innerWidth <= 880) closeMenu(); return; }
    if (raw === "review") { renderReview();    refreshStats(); if (window.innerWidth <= 880) closeMenu(); return; }
    const id = raw || LS.get("last", null);
    if (!id) { location.hash = "home"; return; }                    // first visit → dashboard
    if (("#" + id) !== location.hash) { location.hash = id; return; } // will retrigger
    renderLesson(id); refreshStats();
    if (window.innerWidth <= 880) closeMenu();
  }

  /* search */
  document.getElementById("search").addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll(".track").forEach(tr => {
      let any = false;
      tr.querySelectorAll(".lessons li").forEach(li => {
        const match = li.textContent.toLowerCase().includes(q);
        li.style.display = match ? "" : "none";
        if (match) any = true;
      });
      tr.style.display = any || !q ? "" : "none";
      if (q) tr.classList.remove("collapsed");
    });
  });

  /* theme */
  const themeBtn = document.getElementById("themeToggle");
  function applyTheme(t) { document.documentElement.setAttribute("data-theme", t); LS.set("theme", t); }
  applyTheme(LS.get("theme", "light"));
  themeBtn.addEventListener("click", () =>
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

  /* mobile menu */
  const sidebar = document.getElementById("sidebar"), app = document.getElementById("app");
  function closeMenu() { sidebar.classList.remove("open"); app.classList.remove("nav-open"); }
  document.getElementById("menuToggle").addEventListener("click", () => {
    sidebar.classList.toggle("open"); app.classList.toggle("nav-open");
  });
  app.addEventListener("click", (e) => { if (e.target === app && app.classList.contains("nav-open")) closeMenu(); });

  /* keyboard arrows */
  document.addEventListener("keydown", (e) => {
    if (/INPUT|TEXTAREA/.test((document.activeElement || {}).tagName)) return;
    const id = (location.hash || "").replace(/^#/, ""); const i = indexOf(id);
    if (e.key === "ArrowLeft" && FLAT[i - 1]) location.hash = FLAT[i - 1].id;
    if (e.key === "ArrowRight" && FLAT[i + 1]) location.hash = FLAT[i + 1].id;
  });

  window.addEventListener("hashchange", go);

  /* ---- first-run welcome / how-to (reopen with the "? Guide" button) ---- */
  function showWelcome() {
    if (document.querySelector(".welcome-overlay")) return;
    const last = LS.get("last", null);
    const firstId = (realLessons(CURRICULUM[0])[0] || {}).id || "1.1";
    const useLast = last && CONTENT[last];
    const startId = useLast ? last : firstId;
    const startLabel = useLast ? ("Continue &middot; " + last) : ("Start with " + firstId);
    const ov = el("div", "welcome-overlay");
    ov.innerHTML =
      `<div class="welcome-card">
         <div class="welcome-mark">&#8721;</div>
         <div class="welcome-eyebrow">Welcome</div>
         <h2>You're going to be excellent at the math behind AI.</h2>
         <p class="welcome-lead">This course takes you from first principles all the way to transformers &mdash; in plain English, every step shown. Here's the rhythm that makes it stick:</p>
         <div class="welcome-steps">
           <div class="wstep"><div class="wnum">1</div><div><b>Read</b> each lesson top to bottom &mdash; plain English first, then the math, then the derivation.</div></div>
           <div class="wstep"><div class="wnum">2</div><div><b>Play.</b> Drag the interactive demo and press &#9656; Run on the code. Watching it move is where it clicks.</div></div>
           <div class="wstep"><div class="wnum">3</div><div><b>Check yourself</b> on the quiz, and try a practice problem before opening the solution.</div></div>
           <div class="wstep"><div class="wnum">4</div><div><b>Come back daily.</b> A few minutes in the Review tab locks each idea into long-term memory.</div></div>
         </div>
         <div class="welcome-actions">
           <a class="dash-btn primary" id="welcomeStart" href="#${startId}">${startLabel} &nbsp;&rarr;</a>
           <button class="dash-btn" id="welcomeClose" type="button">I'll explore on my own</button>
         </div>
         <div class="welcome-foot">Take your time &mdash; understanding beats speed. Reopen this anytime with &ldquo;? Guide&rdquo; up top.</div>
       </div>`;
    document.body.appendChild(ov);
    function done() { LS.set("onboarded", true); ov.classList.add("closing"); setTimeout(() => ov.remove(), 260); }
    ov.querySelector("#welcomeStart").addEventListener("click", done);
    ov.querySelector("#welcomeClose").addEventListener("click", done);
    ov.addEventListener("click", e => { if (e.target === ov) done(); });
    requestAnimationFrame(() => ov.classList.add("show"));
  }

  /* ---- boot (wait for DOM + deferred KaTeX to be ready) ---- */
  function boot() {
    injectAppBackdrop(); injectChrome(); buildNav(); go(); refreshStats();
    const hb = document.getElementById("helpBtn");
    if (hb) hb.addEventListener("click", showWelcome);
    if (!LS.get("onboarded", false)) setTimeout(showWelcome, 450);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
