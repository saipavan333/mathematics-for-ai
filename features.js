/* features.js — study-hub extras for "Mathematics for AI Engineers".
   Adds: Glossary page, inline term tooltips, Cheat sheets, Interview bank,
   Job-readiness exam, Concept map, Read-aloud, and PWA registration.
   Self-contained & namespaced (.mh / #mh-*). Hooks into app.js via window.MHUB
   (a route registry + an after-lesson callback) so the core engine is untouched. */
(function () {
  "use strict";
  if (window.__mhLoaded) return; window.__mhLoaded = true;

  var CUR = window.CURRICULUM || [];
  var C = window.LESSON_CONTENT || {};
  var NUT = window.NUTSHELLS || {};
  function glossary() { return (window.GD_ASSIST && window.GD_ASSIST.glossary) || []; }

  var PHASES = [
    { name: "Foundations", blurb: "The algebra and linear algebra everything else stands on.", tracks: ["1", "2", "3", "4", "5"] },
    { name: "Calculus & Optimization", blurb: "Derivatives, gradients, and how models actually learn.", tracks: ["6", "7", "8"] },
    { name: "Probability & Statistics", blurb: "Reasoning under uncertainty; estimating from data.", tracks: ["9", "10", "11"] },
    { name: "Information & Computation", blurb: "Entropy, and doing all this numerically and stably.", tracks: ["12", "13"] },
    { name: "The Bridge to ML", blurb: "Matrix calculus / backprop and Bayesian inference.", tracks: ["14", "15"] },
    { name: "Modern Deep Learning", blurb: "Diffusion, generative models, and the transformer.", tracks: ["16", "17"] },
    { name: "Capstone Projects", blurb: "Five things you build from scratch that actually run.", tracks: ["C"] }
  ];

  /* ---------- tiny helpers ---------- */
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function esc(s) { var d = document.createElement("div"); d.textContent = (s == null ? "" : String(s)); return d.innerHTML; }
  function ce() { return document.getElementById("content"); }
  function trackById(id) { for (var i = 0; i < CUR.length; i++) if (CUR[i].id === id) return CUR[i]; return null; }
  function realLessons(tr) { return (tr.lessons || []).filter(function (l) { return C[l.id] && !C[l.id].exam; }); }
  function lsGet(k, d) { try { var v = localStorage.getItem("mathai:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }
  function typeset(root) {
    if (!window.renderMathInElement) { setTimeout(function () { typeset(root); }, 150); return; }
    try {
      window.renderMathInElement(root, {
        delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false },
        { left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false, ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      });
    } catch (e) { }
  }
  function setActive(route) {
    document.querySelectorAll(".lessons a.active").forEach(function (a) { a.classList.remove("active"); });
    document.querySelectorAll(".sidenav-btn.mh-active").forEach(function (a) { a.classList.remove("mh-active"); });
    var b = document.getElementById("mh-nav-" + route); if (b) b.classList.add("mh-active");
    var cr = document.getElementById("crumbs"); if (cr) cr.innerHTML = "<b>Study hub</b>";
    var ln = document.getElementById("lessonNav"); if (ln) ln.innerHTML = "";
  }
  function mount(node, title) {
    var c = ce(); if (!c) return; c.innerHTML = ""; c.removeAttribute("data-track"); c.appendChild(node);
    document.title = title + " — Math for AI Engineers"; window.scrollTo(0, 0);
    if (window.innerWidth <= 880) { var sb = document.getElementById("sidebar"); if (sb) sb.classList.remove("open"); }
  }

  /* ============================================================ GLOSSARY */
  function renderGlossary() {
    var g = glossary().slice().sort(function (a, b) { return a.term.localeCompare(b.term); });
    var wrap = el("div", "mh-page");
    wrap.appendChild(el("div", "mh-head", '<div class="mh-eyebrow">Reference</div><h2>Glossary</h2>' +
      '<p class="mh-lead">' + g.length + ' terms from across the course, in plain English — each with the key equation where it matters.</p>'));
    var search = el("input", "mh-search"); search.type = "search"; search.placeholder = "Filter terms…"; search.setAttribute("aria-label", "Filter glossary terms");
    wrap.appendChild(search);
    var list = el("dl", "mh-gloss-list"); wrap.appendChild(list);
    function draw(q) {
      list.innerHTML = ""; q = (q || "").toLowerCase().trim();
      var shown = g.filter(function (t) { return !q || t.term.toLowerCase().indexOf(q) >= 0 || t.desc.toLowerCase().indexOf(q) >= 0; });
      if (!shown.length) { list.appendChild(el("p", "mh-empty", "No terms match “" + esc(q) + "”.")); return; }
      shown.forEach(function (t) {
        var item = el("div", "mh-gloss-item", "<dt>" + esc(t.term) + "</dt><dd>" + esc(t.desc) + "</dd>");
        list.appendChild(item);
      });
      typeset(list);
    }
    search.addEventListener("input", function () { draw(search.value); });
    draw(""); mount(wrap, "Glossary"); setTimeout(function () { try { search.focus(); } catch (e) { } }, 50);
  }

  /* ============================================================ CHEAT SHEETS */
  function renderCheats() {
    var wrap = el("div", "mh-page");
    wrap.appendChild(el("div", "mh-head", '<div class="mh-eyebrow">Reference</div><h2>Cheat sheets</h2>' +
      '<p class="mh-lead">Every track’s key results in one place. Tap any lesson to open it in full.</p>'));
    CUR.forEach(function (tr) {
      var ls = realLessons(tr); if (!ls.length) return;
      var card = el("section", "mh-cheat-card"); card.style.setProperty("--th", "var(--h" + tr.id + ")");
      card.appendChild(el("h3", "mh-cheat-h", '<span class="mh-cheat-num">' + esc(tr.id) + "</span> " + esc(tr.title)));
      var body = el("div", "mh-cheat-body");
      ls.forEach(function (l) {
        var c = C[l.id], kps = (c && c.keyPoints) || [];
        var item = el("div", "mh-cheat-item");
        var a = el("a", "mh-cheat-lnk", '<span class="mh-cheat-id">' + esc(l.id) + "</span> " + esc(l.title)); a.href = "#" + l.id;
        item.appendChild(a);
        if (kps.length) { var ul = el("ul", "mh-cheat-kps"); kps.slice(0, 4).forEach(function (k) { ul.appendChild(el("li", null, k)); }); item.appendChild(ul); }
        else if (NUT[l.id]) item.appendChild(el("p", "mh-cheat-nut", esc(NUT[l.id])));
        body.appendChild(item);
      });
      card.appendChild(body); wrap.appendChild(card);
    });
    mount(wrap, "Cheat sheets"); typeset(wrap);
  }

  /* ============================================================ INTERVIEW BANK */
  function allPractice() {
    var out = [];
    CUR.forEach(function (tr) {
      realLessons(tr).forEach(function (l) {
        var c = C[l.id]; (c.practice || []).forEach(function (p) {
          if (!p || !p.prompt) return;
          out.push({ track: tr.id, lesson: l.id, lessonTitle: l.title, level: (p.level || "med"), prompt: p.prompt, solution: p.solution || "" });
        });
      });
    });
    return out;
  }
  function levLabel(l) { return l === "easy" ? "Easy" : l === "hard" ? "Hard" : "Medium"; }
  function renderInterview() {
    var items = allPractice();
    var wrap = el("div", "mh-page");
    wrap.appendChild(el("div", "mh-head", '<div class="mh-eyebrow">Get hired</div><h2>Interview question bank</h2>' +
      '<p class="mh-lead">' + items.length + ' practice problems drawn from every lesson, tagged by difficulty. Think it through, then reveal the worked solution.</p>'));
    var filters = el("div", "mh-iv-filters");
    var curLevel = "all", curTrack = "all";
    var levels = [["all", "All"], ["easy", "Easy"], ["med", "Medium"], ["hard", "Hard"]];
    var lvWrap = el("div", "mh-chip-row");
    levels.forEach(function (lv) {
      var b = el("button", "mh-chip" + (lv[0] === "all" ? " on" : ""), lv[1]); b.type = "button";
      b.addEventListener("click", function () { curLevel = lv[0]; lvWrap.querySelectorAll(".mh-chip").forEach(function (x) { x.classList.remove("on"); }); b.classList.add("on"); draw(); });
      lvWrap.appendChild(b);
    });
    filters.appendChild(lvWrap);
    var sel = el("select", "mh-select"); sel.setAttribute("aria-label", "Filter by track");
    sel.appendChild(el("option", null, "All tracks")); sel.lastChild.value = "all";
    CUR.forEach(function (tr) { if (!realLessons(tr).length) return; var o = el("option", null, esc(tr.id + " · " + (tr.short || tr.title))); o.value = tr.id; sel.appendChild(o); });
    sel.addEventListener("change", function () { curTrack = sel.value; draw(); });
    filters.appendChild(sel);
    wrap.appendChild(filters);
    var count = el("div", "mh-count"); wrap.appendChild(count);
    var list = el("div", "mh-iv-list"); wrap.appendChild(list);
    function draw() {
      list.innerHTML = "";
      var shown = items.filter(function (it) { return (curLevel === "all" || it.level === curLevel) && (curTrack === "all" || it.track === curTrack); });
      count.textContent = shown.length + " question" + (shown.length === 1 ? "" : "s");
      shown.forEach(function (it) {
        var q = el("div", "mh-iv-q");
        q.appendChild(el("div", "mh-iv-meta", '<span class="mh-badge mh-lv-' + it.level + '">' + levLabel(it.level) + "</span>"));
        var src = el("a", "mh-iv-src", esc(it.lesson) + " · " + esc(it.lessonTitle)); src.href = "#" + it.lesson; q.querySelector(".mh-iv-meta").appendChild(src);
        q.appendChild(el("div", "mh-iv-prompt", it.prompt));
        if (it.solution) {
          var sol = el("div", "mh-iv-sol", it.solution); sol.hidden = true;
          var btn = el("button", "mh-reveal", "Show solution"); btn.type = "button";
          btn.addEventListener("click", function () { sol.hidden = !sol.hidden; btn.textContent = sol.hidden ? "Show solution" : "Hide solution"; if (!sol.hidden) typeset(sol); });
          q.appendChild(btn); q.appendChild(sol);
        }
        list.appendChild(q);
      });
      typeset(list);
    }
    draw(); mount(wrap, "Interview bank");
  }

  /* ============================================================ JOB-READINESS EXAM */
  function allQuiz() {
    var out = [];
    CUR.forEach(function (tr) {
      realLessons(tr).forEach(function (l) {
        var c = C[l.id]; (c.quiz || []).forEach(function (q) {
          if (q && q.options && typeof q.answer === "number") out.push({ track: tr.id, lesson: l.id, q: q.q, options: q.options, answer: q.answer, explain: q.explain || "" });
        });
      });
    });
    return out;
  }
  function shuffle(a) { a = a.slice(); for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; } return a; }
  function buildExam(n) {
    var byTrack = {}; allQuiz().forEach(function (q) { (byTrack[q.track] = byTrack[q.track] || []).push(q); });
    var picks = [], keys = Object.keys(byTrack);
    keys.forEach(function (k) { var s = shuffle(byTrack[k]); picks.push(s[0]); if (s[1]) picks.push(s[1]); });      // ensure coverage: up to 2 / track
    picks = shuffle(picks);
    return picks.slice(0, n);
  }
  function renderExam() {
    var qs = buildExam(20);
    var wrap = el("div", "mh-page");
    wrap.appendChild(el("div", "mh-head", '<div class="mh-eyebrow">Get hired</div><h2>Job-readiness exam</h2>' +
      '<p class="mh-lead">' + qs.length + ' questions spanning the whole course. Answer them all, then submit to see your score and every explanation. Aim for 80%+.</p>'));
    var form = el("div", "mh-exam"); var picked = {};
    qs.forEach(function (q, qi) {
      var box = el("div", "mh-exam-q");
      box.appendChild(el("div", "mh-exam-qt", (qi + 1) + ". " + q.q));
      var order = shuffle(q.options.map(function (_, i) { return i; }));
      var opts = el("div", "mh-exam-opts");
      order.forEach(function (origIdx) {
        var b = el("button", "mh-exam-opt", esc(q.options[origIdx])); b.type = "button";
        b.addEventListener("click", function () {
          if (form.classList.contains("done")) return;
          opts.querySelectorAll(".mh-exam-opt").forEach(function (x) { x.classList.remove("sel"); });
          b.classList.add("sel"); picked[qi] = origIdx;
        });
        opts.appendChild(b);
      });
      box.appendChild(opts);
      var exp = el("div", "mh-exam-exp"); exp.hidden = true; box.appendChild(exp);
      box._grade = function () {
        var chosen = picked[qi]; var btns = opts.querySelectorAll(".mh-exam-opt");
        order.forEach(function (origIdx, k) { if (origIdx === q.answer) btns[k].classList.add("correct"); if (origIdx === chosen && chosen !== q.answer) btns[k].classList.add("wrong"); });
        exp.innerHTML = (chosen === q.answer ? "<b>Correct.</b> " : (chosen == null ? "<b>Skipped.</b> " : "<b>Not quite.</b> ")) + q.explain +
          ' <a class="mh-iv-src" href="#' + q.lesson + '">Review ' + esc(q.lesson) + "</a>";
        exp.hidden = false; typeset(exp);
        return chosen === q.answer;
      };
      form.appendChild(box);
    });
    var submit = el("button", "mh-exam-submit", "Submit exam"); submit.type = "button";
    var result = el("div", "mh-exam-result"); result.hidden = true;
    submit.addEventListener("click", function () {
      if (form.classList.contains("done")) { location.hash = "exam"; renderExam(); return; }
      var score = 0; [].forEach.call(form.querySelectorAll(".mh-exam-q"), function (b) { if (b._grade()) score++; });
      form.classList.add("done");
      var pct = Math.round(score / qs.length * 100);
      result.className = "mh-exam-result " + (pct >= 80 ? "pass" : "fail");
      result.innerHTML = "<b>" + score + " / " + qs.length + "</b> · " + pct + "%<br>" +
        (pct >= 80 ? "Job-ready on these fundamentals — nicely done." : "Solid start — review the flagged lessons and retake.");
      result.hidden = false; submit.textContent = "Retake with fresh questions"; result.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    wrap.appendChild(form); wrap.appendChild(submit); wrap.appendChild(result);
    mount(wrap, "Job-readiness exam"); typeset(form);
  }

  /* ============================================================ CONCEPT MAP */
  function renderMap() {
    var done = new Set(lsGet("done", []));
    var wrap = el("div", "mh-page");
    wrap.appendChild(el("div", "mh-head", '<div class="mh-eyebrow">Overview</div><h2>Concept map</h2>' +
      '<p class="mh-lead">The whole curriculum, in dependency order — seven phases, ' + CUR.length + ' tracks. Tap any track to jump in.</p>'));
    var map = el("div", "mh-map");
    PHASES.forEach(function (ph, pi) {
      var band = el("div", "mh-phase");
      band.appendChild(el("div", "mh-phase-side", '<span class="mh-phase-n">' + (pi + 1) + "</span>"));
      var main = el("div", "mh-phase-main");
      main.appendChild(el("div", "mh-phase-h", "<h3>" + esc(ph.name) + "</h3><p>" + esc(ph.blurb) + "</p>"));
      var row = el("div", "mh-track-row");
      ph.tracks.forEach(function (tid) {
        var tr = trackById(tid); if (!tr) return;
        var ls = realLessons(tr); var d = ls.filter(function (l) { return done.has(l.id); }).length;
        var first = (ls[0] || (tr.lessons || [])[0] || {}).id;
        var pct = ls.length ? Math.round(d / ls.length * 100) : 0;
        var node = el("a", "mh-track-node" + (pct === 100 ? " done" : ""));
        node.href = "#" + (first || "home"); node.style.setProperty("--th", "var(--h" + tr.id + ")");
        node.innerHTML = '<span class="mh-track-id">' + esc(tr.id) + '</span>' +
          '<span class="mh-track-name">' + esc(tr.short || tr.title) + '</span>' +
          '<span class="mh-track-prog"><span style="width:' + pct + '%"></span></span>' +
          '<span class="mh-track-meta">' + d + " / " + ls.length + " done</span>";
        row.appendChild(node);
      });
      main.appendChild(row); band.appendChild(main); map.appendChild(band);
    });
    wrap.appendChild(map); mount(wrap, "Concept map");
  }

  /* ============================================================ INLINE TOOLTIPS */
  var TERMS = null;
  function termList() {
    if (TERMS) return TERMS;
    TERMS = glossary().map(function (g) { return { t: g.term, low: g.term.toLowerCase(), desc: g.desc }; })
      .filter(function (x) { return x.low.length >= 4 && !/\s/.test(x.low) || x.low.length >= 4; })
      .sort(function (a, b) { return b.low.length - a.low.length; });
    return TERMS;
  }
  function applyTooltips(root) {
    var terms = termList(); if (!terms.length) return;
    var used = {}, SKIP = { A: 1, CODE: 1, PRE: 1, BUTTON: 1, SVG: 1, TEXT: 1, TSPAN: 1, H2: 1, H3: 1, STYLE: 1, SCRIPT: 1, OPTION: 1, SELECT: 1 };
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        for (var p = n.parentNode; p && p !== root; p = p.parentNode) {
          var nm = (p.nodeName || "").toUpperCase(); if (SKIP[nm]) return NodeFilter.FILTER_REJECT;
          if (p.classList && (p.classList.contains("katex") || p.classList.contains("codecell") || p.classList.contains("mh-term") || p.classList.contains("section-label") || p.classList.contains("nutshell-tag"))) return NodeFilter.FILTER_REJECT;
        }
        return (n.nodeValue && n.nodeValue.trim().length > 4) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    var nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    var capped = 0;
    nodes.forEach(function (node) {
      if (capped >= 60) return;                                   // don't over-decorate a page
      var text = node.nodeValue, hits = [];
      for (var i = 0; i < terms.length; i++) {
        var term = terms[i]; if (used[term.low]) continue;
        var re = new RegExp("\\b" + term.low.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "i");
        var m = re.exec(text); if (!m) continue;
        hits.push({ start: m.index, end: m.index + m[0].length, hit: m[0], desc: term.desc, term: term.t, low: term.low });
      }
      if (!hits.length) return;
      hits.sort(function (a, b) { return a.start - b.start; });
      var kept = [], lastEnd = -1;
      hits.forEach(function (h) { if (h.start >= lastEnd) { kept.push(h); lastEnd = h.end; used[h.low] = 1; capped++; } });
      var frag = document.createDocumentFragment(), pos = 0;
      kept.forEach(function (h) {
        if (h.start > pos) frag.appendChild(document.createTextNode(text.slice(pos, h.start)));
        var span = el("span", "mh-term", esc(h.hit)); span.setAttribute("tabindex", "0"); span.setAttribute("role", "button");
        span.setAttribute("aria-label", h.term + ": " + h.desc);
        span.appendChild(el("span", "mh-tip", esc(h.desc)));
        span.addEventListener("click", function (e) { e.stopPropagation(); this.classList.toggle("open"); });
        frag.appendChild(span); pos = h.end;
      });
      if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  /* ============================================================ READ-ALOUD */
  function cleanText(node) { var c = node.cloneNode(true); [].forEach.call(c.querySelectorAll(".katex,.katex-display,.codecell,script,style,.mh-tip"), function (x) { x.remove(); }); return (c.textContent || "").replace(/\s+/g, " ").trim(); }
  function injectReadAloud(root) {
    if (!("speechSynthesis" in window)) return;
    var head = root.querySelector(".lesson-h"); if (!head || head.querySelector(".mh-ra")) return;
    var bar = el("div", "mh-ra");
    var btn = el("button", "mh-ra-btn", "▶ Read aloud"); btn.type = "button";
    var sel = el("select", "mh-ra-voice"); sel.setAttribute("aria-label", "Read-aloud voice");
    function loadVoices() { var vs = (window.speechSynthesis.getVoices() || []).filter(function (v) { return /^en/i.test(v.lang); }); sel.innerHTML = ""; if (!vs.length) { sel.appendChild(el("option", null, "Default")); return; } vs.forEach(function (v) { var o = el("option", null, esc(v.name.replace(/\(.*\)/, "").trim())); o.value = v.name; sel.appendChild(o); }); }
    loadVoices(); try { window.speechSynthesis.onvoiceschanged = loadVoices; } catch (e) { }
    var on = false;
    function stop() { try { window.speechSynthesis.cancel(); } catch (e) { } on = false; btn.innerHTML = "▶ Read aloud"; btn.classList.remove("on"); }
    btn.addEventListener("click", function () {
      if (on) { stop(); return; }
      var parts = [];
      var h = root.querySelector(".lesson-h h2"); if (h) parts.push(cleanText(h));
      var sub = root.querySelector(".lesson-sub"); if (sub) parts.push(cleanText(sub));
      var nut = root.querySelector(".nutshell"); if (nut) parts.push(cleanText(nut).replace(/^In one breath\s*/i, ""));
      [].forEach.call(root.querySelectorAll(".section-body"), function (sb) { if (sb.closest && sb.closest(".codecell")) return; [].forEach.call(sb.querySelectorAll("p,li"), function (p) { if (p.closest(".codecell")) return; var t = cleanText(p); if (t) parts.push(t); }); });
      var text = parts.join(". "); if (!text) return;
      var u = new SpeechSynthesisUtterance(text);
      var vs = window.speechSynthesis.getVoices() || []; for (var i = 0; i < vs.length; i++) { if (vs[i].name === sel.value) { u.voice = vs[i]; break; } }
      u.rate = 1; u.onend = stop; u.onerror = stop;
      try { window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); on = true; btn.innerHTML = "■ Stop"; btn.classList.add("on"); } catch (e) { }
    });
    bar.appendChild(btn); bar.appendChild(sel); head.appendChild(bar);
  }

  /* ============================================================ NAV + WIRING */
  function injectNav() {
    var sidenav = document.querySelector(".sidenav");
    if (!sidenav) { if ((injectNav._t = (injectNav._t || 0) + 1) < 50) setTimeout(injectNav, 120); return; }
    if (document.getElementById("mh-nav-glossary")) return;
    [["map", "&#9678; Concept map"], ["glossary", "&#9782; Glossary"], ["cheatsheets", "&#9636; Cheat sheets"],
    ["interview", "&#9998; Interview bank"], ["exam", "&#9873; Job-readiness exam"]].forEach(function (l) {
      var a = el("a", "sidenav-btn mh-nav-btn", l[1]); a.id = "mh-nav-" + l[0]; a.href = "#" + l[0]; sidenav.appendChild(a);
    });
  }

  window.MHUB = {
    routes: { glossary: 1, cheatsheets: 1, interview: 1, exam: 1, map: 1 },
    render: function (r) {
      setActive(r);
      if (r === "glossary") renderGlossary();
      else if (r === "cheatsheets") renderCheats();
      else if (r === "interview") renderInterview();
      else if (r === "exam") renderExam();
      else if (r === "map") renderMap();
    },
    onLesson: function (root) { try { applyTooltips(root); } catch (e) { } try { injectReadAloud(root); } catch (e) { } }
  };

  /* stop speech on any navigation */
  window.addEventListener("hashchange", function () { if (window.speechSynthesis) try { window.speechSynthesis.cancel(); } catch (e) { } });

  injectNav();
  /* if the page was opened directly on a feature hash (app.js ran go() before this file loaded), render it now */
  var raw = (location.hash || "").replace(/^#/, "");
  if (window.MHUB.routes[raw]) window.MHUB.render(raw);

  /* ---------- PWA ---------- */
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () { navigator.serviceWorker.register("sw.js").catch(function () { }); });
  }
})();
