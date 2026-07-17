# Part 1: HTML head, CSS, body shell, Track definitions, start of LESSONS array
content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mathematics for AI Engineers</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--sb:#161b22;--card:#21262d;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;--purple:#a371f7;--orange:#f0883e;--sidebar-w:290px}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;line-height:1.65;display:flex;min-height:100vh}
#sidebar{width:var(--sidebar-w);min-width:var(--sidebar-w);background:var(--sb);border-right:1px solid var(--border);height:100vh;position:fixed;top:0;left:0;overflow-y:auto;z-index:100;display:flex;flex-direction:column;transition:transform .25s}
#sidebar.hidden{transform:translateX(-100%)}
#sb-hdr{padding:14px 13px 10px;border-bottom:1px solid var(--border)}
#sb-hdr h1{font-size:14px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:7px}
#sb-hdr h1 span{font-size:18px}
#pb-bg{background:var(--card);border-radius:99px;height:4px;margin:9px 0 3px}
#pb{height:4px;background:var(--blue);border-radius:99px;width:0%;transition:width .3s}
#plbl{font-size:10.5px;color:var(--muted)}
#track-list{flex:1;overflow-y:auto;padding:5px 0}
.th{display:flex;align-items:center;gap:7px;padding:8px 12px;cursor:pointer;user-select:none;background:var(--sb);position:sticky;top:0;z-index:1;border-bottom:1px solid transparent}
.th:hover{background:#1a1f2a}
.th.open{border-bottom-color:var(--border)}
.ta{font-size:9px;color:var(--muted);transition:transform .2s;margin-left:auto}
.ta.o{transform:rotate(90deg)}
.tn{font-size:10.5px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
.ll{display:none}
.ll.o{display:block}
.li{display:flex;align-items:center;gap:7px;padding:6px 11px 6px 28px;cursor:pointer;font-size:12px;color:var(--muted);transition:background .12s,color .12s;border-right:2px solid transparent}
.li:hover{background:var(--card);color:var(--text)}
.li.active{background:rgba(88,166,255,.08);color:var(--blue);border-right-color:var(--blue)}
.li.done .dot{background:var(--green)}
.dot{width:5px;height:5px;border-radius:50%;background:var(--border);flex-shrink:0}
.sp-li{padding:6px 11px 6px 28px;cursor:pointer;font-size:11px;color:var(--purple);display:flex;align-items:center;gap:5px;transition:background .12s}
.sp-li:hover{background:var(--card)}
#main{margin-left:var(--sidebar-w);flex:1;display:flex;flex-direction:column;min-height:100vh;transition:margin .25s}
#main.exp{margin-left:0}
#topbar{position:sticky;top:0;z-index:50;height:50px;background:rgba(13,17,23,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;padding:0 20px}
#mbtn{background:none;border:none;color:var(--muted);font-size:17px;cursor:pointer;padding:4px 8px;border-radius:5px}
#mbtn:hover{color:var(--text);background:var(--card)}
#bcrumb{font-size:11.5px;color:var(--muted);flex:1}
#ca{flex:1;max-width:800px;margin:0 auto;width:100%;padding:26px 20px 88px}
#lc{}
.tb{display:inline-flex;align-items:center;gap:5px;background:var(--card);border:1px solid var(--border);border-radius:18px;padding:3px 10px;font-size:10.5px;color:var(--muted);margin-bottom:8px}
.lh1{font-size:24px;font-weight:700;color:var(--text);margin-bottom:4px}
.ltb{margin-bottom:22px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.sec{margin-bottom:20px;border:1px solid var(--border);border-radius:9px;overflow:hidden}
.sh{display:flex;align-items:center;gap:9px;padding:11px 15px;background:var(--card);cursor:pointer;user-select:none}
.sh:hover{background:#282e38}
.si{font-size:13px}
.st{font-size:12.5px;font-weight:600;color:var(--text);flex:1}
.sc{font-size:9px;color:var(--muted);transition:transform .2s}
.sc.o{transform:rotate(90deg)}
.sb{padding:16px 17px;display:none}
.sb.o{display:block}
p{margin-bottom:7px}
p:last-child{margin-bottom:0}
h3{font-size:13.5px;font-weight:600;color:var(--text);margin:12px 0 5px}
strong{color:var(--text)}
.fb{background:rgba(88,166,255,.05);border:1px solid rgba(88,166,255,.18);border-radius:7px;padding:12px 14px;margin:9px 0;font-size:13px}
.fr{margin:5px 0;display:flex;align-items:flex-start;gap:10px;flex-wrap:wrap}
.fl{color:var(--muted);font-size:11px;min-width:120px;flex-shrink:0;padding-top:2px}
.dl{list-style:none;counter-reset:s}
.dl>li{counter-increment:s;display:grid;grid-template-columns:24px 1fr;gap:0 10px;padding:10px 0;border-bottom:1px solid var(--border)}
.dl>li:last-child{border-bottom:none}
.dl>li::before{content:counter(s);width:22px;height:22px;background:var(--blue);color:#000;border-radius:50%;font-size:10.5px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.slbl{font-weight:600;color:var(--blue);font-size:12px;margin-bottom:2px}
.sbdy{font-size:13px;line-height:1.6}
.cw{border-radius:7px;overflow:hidden;border:1px solid var(--border);margin:3px 0}
.ctb{background:#1c2128;padding:6px 11px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border)}
.clng{font-size:10px;color:var(--muted);font-family:monospace}
.cbs{display:flex;gap:6px}
.cb{background:var(--card);border:1px solid var(--border);color:var(--text);font-size:10.5px;padding:2px 9px;border-radius:4px;cursor:pointer;transition:background .12s}
.cb:hover{background:#2d333b}
.rb{background:rgba(63,185,80,.1);border-color:var(--green);color:var(--green)}
.rb:hover{background:rgba(63,185,80,.2)}
pre{margin:0}
pre code.hljs{font-family:'JetBrains Mono','Fira Code',monospace!important;font-size:12px!important;padding:14px!important}
.co{background:#090c10;border-top:1px solid var(--border);padding:11px 14px;font-family:monospace;font-size:12px;display:none}
.co.show{display:block}
.co img{max-width:100%;border-radius:5px;margin-top:7px;display:block}
.ot{color:#adbac7;white-space:pre-wrap}
.oe{color:var(--red);white-space:pre-wrap}
.dw{display:flex;justify-content:center;padding:5px 0}
.dw svg{max-width:100%}
.kpl{list-style:none}
.kpl li{padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;display:flex;gap:9px;align-items:flex-start}
.kpl li:last-child{border-bottom:none}
.kpl li::before{content:"›";color:var(--blue);font-weight:700;flex-shrink:0}
.mc{background:rgba(248,81,73,.04);border:1px solid rgba(248,81,73,.16);border-radius:7px;padding:12px 14px;margin-bottom:9px}
.mb{color:var(--red);font-weight:600;margin-bottom:4px;font-size:13px}
.mg{color:var(--green);font-size:13px}
.qq{margin-bottom:14px;padding:13px;background:var(--card);border-radius:7px}
.qt{font-size:13px;font-weight:500;margin-bottom:9px}
.qo{display:flex;flex-direction:column;gap:4px}
.qb{text-align:left;background:var(--bg);border:1px solid var(--border);color:var(--muted);padding:7px 12px;border-radius:5px;cursor:pointer;font-size:12.5px;transition:all .12s}
.qb:hover:not(:disabled){background:var(--card);color:var(--text);border-color:var(--blue)}
.qb.ok{background:rgba(63,185,80,.09);border-color:var(--green)!important;color:var(--green)!important}
.qb.no{background:rgba(248,81,73,.09);border-color:var(--red)!important;color:var(--red)!important}
.qe{margin-top:7px;padding:8px 10px;background:rgba(88,166,255,.06);border-radius:5px;font-size:12px;display:none;color:var(--text)}
.qe.show{display:block}
.pi{margin-bottom:11px;border:1px solid var(--border);border-radius:7px;overflow:hidden}
.ph{padding:12px 14px;background:var(--card);display:flex;align-items:flex-start;gap:9px}
.db{font-size:10px;font-weight:700;padding:2px 7px;border-radius:16px;flex-shrink:0;margin-top:2px}
.de{background:rgba(63,185,80,.13);color:var(--green)}
.dm{background:rgba(240,136,62,.13);color:var(--orange)}
.dh{background:rgba(248,81,73,.13);color:var(--red)}
.dai{background:rgba(163,113,247,.13);color:var(--purple)}
.pp{font-size:13px}
.pt{padding:7px 14px;font-size:11.5px;color:var(--blue);cursor:pointer;background:none;border:none;display:flex;align-items:center;gap:5px;border-top:1px solid var(--border);width:100%}
.pt:hover{background:var(--card)}
.ps{padding:13px 14px;font-size:13px;border-top:1px solid var(--border);background:rgba(63,185,80,.02);display:none;line-height:1.7}
.ps.o{display:block}
.ddw{background:rgba(163,113,247,.03);border:1px solid rgba(163,113,247,.16);border-radius:7px;padding:14px 16px;font-size:13px}
.ddw h3{color:var(--purple)}
#bn{position:fixed;bottom:0;right:0;left:var(--sidebar-w);background:rgba(13,17,23,.96);backdrop-filter:blur(8px);border-top:1px solid var(--border);padding:10px 20px;display:flex;justify-content:space-between;align-items:center;z-index:50;transition:left .25s}
#bn.exp{left:0}
.nb{background:var(--card);border:1px solid var(--border);color:var(--text);padding:7px 16px;border-radius:6px;cursor:pointer;font-size:12.5px;transition:all .15s;display:flex;align-items:center;gap:6px}
.nb:hover:not(:disabled){background:#2d333b;border-color:var(--blue)}
.nb:disabled{opacity:.32;cursor:default}
.nb.p{background:var(--blue);color:#0d1117;border-color:var(--blue);font-weight:600}
.nb.p:hover{background:#79b8ff}
#nc{font-size:11.5px;color:var(--muted)}
#pyl{position:fixed;top:59px;right:16px;background:var(--card);border:1px solid var(--border);border-radius:7px;padding:8px 13px;font-size:11.5px;color:var(--muted);z-index:200;display:none;align-items:center;gap:7px}
#pyl.show{display:flex}
.sp{width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.katex-display{overflow-x:auto;padding:3px 0}
@media(max-width:700px){#sidebar{width:100%;transform:translateX(-110%)}#sidebar.open{transform:none}#main,#bn{margin-left:0;left:0}}
</style>
</head>
<body>
<aside id="sidebar">
  <div id="sb-hdr">
    <h1><span>🧮</span> Math for AI Engineers</h1>
    <div id="pb-bg"><div id="pb"></div></div>
    <div id="plbl">0 / 75 lessons</div>
  </div>
  <nav id="track-list"></nav>
</aside>
<main id="main">
  <div id="topbar">
    <button id="mbtn" onclick="toggleSB()">☰</button>
    <span id="bcrumb">Select a lesson to begin</span>
  </div>
  <div id="ca"><div id="lc">
    <div style="text-align:center;padding:56px 16px 20px;color:var(--muted)">
      <div style="font-size:50px;margin-bottom:12px">🧮</div>
      <h2 style="font-size:22px;color:var(--text);margin-bottom:7px">Mathematics for AI Engineers</h2>
      <p style="max-width:400px;margin:0 auto;line-height:1.7">13 tracks · 75 lessons · Full derivations, runnable Python, and an AI anchor for every concept.</p>
      <button class="nb p" style="margin:22px auto 0;display:inline-flex" onclick="goTo(0)">Start Track 1 →</button>
    </div>
  </div></div>
</main>
<div id="bn">
  <button class="nb" id="pb2" onclick="navigate(-1)" disabled>← Prev</button>
  <span id="nc"></span>
  <button class="nb p" id="nb2" onclick="navigate(1)" disabled>Next →</button>
</div>
<div id="pyl"><div class="sp"></div><span id="pyst">Loading Python…</span></div>
<script>
const TRACKS=[
{id:1,title:"Arithmetic & Algebra",emoji:"🔢",color:"#f59e0b"},
{id:2,title:"Linear Algebra I — Vectors & Matrices",emoji:"📐",color:"#3b82f6"},
{id:3,title:"Linear Algebra II — Systems, Spaces & Rank",emoji:"🔲",color:"#06b6d4"},
{id:4,title:"Linear Algebra III — Norms, Distances & Inverses",emoji:"📏",color:"#8b5cf6"},
{id:5,title:"Linear Algebra IV — Eigenvalues, PCA & SVD",emoji:"🌀",color:"#ec4899"},
{id:6,title:"Calculus I — Derivatives & Gradients",emoji:"📈",color:"#10b981"},
{id:7,title:"Calculus II — Integration & Expectation",emoji:"∫",color:"#14b8a6"},
{id:8,title:"Optimization Theory",emoji:"⛰️",color:"#f97316"},
{id:9,title:"Probability I — Foundations",emoji:"🎲",color:"#eab308"},
{id:10,title:"Probability II — Continuous Distributions",emoji:"🔔",color:"#a855f7"},
{id:11,title:"Statistics & Estimation",emoji:"📊",color:"#ef4444"},
{id:12,title:"Information Theory",emoji:"💡",color:"#06b6d4"},
{id:13,title:"Numerical Methods",emoji:"⚙️",color:"#84cc16"}
];
const LESSONS=[
"""
with open('/sessions/peaceful-serene-bardeen/mnt/Mathematics/ai-math/index.html', 'w') as f:
    f.write(content)
print("Part 1 written:", len(content), "bytes")
