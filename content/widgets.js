/* interactive lesson widgets — vanilla JS canvas, no external libs, theme-safe
   (each draws on a fixed light plate like the diagrams). window.WIDGETS[id](mount) */
window.WIDGETS = {};
(function () {
  var DPR = (typeof window !== "undefined" && window.devicePixelRatio) ? window.devicePixelRatio : 1;
  var PAL = { ink:"#1f2a44", ink2:"#4a5878", ink3:"#6b7891", axis:"#9aa6bd", grid:"#e7ecf4",
    blue:"#2a6f97", green:"#3a7d44", indigo:"#4f46e5", amber:"#b9722a", red:"#c0392b",
    accent:"#4f46e5", plate:"#f8fafc",
    blueF:"#dbe8fb", greenF:"#e3efe5", redF:"#f7e0dc", amberF:"#f7e7d2",
    indigoF:"#e5e7ff", neutralF:"#eef2f7", violet:"#7c4dff", line:"#e8eaef" };

  function elt(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function card(mount) { var w = elt("div", "widget"); mount.appendChild(w); return w; }
  function canvasIn(wrap, w, h) {
    var cv = elt("canvas"); cv.width = Math.round(w * DPR); cv.height = Math.round(h * DPR);
    cv.style.width = w + "px"; cv.style.height = h + "px"; cv.style.touchAction = "none";
    wrap.appendChild(cv); var ctx = cv.getContext("2d"); ctx.scale(DPR, DPR);
    return { cv: cv, ctx: ctx };
  }
  function controlsIn(wrap) { var c = elt("div", "widget-controls"); wrap.appendChild(c); return c; }
  function slider(controls, o) {
    var row = elt("label", "widget-row"); row.appendChild(elt("span", "widget-lab", o.label));
    var inp = elt("input"); inp.type = "range"; inp.min = o.min; inp.max = o.max; inp.step = o.step; inp.value = o.val;
    var out = elt("span", "widget-readout", o.fmt ? o.fmt(+o.val) : ("" + o.val));
    inp.addEventListener("input", function () { var v = +inp.value; out.textContent = o.fmt ? o.fmt(v) : ("" + v); o.on(v); });
    row.appendChild(inp); row.appendChild(out); controls.appendChild(row); return inp;
  }
  function button(controls, label, on) { var b = elt("button", "widget-btn", label); b.type = "button"; b.addEventListener("click", on); controls.appendChild(b); return b; }

  function plot(ctx, W, H, xmin, xmax, ymin, ymax, pad) {
    pad = pad || { l: 36, r: 16, t: 22, b: 26 };
    var px = pad.l, py = pad.t, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    function X(x) { return px + (x - xmin) / (xmax - xmin) * pw; }
    function Y(y) { return py + ph - (y - ymin) / (ymax - ymin) * ph; }
    return {
      X: X, Y: Y, px: px, py: py, pw: pw, ph: ph,
      invX: function (xp) { return xmin + (xp - px) / pw * (xmax - xmin); },
      invY: function (yp) { return ymin + (py + ph - yp) / ph * (ymax - ymin); },
      clear: function () { ctx.clearRect(0, 0, W, H); },
      grid: function (xs, ys) { ctx.strokeStyle = PAL.grid; ctx.lineWidth = 1; ctx.setLineDash([]);
        xs.forEach(function (x) { ctx.beginPath(); ctx.moveTo(X(x), py); ctx.lineTo(X(x), py + ph); ctx.stroke(); });
        ys.forEach(function (y) { ctx.beginPath(); ctx.moveTo(px, Y(y)); ctx.lineTo(px + pw, Y(y)); ctx.stroke(); }); },
      axes: function () { ctx.strokeStyle = PAL.axis; ctx.lineWidth = 1.3; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(px, py + ph); ctx.lineTo(px + pw, py + ph); ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.stroke(); },
      line: function (x1, y1, x2, y2, c, w, dash) { ctx.strokeStyle = c || PAL.ink; ctx.lineWidth = w || 2; ctx.setLineDash(dash || []);
        ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke(); ctx.setLineDash([]); },
      fn: function (f, a, b, c, w, n) { n = n || 140; ctx.strokeStyle = c || PAL.indigo; ctx.lineWidth = w || 2.4; ctx.setLineDash([]);
        ctx.beginPath(); var started = false;
        for (var i = 0; i <= n; i++) { var x = a + (b - a) * i / n; var y = f(x);
          if (!isFinite(y)) { started = false; continue; } var yy = Y(y);
          if (yy < py - 60 || yy > py + ph + 60) { started = false; continue; }
          if (!started) { ctx.moveTo(X(x), yy); started = true; } else ctx.lineTo(X(x), yy); }
        ctx.stroke(); },
      poly: function (pts, c, w, dash) { ctx.strokeStyle = c || PAL.ink; ctx.lineWidth = w || 2; ctx.setLineDash(dash || []);
        ctx.beginPath(); pts.forEach(function (p, i) { var xx = X(p[0]), yy = Y(p[1]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); }); ctx.stroke(); ctx.setLineDash([]); },
      fill: function (pts, c) { ctx.fillStyle = c; ctx.beginPath(); pts.forEach(function (p, i) { var xx = X(p[0]), yy = Y(p[1]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); }); ctx.closePath(); ctx.fill(); },
      dot: function (x, y, c, r) { ctx.fillStyle = c || PAL.ink; ctx.beginPath(); ctx.arc(X(x), Y(y), r || 4, 0, 7); ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.4; ctx.stroke(); },
      arrow: function (x1, y1, x2, y2, c, w) { var X1 = X(x1), Y1 = Y(y1), X2 = X(x2), Y2 = Y(y2); ctx.strokeStyle = ctx.fillStyle = c || PAL.ink; ctx.lineWidth = w || 2;
        var ang = Math.atan2(Y2 - Y1, X2 - X1), hl = 8; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X2 - hl * 0.8 * Math.cos(ang), Y2 - hl * 0.8 * Math.sin(ang)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X2, Y2); ctx.lineTo(X2 - hl * Math.cos(ang - 0.4), Y2 - hl * Math.sin(ang - 0.4)); ctx.lineTo(X2 - hl * Math.cos(ang + 0.4), Y2 - hl * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill(); },
      txt: function (xx, yy, s, c, size, align, weight) { ctx.fillStyle = c || PAL.ink; ctx.font = (weight ? weight + " " : "") + (size || 12) + "px system-ui,Segoe UI,Arial"; ctx.textAlign = align || "left"; ctx.textBaseline = "middle"; ctx.fillText(s, xx, yy); },
      title: function (s) { ctx.fillStyle = PAL.ink; ctx.font = "700 12.5px system-ui,Segoe UI,Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(s, W / 2, 12); }
    };
  }
  function reg(id, build) { window.WIDGETS[id] = build; }

  /* 6.1 — secant collapses into the tangent */
  reg("6.1", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 330), controls = controlsIn(wrap);
    var f = function (x) { return 0.25 * x * x; }, x0 = 2, h = 1.6;
    function draw() {
      var P = plot(C.ctx, 600, 330, -0.3, 4.3, -0.3, 3.7); P.clear(); P.grid([0,1,2,3,4],[0,1,2,3]); P.axes();
      P.title("Slide the second point toward P — the secant becomes the tangent");
      P.fn(f, -0.3, 3.9, PAL.indigo, 2.4);
      P.line(x0 - 1.4, f(x0) - 1.4, x0 + 1.4, f(x0) + 1.4, PAL.red, 2);
      var xq = x0 + h; P.line(x0, f(x0), xq, f(xq), PAL.green, 1.8, [5,3]);
      P.dot(xq, f(xq), PAL.green, 4); P.dot(x0, f(x0), PAL.red, 4.5);
      P.txt(P.X(x0) - 8, P.Y(f(x0)) + 2, "P", PAL.red, 12, "right", "700");
      var sec = (f(xq) - f(x0)) / h;
      P.txt(16, 314, "secant slope = " + sec.toFixed(3), PAL.green, 13, "left", "600");
      P.txt(360, 314, "tangent slope = 1.000", PAL.red, 13, "left", "600");
    }
    slider(controls, { label:"gap h", min:0.05, max:2, val:h, step:0.05, fmt:function(v){return v.toFixed(2);}, on:function(v){ h=v; draw(); } });
    draw();
  });

  /* 2.3 — drag vector b: dot product & projection */
  reg("2.3", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 360), controls = controlsIn(wrap);
    var a = [4,1], b = [2,3], cfg = { xmin:-0.5, xmax:5, ymin:-0.5, ymax:4 }, P;
    function draw() {
      P = plot(C.ctx, 600, 360, cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax); P.clear(); P.grid([0,1,2,3,4],[0,1,2,3]); P.axes();
      P.title("Drag the green dot (vector b) — watch a·b and the projection");
      var dot = a[0]*b[0] + a[1]*b[1], la = Math.hypot(a[0],a[1]), lb = Math.hypot(b[0],b[1]);
      var t = dot/(la*la), foot = [t*a[0], t*a[1]];
      P.line(b[0], b[1], foot[0], foot[1], PAL.amber, 1.6, [4,3]);
      P.arrow(0,0,a[0],a[1], PAL.blue, 2.6); P.arrow(0,0,b[0],b[1], PAL.green, 2.6);
      P.dot(foot[0], foot[1], PAL.amber, 3.5); P.dot(b[0], b[1], PAL.green, 5.5);
      P.txt(P.X(a[0])+6, P.Y(a[1]), "a", PAL.blue, 12, "left", "700");
      var cos = dot/(la*lb);
      P.txt(16, 344, "a·b = " + dot.toFixed(1), dot<0?PAL.red:PAL.ink2, 13, "left", "600");
      P.txt(210, 344, "cos θ = " + cos.toFixed(2), PAL.ink2, 13, "left", "600");
      P.txt(400, 344, Math.abs(dot)<0.35 ? "⊥ perpendicular" : (dot>0?"aligned":"opposed"), PAL.ink3, 12, "left");
    }
    draw();
    function fromMouse(e) { var r = C.cv.getBoundingClientRect(); var mx = e.clientX-r.left, my = e.clientY-r.top;
      b = [Math.max(cfg.xmin,Math.min(cfg.xmax,P.invX(mx))), Math.max(cfg.ymin,Math.min(cfg.ymax,P.invY(my)))]; draw(); }
    var dragging = false;
    C.cv.addEventListener("mousedown", function(e){ dragging=true; fromMouse(e); });
    C.cv.addEventListener("mousemove", function(e){ if(dragging) fromMouse(e); });
    window.addEventListener("mouseup", function(){ dragging=false; });
    slider(controls, { label:"b.x", min:-1, max:5, val:b[0], step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ b[0]=v; draw(); } });
    slider(controls, { label:"b.y", min:-1, max:4, val:b[1], step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ b[1]=v; draw(); } });
  });

  /* 8.3 — gradient descent stepper */
  reg("8.3", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 330), controls = controlsIn(wrap);
    var f = function(x){return 0.6*x*x;}, fp = function(x){return 1.2*x;};
    var eta = 0.35, x = 2.3, steps = 0, path = [x];
    function draw() {
      var P = plot(C.ctx, 600, 330, -2.7, 2.7, -0.3, 4.2); P.clear(); P.grid([-2,-1,0,1,2],[0,1,2,3,4]); P.axes();
      P.title("Set a learning rate, then Step — descend, or overshoot and diverge");
      P.fn(f, -2.6, 2.6, PAL.indigo, 2.4);
      for (var i=0;i<path.length-1;i++) P.line(path[i], f(path[i]), path[i+1], f(path[i+1]), PAL.amber, 1.4);
      path.forEach(function(xx,i){ P.dot(xx, f(xx), i===path.length-1?PAL.green:PAL.red, i===path.length-1?4.5:3); });
      P.txt(16, 314, "x = " + x.toFixed(3) + "      step " + steps, PAL.ink2, 13, "left", "600");
      P.txt(410, 314, "η = " + eta.toFixed(2), eta>1.6?PAL.red:PAL.accent, 13, "left", "600");
    }
    slider(controls, { label:"learning rate η", min:0.05, max:1.9, val:eta, step:0.05, fmt:function(v){return v.toFixed(2);}, on:function(v){ eta=v; } });
    button(controls, "Step", function(){ x = x - eta*fp(x); path.push(x); steps++; if(path.length>40) path.shift(); draw(); });
    button(controls, "Reset", function(){ x=2.3; steps=0; path=[x]; draw(); });
    draw();
  });

  /* 10.2 — Gaussian mean & spread */
  reg("10.2", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 330), controls = controlsIn(wrap);
    var mu = 0, sg = 1;
    function draw() {
      var P = plot(C.ctx, 600, 330, -6, 6, -0.02, 0.62); P.clear(); P.axes();
      P.title("Slide μ (center) and σ (width) of the bell curve");
      var pdf = function(x){ return Math.exp(-Math.pow((x-mu)/sg,2)/2)/(sg*Math.sqrt(2*Math.PI)); };
      var pts = [[mu-sg,0]]; for (var x=mu-sg; x<=mu+sg; x+=0.05) pts.push([x, pdf(x)]); pts.push([mu+sg,0]);
      P.fill(pts, "rgba(79,70,229,0.16)");
      P.fn(pdf, -6, 6, PAL.indigo, 2.6);
      P.line(mu, 0, mu, pdf(mu), PAL.red, 1.4, [4,3]);
      P.txt(16, 314, "μ = " + mu.toFixed(1) + "      σ = " + sg.toFixed(1), PAL.ink2, 13, "left", "600");
      P.txt(410, 314, "peak = " + pdf(mu).toFixed(3), PAL.indigo, 13, "left", "600");
    }
    slider(controls, { label:"μ", min:-3, max:3, val:mu, step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ mu=v; draw(); } });
    slider(controls, { label:"σ", min:0.4, max:2.2, val:sg, step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ sg=v; draw(); } });
    draw();
  });

  /* 2.4 — matrix transform of the unit square */
  reg("2.4", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 360), controls = controlsIn(wrap);
    var a = 1.5, b = 0.6, c = 0.3, d = 1.2;
    function draw() {
      var P = plot(C.ctx, 600, 360, -1.6, 3.2, -1.6, 3.2); P.clear(); P.grid([-1,0,1,2,3],[-1,0,1,2,3]); P.axes();
      P.title("Change the matrix entries — watch the unit square transform");
      P.poly([[0,0],[1,0],[1,1],[0,1],[0,0]], PAL.ink3, 1.2);
      var i = [a,c], j = [b,d];
      P.fill([[0,0], i, [i[0]+j[0], i[1]+j[1]], j], "rgba(79,70,229,0.14)");
      P.poly([[0,0], i, [i[0]+j[0], i[1]+j[1]], j, [0,0]], PAL.indigo, 1.8);
      P.arrow(0,0,i[0],i[1], PAL.blue, 2.2); P.arrow(0,0,j[0],j[1], PAL.green, 2.2);
      var det = a*d - b*c;
      P.txt(16, 344, "det = " + det.toFixed(2) + (Math.abs(det)<0.06 ? "    (collapses — no inverse!)" : ""), det<0?PAL.red:PAL.ink2, 13, "left", "600");
    }
    slider(controls, { label:"a", min:-1, max:2.5, val:a, step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ a=v; draw(); } });
    slider(controls, { label:"b", min:-1, max:2.5, val:b, step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ b=v; draw(); } });
    slider(controls, { label:"c", min:-1, max:2.5, val:c, step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ c=v; draw(); } });
    slider(controls, { label:"d", min:-1, max:2.5, val:d, step:0.1, fmt:function(v){return v.toFixed(1);}, on:function(v){ d=v; draw(); } });
    draw();
  });

  /* 12.1 — entropy of a coin */
  reg("12.1", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 330), controls = controlsIn(wrap);
    var p = 0.5, H = function(q){ return (q<=0||q>=1) ? 0 : -(q*Math.log2(q) + (1-q)*Math.log2(1-q)); };
    function draw() {
      var P = plot(C.ctx, 600, 330, -0.02, 1.02, -0.03, 1.12); P.clear(); P.axes();
      P.title("Slide the coin's bias p — watch the uncertainty H(p)");
      P.fn(H, 0.001, 0.999, PAL.indigo, 2.4);
      P.line(p, 0, p, H(p), PAL.red, 1.2, [4,3]); P.dot(p, H(p), PAL.red, 4.5);
      P.txt(16, 314, "p = " + p.toFixed(2), PAL.ink2, 13, "left", "600");
      P.txt(320, 314, "H(p) = " + H(p).toFixed(3) + " bits", PAL.indigo, 13, "left", "600");
    }
    slider(controls, { label:"p", min:0.01, max:0.99, val:p, step:0.01, fmt:function(v){return v.toFixed(2);}, on:function(v){ p=v; draw(); } });
    draw();
  });

  /* 2.2 — tip-to-tail vector addition (draggable) */
  reg("2.2", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 360), controls = controlsIn(wrap);
    var u = [3, 1], v = [1, 2], cfg = { xmin:-1.5, xmax:6, ymin:-1.5, ymax:4.5 }, P;
    function draw() {
      P = plot(C.ctx, 600, 360, cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax);
      P.clear(); P.grid([-1,0,1,2,3,4,5],[-1,0,1,2,3,4]); P.axes();
      P.title("Drag the green arrow's tip — the sum follows, tip to tail");
      var s = [u[0]+v[0], u[1]+v[1]];
      P.line(0,0,v[0],v[1],PAL.green,1.2,[3,3]);
      P.line(v[0],v[1],s[0],s[1],PAL.blue,1.2,[3,3]);
      P.arrow(0,0,u[0],u[1],PAL.blue,2.6);
      P.arrow(u[0],u[1],s[0],s[1],PAL.green,2.6);
      P.arrow(0,0,s[0],s[1],PAL.indigo,3);
      P.dot(s[0],s[1],PAL.indigo,5.5);
      P.txt(16,344,"u = ("+u[0]+", "+u[1]+")",PAL.blue,13,"left","600");
      P.txt(180,344,"v = ("+v[0].toFixed(1)+", "+v[1].toFixed(1)+")",PAL.green,13,"left","600");
      P.txt(370,344,"u + v = ("+s[0].toFixed(1)+", "+s[1].toFixed(1)+")",PAL.indigo,13,"left","700");
    }
    draw();
    function fromMouse(e){ var r=C.cv.getBoundingClientRect();
      var sx=Math.max(cfg.xmin,Math.min(cfg.xmax,P.invX(e.clientX-r.left)));
      var sy=Math.max(cfg.ymin,Math.min(cfg.ymax,P.invY(e.clientY-r.top)));
      v=[sx-u[0], sy-u[1]]; draw(); }
    var dragging=false;
    C.cv.addEventListener("mousedown",function(e){dragging=true;fromMouse(e);});
    C.cv.addEventListener("mousemove",function(e){if(dragging)fromMouse(e);});
    window.addEventListener("mouseup",function(){dragging=false;});
    slider(controls,{label:"v.x",min:-2,max:4,val:v[0],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){v[0]=x;draw();}});
    slider(controls,{label:"v.y",min:-2,max:4,val:v[1],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){v[1]=x;draw();}});
  });

  /* 5.1 — rotate v until Av lines up: that's an eigenvector */
  reg("5.1", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var A = [[2,1],[1,2]], th = 20;
    function draw() {
      var P = plot(C.ctx, 600, 340, -4, 4, -4, 4);
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("Rotate v — when Av points the SAME way, v is an eigenvector");
      P.line(-3.7,-3.7,3.7,3.7,PAL.grid,2); P.line(-3.7,3.7,3.7,-3.7,PAL.grid,2);
      var t = th*Math.PI/180, r = 1.55;
      var v = [r*Math.cos(t), r*Math.sin(t)];
      var Av = [A[0][0]*v[0]+A[0][1]*v[1], A[1][0]*v[0]+A[1][1]*v[1]];
      P.arrow(0,0,Av[0],Av[1],PAL.blue,2.4);
      P.arrow(0,0,v[0],v[1],PAL.indigo,3.2);
      var dp = v[0]*Av[0]+v[1]*Av[1], lv = Math.hypot(v[0],v[1]), la = Math.hypot(Av[0],Av[1]);
      var ang = Math.acos(Math.max(-1,Math.min(1,dp/(lv*la))))*180/Math.PI;
      var lam = (la/lv)*(dp<0?-1:1), ok = ang < 2.5;
      P.txt(16,324,"angle(v, Av) = "+ang.toFixed(1)+"°", ok?PAL.green:PAL.ink2,13,"left","600");
      P.txt(250,324, ok ? ("EIGENVECTOR — Av = λv,  λ = "+lam.toFixed(2)) : "not an eigenvector — Av swings away",
            ok?PAL.green:PAL.ink3,13,"left",ok?"700":"600");
    }
    slider(controls,{label:"angle of v",min:0,max:360,val:th,step:1,fmt:function(x){return x+"°";},on:function(x){th=x;draw();}});
    draw();
  });

  /* 9.3 — Bayes: drag the base rate and watch false alarms swamp the positives */
  reg("9.3", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var prior = 8, sens = 90, fpr = 15;
    function draw() {
      var ctx = C.ctx, P = plot(C.ctx, 600, 340, 0, 1, 0, 1);
      P.clear(); P.title("Slide the base rate — see how many 'positives' are false alarms");
      var x0=54, y0=44, W=330, H=222;
      var p=prior/100, se=sens/100, fp=fpr/100, dw=p*W;
      ctx.fillStyle=PAL.red;      ctx.fillRect(x0, y0, dw, se*H);
      ctx.fillStyle=PAL.redF;     ctx.fillRect(x0, y0+se*H, dw, (1-se)*H);
      ctx.fillStyle=PAL.amber;    ctx.fillRect(x0+dw, y0, W-dw, fp*H);
      ctx.fillStyle=PAL.neutralF; ctx.fillRect(x0+dw, y0+fp*H, W-dw, (1-fp)*H);
      ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1; ctx.strokeRect(x0,y0,W,H);
      ctx.beginPath(); ctx.moveTo(x0+dw,y0); ctx.lineTo(x0+dw,y0+H); ctx.stroke();
      var tp=p*se, fa=(1-p)*fp, post=tp/(tp+fa)*100;
      P.txt(x0, y0+H+18, "sick "+prior+"%", PAL.red, 11, "left", "600");
      P.txt(x0+dw+8, y0+H+18, "healthy "+(100-prior).toFixed(1)+"%", PAL.ink3, 11, "left");
      var tx=x0+W+26;
      P.txt(tx, y0+16, "true positives", PAL.red, 12, "left", "600");
      P.txt(tx, y0+34, (tp*100).toFixed(1)+"% of everyone", PAL.ink2, 11.5, "left");
      P.txt(tx, y0+62, "false alarms", PAL.amber, 12, "left", "600");
      P.txt(tx, y0+80, (fa*100).toFixed(1)+"% of everyone", PAL.ink2, 11.5, "left");
      P.txt(tx, y0+118, "P(sick | positive)", PAL.ink, 12.5, "left", "700");
      P.txt(tx, y0+148, post.toFixed(0)+"%", post<50?PAL.red:PAL.green, 22, "left", "700");
    }
    slider(controls,{label:"base rate",min:0.5,max:50,val:prior,step:.5,fmt:function(x){return x+"%";},on:function(x){prior=x;draw();}});
    slider(controls,{label:"sensitivity",min:50,max:99,val:sens,step:1,fmt:function(x){return x+"%";},on:function(x){sens=x;draw();}});
    slider(controls,{label:"false-pos rate",min:1,max:30,val:fpr,step:1,fmt:function(x){return x+"%";},on:function(x){fpr=x;draw();}});
    draw();
  });

  /* 7.1 — Riemann sum converging to the integral */
  reg("7.1", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var n = 6, a = 0.5, b = 3.5;
    var f = function(x){ return 0.5 + 0.35*x + 0.18*x*x; };
    var Fn = function(x){ return 0.5*x + 0.35*x*x/2 + 0.18*x*x*x/3; };
    var exact = Fn(b) - Fn(a);
    function draw() {
      var P = plot(C.ctx, 600, 340, -0.2, 4.4, -0.2, 5.0), ctx = C.ctx;
      P.clear(); P.grid([0,1,2,3,4],[0,1,2,3,4]); P.axes();
      P.title("More rectangles → the sum closes in on the exact area");
      var w=(b-a)/n, sum=0;
      for (var i=0;i<n;i++){
        var xm=a+(i+.5)*w, h=f(xm); sum+=h*w;
        var X=P.X(a+i*w), Y=P.Y(h), Wp=P.X(a+w)-P.X(a), Hp=P.Y(0)-P.Y(h);
        ctx.fillStyle="rgba(42,111,151,0.18)"; ctx.fillRect(X,Y,Wp,Hp);
        ctx.strokeStyle=PAL.blue; ctx.lineWidth=1; ctx.strokeRect(X,Y,Wp,Hp);
      }
      P.fn(f,-0.2,4.2,PAL.indigo,2.6);
      P.txt(16,324,"n = "+n+"    sum = "+sum.toFixed(4),PAL.blue,13,"left","600");
      P.txt(310,324,"exact = "+exact.toFixed(4)+"    error = "+Math.abs(sum-exact).toFixed(4),PAL.ink2,13,"left","600");
    }
    slider(controls,{label:"rectangles n",min:1,max:60,val:n,step:1,fmt:function(x){return ""+x;},on:function(x){n=x;draw();}});
    draw();
  });

  /* 10.6 — Central Limit Theorem, simulated live */
  reg("10.6", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var n = 1, M = 3000;
    function draw() {
      var ctx = C.ctx, P = plot(C.ctx, 600, 340, 0, 1, 0, 1);
      P.clear(); P.title("Average n uniform draws — the bell appears as n grows");
      var bins=44, counts=new Array(bins); for (var i=0;i<bins;i++) counts[i]=0;
      for (var k=0;k<M;k++){ var s=0; for (var j=0;j<n;j++) s+=Math.random();
        var v=s/n, bi=Math.min(bins-1, Math.floor(v*bins)); counts[bi]++; }
      var mx=1; for (i=0;i<bins;i++) if (counts[i]>mx) mx=counts[i];
      var x0=54, y0=44, W=496, H=224;
      for (i=0;i<bins;i++){ var h=counts[i]/mx*H;
        ctx.fillStyle=PAL.indigo; ctx.globalAlpha=.78;
        ctx.fillRect(x0+i*(W/bins), y0+H-h, (W/bins)-1.5, h); }
      ctx.globalAlpha=1;
      ctx.strokeStyle=PAL.axis; ctx.lineWidth=1.2;
      ctx.beginPath(); ctx.moveTo(x0,y0+H); ctx.lineTo(x0+W,y0+H); ctx.stroke();
      var sd=1/Math.sqrt(12*n);
      P.txt(16,324,"n = "+n,PAL.ink2,13,"left","600");
      P.txt(110,324,"spread (sd) = "+sd.toFixed(4),PAL.indigo,13,"left","600");
      P.txt(330,324, n===1 ? "flat — nothing averaged yet" : "bell-shaped, and tightening", PAL.ink3,12.5,"left");
    }
    slider(controls,{label:"sample size n",min:1,max:50,val:n,step:1,fmt:function(x){return ""+x;},on:function(x){n=x;draw();}});
    button(controls,"Resample",function(){ draw(); });
    draw();
  });

  /* 8.2 — convexity: does the chord ever dip below? */
  reg("8.2", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var ax=-1.8, bx=2.2, convex=true;
    function f(x){ return convex ? 0.55*x*x + 0.2 : 0.12*x*x*x*x - 0.75*x*x + 1.9; }
    function draw() {
      var P = plot(C.ctx, 600, 340, -2.7, 2.7, -0.3, 4.4);
      P.clear(); P.grid([-2,-1,0,1,2],[0,1,2,3,4]); P.axes();
      P.title(convex ? "Convex — the chord never dips below the curve"
                     : "Not convex — the chord cuts underneath the curve");
      P.fn(f,-2.6,2.6,PAL.indigo,2.6);
      var A=[ax,f(ax)], B=[bx,f(bx)];
      var chord=function(x){ return A[1] + (B[1]-A[1])*(x-A[0])/(B[0]-A[0]); };
      var N=70, below=false, up=[], dn=[];
      for (var i=0;i<=N;i++){ var x=A[0]+(B[0]-A[0])*i/N;
        up.push([x,chord(x)]); if (chord(x) < f(x)-1e-9) below=true; }
      for (i=N;i>=0;i--){ var x2=A[0]+(B[0]-A[0])*i/N; dn.push([x2,f(x2)]); }
      P.fill(up.concat(dn), below ? "rgba(192,57,43,0.18)" : "rgba(185,114,42,0.18)");
      P.line(A[0],A[1],B[0],B[1], below?PAL.red:PAL.amber, 2.2);
      P.dot(A[0],A[1],PAL.ink,4.5); P.dot(B[0],B[1],PAL.ink,4.5);
      P.txt(16,324, below ? "chord dips BELOW → not convex" : "chord stays above → convex ✓",
            below?PAL.red:PAL.green, 13, "left", "700");
      P.txt(360,324, "press Toggle to try the other function", PAL.ink3, 11.5, "left");
    }
    slider(controls,{label:"A",min:-2.5,max:-0.2,val:ax,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){ax=x;draw();}});
    slider(controls,{label:"B",min:0.2,max:2.5,val:bx,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){bx=x;draw();}});
    button(controls,"Toggle function",function(){ convex=!convex; draw(); });
    draw();
  });

  /* 6.3 — the gradient always meets the contour at 90° */
  reg("6.3", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var th = 35;
    function draw() {
      var P = plot(C.ctx, 600, 340, -2.8, 2.8, -2.8, 2.8);
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      P.title("Move the point — the gradient always crosses the contour at 90°");
      var cs=[0.6,1.6,3.2], cols=[PAL.blueF,PAL.blue,PAL.indigo];
      for (var ci=0;ci<3;ci++){ var c=cs[ci], pts=[];
        for (var t2=0;t2<=140;t2++){ var a=t2/140*2*Math.PI;
          pts.push([Math.sqrt(c)*Math.cos(a), Math.sqrt(c/3)*Math.sin(a)]); }
        P.poly(pts, cols[ci], 1.9); }
      var t=th*Math.PI/180, c0=1.6;
      var px=Math.sqrt(c0)*Math.cos(t), py=Math.sqrt(c0/3)*Math.sin(t);
      var g=[2*px,6*py], gl=Math.hypot(g[0],g[1]);
      var gx=g[0]/gl*1.25, gy=g[1]/gl*1.25;
      var tx=-Math.sqrt(c0)*Math.sin(t), ty=Math.sqrt(c0/3)*Math.cos(t), tl=Math.hypot(tx,ty);
      P.line(px-tx/tl*0.95, py-ty/tl*0.95, px+tx/tl*0.95, py+ty/tl*0.95, PAL.green, 1.8, [4,3]);
      P.arrow(px,py,px+gx,py+gy,PAL.red,2.6);
      P.dot(px,py,PAL.red,4.5);
      var dp = gx*(tx/tl) + gy*(ty/tl);
      P.txt(16,324,"∇f · tangent = "+dp.toFixed(3)+"  → perpendicular",PAL.green,13,"left","600");
      P.txt(400,324,"f = x² + 3y²",PAL.ink3,12,"left");
    }
    slider(controls,{label:"point",min:0,max:360,val:th,step:1,fmt:function(x){return x+"°";},on:function(x){th=x;draw();}});
    draw();
  });

  /* 4.1 — the Lp unit ball morphing with p */
  reg("4.1", function (mount) {
    var wrap = card(mount), C = canvasIn(wrap, 600, 340), controls = controlsIn(wrap);
    var p = 2;
    function draw() {
      var P = plot(C.ctx, 600, 340, -1.6, 1.6, -1.6, 1.6, {l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-1,0,1],[-1,0,1]); P.axes();
      P.title("Slide p — the “circle of radius 1” morphs: diamond → circle → square");
      var pts=[];
      for (var i=0;i<=260;i++){ var a=i/260*2*Math.PI, ca=Math.cos(a), sa=Math.sin(a);
        var r = (p>=20) ? 1/Math.max(Math.abs(ca),Math.abs(sa))
                        : Math.pow(Math.pow(Math.abs(ca),p)+Math.pow(Math.abs(sa),p), -1/p);
        pts.push([r*ca, r*sa]); }
      P.poly(pts, PAL.indigo, 2.7);
      var name = (p<1.06) ? "L1 — taxicab (diamond)"
               : (Math.abs(p-2)<0.06 ? "L2 — Euclidean (circle)"
               : (p>=20 ? "L∞ — max (square)" : "Lp"));
      P.txt(16,324,"p = "+(p>=20?"∞":p.toFixed(1)),PAL.ink2,13,"left","600");
      P.txt(120,324,name,PAL.indigo,13,"left","700");
      if (p < 1.5) P.txt(400,324,"corners on the axes → sparsity",PAL.green,12,"left","600");
    }
    slider(controls,{label:"p",min:1,max:24,val:p,step:.1,fmt:function(x){return x>=20?"∞":x.toFixed(1);},on:function(x){p=x;draw();}});
    draw();
  });


  /* 1.1 — exp and log are mirrors across y = x */
  reg("1.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var b=2, x0=1.4;
    function draw(){
      var P=plot(C.ctx,600,340,-2.6,4.4,-2.6,4.4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-2,-1,0,1,2,3,4],[-2,-1,0,1,2,3,4]); P.axes();
      P.title("Reflect a point of bˣ across y = x and you land on log");
      P.line(-2.6,-2.6,4.4,4.4,PAL.axis,1.3,[5,4]);
      P.fn(function(x){return Math.pow(b,x);},-2.6,4.4,PAL.indigo,2.5);
      P.fn(function(x){return x>0?Math.log(x)/Math.log(b):NaN;},0.02,4.4,PAL.green,2.5);
      var y0=Math.pow(b,x0);
      P.line(x0,y0,y0,x0,PAL.amber,1.4,[3,3]);
      P.dot(x0,y0,PAL.indigo,4.5); P.dot(y0,x0,PAL.green,4.5);
      P.txt(16,324,"b = "+b.toFixed(1)+"    "+b.toFixed(1)+"^"+x0.toFixed(1)+" = "+y0.toFixed(2),PAL.indigo,13,"left","600");
      P.txt(330,324,"log_"+b.toFixed(1)+"("+y0.toFixed(2)+") = "+x0.toFixed(1),PAL.green,13,"left","600");
    }
    slider(controls,{label:"base b",min:1.5,max:4,val:b,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){b=v;draw();}});
    slider(controls,{label:"x",min:-1.5,max:2,val:x0,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){x0=v;draw();}});
    draw();
  });

  /* 1.4 — how fast each complexity class explodes */
  reg("1.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var n=64;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Slide n — see how many operations each class actually costs");
      var rows=[["O(1)",1,PAL.green],["O(log n)",Math.log2(n),PAL.blue],["O(n)",n,PAL.amber],
                ["O(n log n)",n*Math.log2(n),PAL.violet||PAL.indigo],["O(n²)",n*n,PAL.red]];
      var mx=rows[4][1], x0=132, y0=52, W=396, bh=30, gap=14;
      for (var i=0;i<rows.length;i++){
        var y=y0+i*(bh+gap), w=Math.max(2, Math.log10(1+rows[i][1])/Math.log10(1+mx)*W);
        ctx.fillStyle=rows[i][2]; ctx.globalAlpha=.85; ctx.fillRect(x0,y,w,bh); ctx.globalAlpha=1;
        P.txt(x0-10,y+bh/2+1,rows[i][0],rows[i][2],12.5,"right","700");
        var v=rows[i][1];
        P.txt(x0+w+10,y+bh/2+1, v>=1000? Math.round(v).toLocaleString() : v.toFixed(v<10?1:0), PAL.ink2,12.5,"left","600");
      }
      P.txt(16,324,"n = "+n,PAL.ink,13,"left","700");
      P.txt(120,324,"n² is "+Math.round(n*n/n)+"× the work of n — that gap is why attention is expensive",PAL.ink3,12,"left");
    }
    slider(controls,{label:"n",min:2,max:512,val:n,step:1,fmt:function(v){return ""+v;},on:function(v){n=v;draw();}});
    draw();
  });

  /* 3.4 — least squares as projection (drag b) */
  reg("3.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,360), controls=controlsIn(wrap);
    var a=[3,1], b=[1,2.5], cfg={xmin:-0.5,xmax:4.6,ymin:-0.5,ymax:3.6}, P;
    function draw(){
      P=plot(C.ctx,600,360,cfg.xmin,cfg.xmax,cfg.ymin,cfg.ymax);
      P.clear(); P.grid([0,1,2,3,4],[0,1,2,3]); P.axes();
      P.title("Drag b — the projection is the closest point, error always ⊥");
      P.line(0,0,4.5,4.5/3,PAL.blue,2);
      var t=(a[0]*b[0]+a[1]*b[1])/(a[0]*a[0]+a[1]*a[1]), p=[t*a[0],t*a[1]];
      P.line(b[0],b[1],p[0],p[1],PAL.amber,1.8,[4,3]);
      P.arrow(0,0,b[0],b[1],PAL.green,2.5);
      P.dot(p[0],p[1],PAL.indigo,5);
      var err=Math.hypot(b[0]-p[0],b[1]-p[1]);
      var dp=(b[0]-p[0])*a[0]+(b[1]-p[1])*a[1];
      P.txt(16,344,"‖error‖ = "+err.toFixed(3),PAL.amber,13,"left","600");
      P.txt(200,344,"error · a = "+dp.toFixed(3)+"  → perpendicular",PAL.green,13,"left","600");
    }
    draw();
    function fm(e){var r=C.cv.getBoundingClientRect();
      b=[Math.max(cfg.xmin,Math.min(cfg.xmax,P.invX(e.clientX-r.left))),Math.max(cfg.ymin,Math.min(cfg.ymax,P.invY(e.clientY-r.top)))];draw();}
    var dg=false;
    C.cv.addEventListener("mousedown",function(e){dg=true;fm(e);});
    C.cv.addEventListener("mousemove",function(e){if(dg)fm(e);});
    window.addEventListener("mouseup",function(){dg=false;});
    slider(controls,{label:"b.x",min:-0.5,max:4.5,val:b[0],step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){b[0]=v;draw();}});
    slider(controls,{label:"b.y",min:-0.5,max:3.5,val:b[1],step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){b[1]=v;draw();}});
  });

  /* 5.4 — SVD: circle in, ellipse out */
  reg("5.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var a=1.4,bb=0.7,c=0.2,d=1.1;
    function svd2(A){ var a1=A[0][0],b1=A[0][1],c1=A[1][0],d1=A[1][1];
      var E=(a1+d1)/2,F=(a1-d1)/2,G=(c1+b1)/2,H=(c1-b1)/2;
      var Q=Math.hypot(E,H),R=Math.hypot(F,G);
      return {s1:Q+R, s2:Math.abs(Q-R), phi:(Math.atan2(H,E)+Math.atan2(G,F))/2}; }
    function draw(){
      var P=plot(C.ctx,600,340,-2.6,2.6,-2.6,2.6,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      P.title("A stretches the unit circle into an ellipse — σ₁,σ₂ are the half-axes");
      var circ=[],ell=[];
      for(var i=0;i<=140;i++){var t=i/140*2*Math.PI,cx=Math.cos(t),cy=Math.sin(t);
        circ.push([cx,cy]); ell.push([a*cx+bb*cy, c*cx+d*cy]);}
      P.poly(circ,PAL.ink3,1.5,[4,3]); P.poly(ell,PAL.indigo,2.5);
      var S=svd2([[a,bb],[c,d]]);
      var u1=[Math.cos(S.phi),Math.sin(S.phi)], u2=[-u1[1],u1[0]];
      P.arrow(0,0,u1[0]*S.s1,u1[1]*S.s1,PAL.red,2.4);
      P.arrow(0,0,u2[0]*S.s2,u2[1]*S.s2,PAL.green,2.2);
      P.txt(16,324,"σ₁ = "+S.s1.toFixed(2)+"   σ₂ = "+S.s2.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(300,324,"condition σ₁/σ₂ = "+(S.s2>1e-6?(S.s1/S.s2).toFixed(1):"∞ (singular)"),
            (S.s2>1e-6&&S.s1/S.s2<8)?PAL.green:PAL.red,13,"left","600");
    }
    slider(controls,{label:"a",min:-2,max:2,val:a,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){a=v;draw();}});
    slider(controls,{label:"b",min:-2,max:2,val:bb,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){bb=v;draw();}});
    slider(controls,{label:"c",min:-2,max:2,val:c,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){c=v;draw();}});
    slider(controls,{label:"d",min:-2,max:2,val:d,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){d=v;draw();}});
    draw();
  });

  /* 6.5 — Taylor: add terms, watch the fit widen */
  reg("6.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var A=0.9, K=2;
    function taylor(x){ var s=0, dv=[Math.cos(A),-Math.sin(A),-Math.cos(A),Math.sin(A)], f=1;
      for(var j=0;j<=K;j++){ if(j>0) f*=j; s+= dv[j%4]/f*Math.pow(x-A,j); } return s; }
    function draw(){
      var P=plot(C.ctx,600,340,-0.4,6.6,-2.0,2.0);
      P.clear(); P.grid([0,1,2,3,4,5,6],[-1,0,1]); P.axes();
      P.title("Add Taylor terms — the fit hugs cos further from the point a");
      P.fn(Math.cos,-0.4,6.6,PAL.indigo,2.5);
      P.fn(taylor,-0.4,6.6,PAL.red,2.2);
      P.dot(A,Math.cos(A),PAL.ink,5);
      var e=0,n=0; for(var x=A-1.6;x<=A+1.6;x+=0.05){e+=Math.abs(taylor(x)-Math.cos(x));n++;}
      P.txt(16,324,"a = "+A.toFixed(1)+"    order = "+K,PAL.ink2,13,"left","600");
      P.txt(260,324,"avg |error| near a = "+(e/n).toFixed(4),PAL.red,13,"left","600");
    }
    slider(controls,{label:"point a",min:0,max:6,val:A,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){A=v;draw();}});
    slider(controls,{label:"order",min:0,max:8,val:K,step:1,fmt:function(v){return ""+v;},on:function(v){K=v;draw();}});
    draw();
  });

  /* 10.3 — the Beta distribution's shape */
  reg("10.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var A=2,B=5;
    function lgamma(z){ var g=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
      if(z<0.5) return Math.log(Math.PI/Math.sin(Math.PI*z))-lgamma(1-z);
      z-=1; var x=0.99999999999980993; for(var i=0;i<8;i++) x+=g[i]/(z+i+1);
      var t=z+7.5; return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(x); }
    function bpdf(x){ if(x<=0||x>=1) return 0;
      return Math.exp((A-1)*Math.log(x)+(B-1)*Math.log(1-x)+lgamma(A+B)-lgamma(A)-lgamma(B)); }
    function draw(){
      var P=plot(C.ctx,600,340,-0.02,1.02,-0.1,4.2);
      P.clear(); P.grid([0,0.25,0.5,0.75,1],[0,1,2,3,4]); P.axes(); 
      P.title("Beta(a,b) — two counts bend the curve to any belief on [0,1]");
      P.fn(function(x){return Math.min(bpdf(x),4.15);},0.001,0.999,PAL.indigo,2.6);
      var mean=A/(A+B), mode=(A>1&&B>1)?(A-1)/(A+B-2):NaN;
      P.line(mean,0,mean,Math.min(bpdf(mean),4.1),PAL.red,1.4,[4,3]);
      P.txt(16,324,"a = "+A.toFixed(1)+"   b = "+B.toFixed(1),PAL.ink2,13,"left","600");
      P.txt(220,324,"mean = "+mean.toFixed(3),PAL.red,13,"left","600");
      P.txt(400,324, isNaN(mode)?"":"peak = "+mode.toFixed(3),PAL.indigo,13,"left","600");
    }
    slider(controls,{label:"a",min:0.5,max:8,val:A,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){A=v;draw();}});
    slider(controls,{label:"b",min:0.5,max:8,val:B,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){B=v;draw();}});
    draw();
  });

  /* 11.2 — likelihood peaks at the observed frequency */
  reg("11.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var n=10,k=7;
    function draw(){
      var P=plot(C.ctx,600,340,-0.02,1.02,-0.06,1.15);
      P.clear(); P.grid([0,0.25,0.5,0.75,1],[0,0.5,1]); P.axes();
      P.title("Likelihood of the data for each θ — the peak is the MLE");
      var L=function(t){ return Math.pow(t,k)*Math.pow(1-t,n-k); };
      var mx=0; for(var x=0.001;x<1;x+=0.001) mx=Math.max(mx,L(x));
      P.fn(function(t){return L(t)/(mx||1);},0.001,0.999,PAL.indigo,2.6);
      var th=k/n;
      P.line(th,0,th,1,PAL.red,1.6,[4,3]); P.dot(th,1,PAL.red,5);
      P.txt(16,324,k+" heads in "+n+" flips",PAL.ink2,13,"left","600");
      P.txt(240,324,"θ̂ = "+k+"/"+n+" = "+th.toFixed(2),PAL.red,13,"left","700");
      P.txt(430,324,"= the observed frequency",PAL.ink3,12,"left");
    }
    slider(controls,{label:"flips n",min:1,max:40,val:n,step:1,fmt:function(v){return ""+v;},on:function(v){n=v;if(k>n)k=n;draw();}});
    slider(controls,{label:"heads k",min:0,max:40,val:k,step:1,fmt:function(v){return ""+v;},on:function(v){k=Math.min(v,n);draw();}});
    draw();
  });

  /* 11.4 — the bias-variance U */
  reg("11.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var cx=2.3;
    var bias2=function(c){return 5.5/(c+0.7);}, vari=function(c){return 0.22*Math.pow(c,1.6);};
    var tot=function(c){return bias2(c)+vari(c);};
    function draw(){
      var P=plot(C.ctx,600,340,-0.1,5.1,-0.2,5.4);
      P.clear(); P.grid([0,1,2,3,4,5],[0,1,2,3,4,5]); P.axes();
      P.title("Slide model complexity — total error is a U, not a slope");
      P.fn(function(c){return Math.min(bias2(c),5.3);},0.25,5,PAL.blue,2.2);
      P.fn(function(c){return Math.min(vari(c),5.3);},0.25,5,PAL.green,2.2);
      P.fn(function(c){return Math.min(tot(c),5.3);},0.25,5,PAL.red,2.8);
      var best=0.3,bv=1e9; for(var c=0.3;c<=4.9;c+=0.01) if(tot(c)<bv){bv=tot(c);best=c;}
      P.line(best,0,best,tot(best),PAL.ink3,1,[3,3]);
      P.line(cx,0,cx,Math.min(tot(cx),5.3),PAL.ink,1.4);
      P.dot(cx,Math.min(tot(cx),5.3),PAL.red,5);
      P.txt(16,324,"bias² = "+bias2(cx).toFixed(2)+"   variance = "+vari(cx).toFixed(2),PAL.ink2,13,"left","600");
      P.txt(330,324,"total = "+tot(cx).toFixed(2)+(Math.abs(cx-best)<0.25?"  ← sweet spot":""),
            Math.abs(cx-best)<0.25?PAL.green:PAL.red,13,"left","700");
    }
    slider(controls,{label:"complexity",min:0.3,max:5,val:cx,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){cx=v;draw();}});
    draw();
  });

  /* 12.2 — KL shrinks to zero as the model matches the truth */
  reg("12.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var t=0;
    var p=[0.5,0.3,0.15,0.05];
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Slide the model toward the truth — watch KL fall to zero");
      var q=[],u=0.25,i;
      for(i=0;i<4;i++) q.push((1-t)*u + t*p[i]);
      var H=0,CE=0; for(i=0;i<4;i++){ H-=p[i]*Math.log2(p[i]); CE-=p[i]*Math.log2(q[i]); }
      var KL=CE-H;
      var x0=70,y0=56,W=340,H0=190;
      for(i=0;i<4;i++){
        var bw=W/4-18, bx=x0+i*(W/4);
        ctx.fillStyle=PAL.blueF; ctx.fillRect(bx,y0+H0-p[i]*H0,bw*0.46,p[i]*H0);
        ctx.strokeStyle=PAL.blue; ctx.lineWidth=1; ctx.strokeRect(bx,y0+H0-p[i]*H0,bw*0.46,p[i]*H0);
        ctx.fillStyle=PAL.amberF; ctx.fillRect(bx+bw*0.52,y0+H0-q[i]*H0,bw*0.46,q[i]*H0);
        ctx.strokeStyle=PAL.amber; ctx.strokeRect(bx+bw*0.52,y0+H0-q[i]*H0,bw*0.46,q[i]*H0);
        P.txt(bx+bw*0.5,y0+H0+16,"class "+i,PAL.ink3,10.5,"center");
      }
      ctx.strokeStyle=PAL.axis; ctx.beginPath(); ctx.moveTo(x0-8,y0+H0); ctx.lineTo(x0+W,y0+H0); ctx.stroke();
      var tx=x0+W+34;
      P.txt(tx,y0+16,"true p",PAL.blue,12,"left","700");
      P.txt(tx,y0+36,"model q",PAL.amber,12,"left","700");
      P.txt(tx,y0+72,"H(p)  = "+H.toFixed(3),PAL.ink2,12,"left");
      P.txt(tx,y0+94,"H(p,q) = "+CE.toFixed(3),PAL.ink2,12,"left");
      P.txt(tx,y0+126,"KL(p‖q)",PAL.ink,12.5,"left","700");
      P.txt(tx,y0+150,KL.toFixed(3)+" bits",KL<0.02?PAL.green:PAL.red,17,"left","700");
    }
    slider(controls,{label:"model → truth",min:0,max:1,val:t,step:.02,fmt:function(v){return (v*100).toFixed(0)+"%";},on:function(v){t=v;draw();}});
    draw();
  });

  /* 13.5 — finite differences: too big truncates, too small round-offs */
  reg("13.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var e=-6;
    var f=function(x){return Math.sin(x)+0.3*x*x;}, fp=function(x){return Math.cos(x)+0.6*x;};
    function err(h){ var x=1.2; return Math.abs((f(x+h)-f(x-h))/(2*h) - fp(x)); }
    function draw(){
      var P=plot(C.ctx,600,340,-12,0.5,-12,1);
      P.clear(); P.grid([-12,-9,-6,-3,0],[-12,-9,-6,-3,0]); P.axes(); 
      P.title("Step size h: too big = truncation, too small = round-off");
      var pts=[]; for(var k=-12;k<=0;k+=0.1){ var E=err(Math.pow(10,k)); pts.push([k, Math.log10(Math.max(E,1e-13))]); }
      P.poly(pts,PAL.indigo,2.5);
      var h=Math.pow(10,e), E=err(h);
      P.dot(e,Math.log10(Math.max(E,1e-13)),PAL.red,5);
      P.line(e,-12,e,Math.log10(Math.max(E,1e-13)),PAL.red,1.2,[3,3]);
      P.txt(16,324,"h = 1e"+e.toFixed(0)+"    error = "+E.toExponential(2),PAL.ink2,13,"left","600");
      P.txt(330,324, (e>-3?"h too large — truncation":(e<-9?"h too small — round-off noise":"good zone ✓")),
            (e<=-3&&e>=-9)?PAL.green:PAL.red,13,"left","700");
      P.txt(16,58,"log₁₀ error",PAL.ink3,11,"left");
      P.txt(520,300,"log₁₀ h",PAL.ink3,11,"left");
    }
    slider(controls,{label:"log₁₀ h",min:-12,max:0,val:e,step:.5,fmt:function(v){return "1e"+v.toFixed(0);},on:function(v){e=v;draw();}});
    draw();
  });


  function _erf(x){ var s=x<0?-1:1; x=Math.abs(x);
    var a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;
    var t=1/(1+p*x); return s*(1-(((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t)*Math.exp(-x*x)); }
  function _ncdf(z){ return .5*(1+_erf(z/Math.SQRT2)); }
  function _randn(){ var u=0,v=0; while(!u)u=Math.random(); while(!v)v=Math.random();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

  /* 2.1 — a vector is its components */
  reg("2.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,360), controls=controlsIn(wrap);
    var v=[3,2], cfg={xmin:-0.6,xmax:4.6,ymin:-0.6,ymax:3.6}, P;
    function draw(){
      P=plot(C.ctx,600,360,cfg.xmin,cfg.xmax,cfg.ymin,cfg.ymax);
      P.clear(); P.grid([0,1,2,3,4],[0,1,2,3]); P.axes();
      P.title("Drag the arrow — the components and length follow");
      P.line(0,0,v[0],0,PAL.amber,1.6,[4,3]); P.line(v[0],0,v[0],v[1],PAL.amber,1.6,[4,3]);
      P.arrow(0,0,v[0],v[1],PAL.indigo,3); P.dot(v[0],v[1],PAL.indigo,5.5);
      P.txt(P.X(v[0]/2),P.Y(0)+18,"Δx = "+v[0].toFixed(1),PAL.amber,12,"center","600");
      P.txt(P.X(v[0])+10,P.Y(v[1]/2),"Δy = "+v[1].toFixed(1),PAL.amber,12,"left","600");
      var L=Math.hypot(v[0],v[1]);
      P.txt(16,344,"v = ("+v[0].toFixed(1)+", "+v[1].toFixed(1)+")",PAL.indigo,13,"left","700");
      P.txt(230,344,"‖v‖ = √("+v[0].toFixed(1)+"² + "+v[1].toFixed(1)+"²) = "+L.toFixed(3),PAL.ink2,13,"left","600");
    }
    draw();
    function fm(e){var r=C.cv.getBoundingClientRect();
      v=[Math.max(cfg.xmin,Math.min(cfg.xmax,P.invX(e.clientX-r.left))),Math.max(cfg.ymin,Math.min(cfg.ymax,P.invY(e.clientY-r.top)))];draw();}
    var dg=false;
    C.cv.addEventListener("mousedown",function(e){dg=true;fm(e);});
    C.cv.addEventListener("mousemove",function(e){if(dg)fm(e);});
    window.addEventListener("mouseup",function(){dg=false;});
    slider(controls,{label:"Δx",min:-0.5,max:4.5,val:v[0],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){v[0]=x;draw();}});
    slider(controls,{label:"Δy",min:-0.5,max:3.5,val:v[1],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){v[1]=x;draw();}});
  });

  /* 3.1 — one solution, none, or infinitely many */
  reg("3.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var m=1, c=-1;
    function draw(){
      var P=plot(C.ctx,600,340,-1,5,-2,4);
      P.clear(); P.grid([0,1,2,3,4],[-1,0,1,2,3]); P.axes();
      P.title("Two lines: they cross once, never, or lie on top of each other");
      P.fn(function(x){return 3-x;},-1,5,PAL.blue,2.4);
      P.fn(function(x){return m*x+c;},-1,5,PAL.green,2.4);
      var msg, col;
      if (Math.abs(m+1) > 1e-9){
        var xs=(3-c)/(m+1), ys=3-xs;
        P.dot(xs,ys,PAL.indigo,5.5);
        msg="one crossing → unique solution  ("+xs.toFixed(2)+", "+ys.toFixed(2)+")"; col=PAL.indigo;
      } else if (Math.abs(c-3) > 1e-9){ msg="parallel → NO solution (inconsistent)"; col=PAL.red; }
      else { msg="same line → INFINITELY many solutions"; col=PAL.amber; }
      P.txt(16,324,"x + y = 3    and    y = "+m.toFixed(1)+"x + "+c.toFixed(1),PAL.ink2,13,"left","600");
      P.txt(300,324,msg,col,13,"left","700");
    }
    slider(controls,{label:"slope",min:-2,max:2,val:m,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){m=x;draw();}});
    slider(controls,{label:"intercept",min:-2,max:4,val:c,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){c=x;draw();}});
    draw();
  });

  /* 4.2 — the spectral norm is the biggest stretch */
  reg("4.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var A=[[1.6,.6],[.3,1.2]], th=20;
    function draw(){
      var P=plot(C.ctx,600,340,-2.6,2.6,-2.6,2.6,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      P.title("Spin a unit vector — ‖Ax‖ peaks at the spectral norm");
      var circ=[],ell=[];
      for(var i=0;i<=140;i++){var t=i/140*2*Math.PI,cx=Math.cos(t),cy=Math.sin(t);
        circ.push([cx,cy]); ell.push([A[0][0]*cx+A[0][1]*cy, A[1][0]*cx+A[1][1]*cy]);}
      P.poly(circ,PAL.ink3,1.4,[4,3]); P.poly(ell,PAL.indigo,2.3);
      var t2=th*Math.PI/180, x=[Math.cos(t2),Math.sin(t2)];
      var Ax=[A[0][0]*x[0]+A[0][1]*x[1], A[1][0]*x[0]+A[1][1]*x[1]];
      P.arrow(0,0,x[0],x[1],PAL.green,2.2); P.arrow(0,0,Ax[0],Ax[1],PAL.red,2.5);
      var nrm=Math.hypot(Ax[0],Ax[1]), best=0;
      for(var k=0;k<360;k++){var a=k*Math.PI/180;
        best=Math.max(best,Math.hypot(A[0][0]*Math.cos(a)+A[0][1]*Math.sin(a), A[1][0]*Math.cos(a)+A[1][1]*Math.sin(a)));}
      P.txt(16,324,"‖x‖ = 1    ‖Ax‖ = "+nrm.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(300,324,"max = ‖A‖₂ = "+best.toFixed(3)+(Math.abs(nrm-best)<.01?"  ← you found it":""),
            Math.abs(nrm-best)<.01?PAL.green:PAL.red,13,"left","700");
    }
    slider(controls,{label:"direction",min:0,max:360,val:th,step:1,fmt:function(x){return x+"°";},on:function(x){th=x;draw();}});
    draw();
  });

  /* 4.4 — conditioning: how far the answer moves when the data twitches */
  reg("4.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var m2=-1;
    function draw(){
      var P=plot(C.ctx,600,340,-1,5,-1,5);
      P.clear(); P.grid([0,1,2,3,4],[0,1,2,3,4]); P.axes();
      P.title("Nudge one line a hair — how far does the crossing move?");
      var m1=1,c1=0, c2=3, nud=0.25;
      P.fn(function(x){return m1*x+c1;},-1,5,PAL.blue,2.2);
      P.fn(function(x){return m2*x+c2;},-1,5,PAL.green,2.2);
      P.fn(function(x){return m2*x+c2+nud;},-1,5,PAL.green,1.5,[5,3]);
      function ix(mm,cc){ var x=(cc-c1)/(m1-mm); return [x, m1*x+c1]; }
      var p0=ix(m2,c2), p1=ix(m2,c2+nud);
      P.dot(p0[0],p0[1],PAL.ink,5); P.dot(p1[0],p1[1],PAL.red,5);
      var move=Math.hypot(p1[0]-p0[0],p1[1]-p0[1]);
      if (move>0.25) P.arrow(p0[0],p0[1],p1[0],p1[1],PAL.red,1.8);
      var ang=Math.abs(Math.atan(m1)-Math.atan(m2))*180/Math.PI;
      P.txt(16,324,"angle between lines = "+ang.toFixed(1)+"°",PAL.ink2,13,"left","600");
      P.txt(280,324,"solution moved "+move.toFixed(2)+(move>1?"  ← ill-conditioned!":"  (stable)"),
            move>1?PAL.red:PAL.green,13,"left","700");
    }
    slider(controls,{label:"2nd line slope",min:-2,max:0.92,val:m2,step:.02,fmt:function(x){return x.toFixed(2);},on:function(x){m2=x;draw();}});
    draw();
  });

  /* 5.3 — positive definite = bowl; indefinite = saddle */
  reg("5.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var a=2, b=0.6, c=1.2;
    function draw(){
      var P=plot(C.ctx,600,340,-2.8,2.8,-2.8,2.8,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      var tr=(a+c)/2, dd=Math.sqrt(Math.pow((a-c)/2,2)+b*b);
      var l1=tr+dd, l2=tr-dd, pd=(l1>0&&l2>0);
      P.title(pd ? "Both eigenvalues positive — a bowl (one true minimum)"
                 : "Mixed signs — a saddle (flat gradient, but no minimum)");
      var levels=[0.7,1.8,3.4];
      for (var li=0;li<3;li++){
        var L=levels[li], pts=[], seg=[];
        for (var i=0;i<=300;i++){
          var t=i/300*2*Math.PI, cx=Math.cos(t), sy=Math.sin(t);
          var q=a*cx*cx+2*b*cx*sy+c*sy*sy;
          if (q<=1e-6) { if(seg.length>1) pts.push(seg); seg=[]; continue; }
          var r=Math.sqrt(L/q);
          if (r>3.6) { if(seg.length>1) pts.push(seg); seg=[]; continue; }
          seg.push([r*cx, r*sy]);
        }
        if (seg.length>1) pts.push(seg);
        pts.forEach(function(sg){ P.poly(sg,[PAL.blueF,PAL.blue,PAL.indigo][li],1.9); });
      }
      P.dot(0,0,pd?PAL.green:PAL.red,5);
      P.txt(16,324,"λ₁ = "+l1.toFixed(2)+"   λ₂ = "+l2.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(300,324, pd?"positive definite ✓":"indefinite — saddle", pd?PAL.green:PAL.red,13,"left","700");
    }
    slider(controls,{label:"a",min:-2,max:3,val:a,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){a=x;draw();}});
    slider(controls,{label:"b",min:-2,max:2,val:b,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){b=x;draw();}});
    slider(controls,{label:"c",min:-2,max:3,val:c,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){c=x;draw();}});
    draw();
  });

  /* 6.4 — curvature flips the surface from bowl to saddle */
  reg("6.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var k=1;
    function draw(){
      var P=plot(C.ctx,600,340,-2.6,2.6,-2.6,2.6,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      var bowl = k>0.02;
      P.title(bowl ? "Hessian eigenvalues 2 and "+(2*k).toFixed(1)+" — both up: a bowl"
                   : "One curvature negative — a saddle point");
      var ys=[]; for(var y=-2.5;y<=2.5;y+=0.05) ys.push(y);
      [0.7,1.8,3.4].forEach(function(L,li){
        var col=[PAL.blueF,PAL.blue,PAL.indigo][li];
        if (bowl){ var pts=[];
          for(var i=0;i<=160;i++){var t=i/160*2*Math.PI;
            pts.push([Math.sqrt(L)*Math.cos(t), Math.sqrt(L/k)*Math.sin(t)]);}
          P.poly(pts,col,1.9);
        } else { var ak=Math.abs(k)||0.001;
          [1,-1].forEach(function(sgn){ var s1=[],s2=[];
            ys.forEach(function(y){ var x2=L+ak*y*y; if(x2>0){var x=Math.sqrt(x2); if(x<2.6){s1.push([sgn*x,y]);}} });
            if(s1.length>1) P.poly(s1,col,1.8);
          });
          [1,-1].forEach(function(sgn){ var s=[];
            for(var x=-2.5;x<=2.5;x+=0.05){ var y2=(x*x-L)/ak; if(y2>0){var y=Math.sqrt(y2); if(y<2.6) s.push([x,sgn*y]);} }
            if(s.length>1) P.poly(s,PAL.amber,1.6);
          });
        }
      });
      P.dot(0,0,bowl?PAL.green:PAL.red,5);
      P.txt(16,324,"f = x² + ("+k.toFixed(1)+")·y²",PAL.ink2,13,"left","600");
      P.txt(260,324, bowl?"minimum ✓ — every direction curves up":"saddle — up one way, down the other",
            bowl?PAL.green:PAL.red,13,"left","700");
    }
    slider(controls,{label:"y-curvature",min:-2,max:3,val:k,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){k=x;draw();}});
    draw();
  });

  /* 7.4 — the bell's area converges to √π */
  reg("7.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var L=1;
    function draw(){
      var P=plot(C.ctx,600,340,-3.2,3.2,-0.08,1.15);
      P.clear(); P.axes();
      P.title("Widen the window — the area under e^(−x²) closes in on √π");
      var f=function(x){return Math.exp(-x*x);}, poly=[[-L,0]];
      for(var x=-L;x<=L;x+=0.02) poly.push([x,f(x)]);
      poly.push([L,0]);
      P.fill(poly,"rgba(79,70,229,0.22)");
      P.fn(f,-3.2,3.2,PAL.indigo,2.6);
      P.line(-L,0,-L,f(-L),PAL.red,1.4,[4,3]); P.line(L,0,L,f(L),PAL.red,1.4,[4,3]);
      var area=Math.sqrt(Math.PI)*_erf(L), target=Math.sqrt(Math.PI);
      P.txt(16,324,"∫ from −"+L.toFixed(1)+" to "+L.toFixed(1)+" = "+area.toFixed(5),PAL.indigo,13,"left","600");
      P.txt(320,324,"√π = "+target.toFixed(5)+"   gap = "+(target-area).toExponential(1),PAL.ink2,13,"left","600");
    }
    slider(controls,{label:"half-width",min:0.2,max:3,val:L,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){L=x;draw();}});
    draw();
  });

  /* 9.5 — correlation you can dial */
  reg("9.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var rho=0.8, base=[];
    for (var i=0;i<220;i++) base.push([_randn(),_randn()]);
    function draw(){
      var P=plot(C.ctx,600,340,-3.4,3.4,-3.4,3.4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("Dial ρ — watch the cloud tilt from a line to a blob to a line");
      var s=Math.sqrt(Math.max(0,1-rho*rho)), sx=0,sy=0,sxy=0,sxx=0,syy=0,n=0;
      base.forEach(function(z){ var x=z[0], y=rho*z[0]+s*z[1];
        if (Math.abs(x)<3.3 && Math.abs(y)<3.3){ C.ctx.fillStyle=PAL.blue; C.ctx.globalAlpha=.75;
          C.ctx.beginPath(); C.ctx.arc(P.X(x),P.Y(y),2.4,0,7); C.ctx.fill(); }
        sx+=x; sy+=y; sxy+=x*y; sxx+=x*x; syy+=y*y; n++; });
      C.ctx.globalAlpha=1;
      var r=(n*sxy-sx*sy)/Math.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy));
      P.txt(16,324,"ρ set to "+rho.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(220,324,"measured r = "+r.toFixed(3),PAL.indigo,13,"left","700");
      P.txt(400,324, Math.abs(rho)<0.15?"no linear link":(rho>0?"rise together":"one up, one down"),PAL.ink3,12,"left");
    }
    slider(controls,{label:"ρ",min:-1,max:1,val:rho,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){rho=x;draw();}});
    button(controls,"New sample",function(){ base=[]; for(var i=0;i<220;i++) base.push([_randn(),_randn()]); draw(); });
    draw();
  });

  /* 10.1 — the CDF is the area swept so far */
  reg("10.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var x0=0;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,-3.4,3.4,-0.05,0.55,{l:52,r:20,t:26,b:150});
      P.clear();
      P.title("Slide x — shaded area under the PDF = height of the CDF");
      var pdf=function(x){return Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);};
      var poly=[[-3.4,0]]; for(var x=-3.4;x<=x0;x+=0.02) poly.push([x,pdf(x)]); poly.push([x0,0]);
      P.fill(poly,"rgba(79,70,229,0.25)");
      P.fn(pdf,-3.4,3.4,PAL.indigo,2.5); P.axes();
      P.line(x0,0,x0,pdf(x0),PAL.red,1.6);
      // CDF strip below
      var cy0=232, cH=76, cx0=P.px, cW=P.pw;
      ctx.strokeStyle=PAL.axis; ctx.lineWidth=1.2;
      ctx.beginPath(); ctx.moveTo(cx0,cy0+cH); ctx.lineTo(cx0+cW,cy0+cH); ctx.stroke();
      ctx.strokeStyle=PAL.green; ctx.lineWidth=2.2; ctx.beginPath();
      for (var i=0;i<=200;i++){ var xx=-3.4+6.8*i/200, yy=_ncdf(xx);
        var px=cx0+(xx+3.4)/6.8*cW, py=cy0+cH-yy*cH; i?ctx.lineTo(px,py):ctx.moveTo(px,py); }
      ctx.stroke();
      var Fx=_ncdf(x0), mx=cx0+(x0+3.4)/6.8*cW, my=cy0+cH-Fx*cH;
      ctx.fillStyle=PAL.red; ctx.beginPath(); ctx.arc(mx,my,4.5,0,7); ctx.fill();
      ctx.strokeStyle=PAL.red; ctx.setLineDash([4,3]); ctx.beginPath();
      ctx.moveTo(mx,cy0+cH); ctx.lineTo(mx,my); ctx.stroke(); ctx.setLineDash([]);
      P.txt(cx0,cy0-6,"CDF  P(X ≤ x)",PAL.green,11.5,"left","700");
      P.txt(16,324,"x = "+x0.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(180,324,"shaded area = P(X ≤ x) = "+Fx.toFixed(4),PAL.indigo,13,"left","700");
    }
    slider(controls,{label:"x",min:-3,max:3,val:x0,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){x0=v;draw();}});
    draw();
  });

  /* 11.5 — the p-value is the tail you're buying */
  reg("11.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var z=1.9;
    function draw(){
      var P=plot(C.ctx,600,340,-3.8,3.8,-0.04,0.48);
      P.clear(); P.axes();
      P.title("Slide your observed result — the red tail is the p-value");
      var pdf=function(x){return Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);};
      var body=[[-3.8,0]]; for(var x=-3.8;x<=3.8;x+=0.02) body.push([x,pdf(x)]); body.push([3.8,0]);
      P.fill(body,"rgba(42,111,151,0.13)");
      var tail=[[z,0]]; for(x=z;x<=3.8;x+=0.02) tail.push([x,pdf(x)]); tail.push([3.8,0]);
      P.fill(tail,"rgba(192,57,43,0.55)");
      P.fn(pdf,-3.8,3.8,PAL.indigo,2.5);
      P.line(z,0,z,pdf(z)+0.06,PAL.red,1.8);
      var p=1-_ncdf(z);
      P.txt(16,324,"observed z = "+z.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(210,324,"p = "+p.toFixed(4),PAL.red,13,"left","700");
      P.txt(340,324, p<0.05?"below 0.05 — 'significant' by convention":"above 0.05 — not surprising under H₀",
            p<0.05?PAL.green:PAL.ink3,12.5,"left","600");
    }
    slider(controls,{label:"observed z",min:-1,max:4,val:z,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){z=v;draw();}});
    draw();
  });


  /* 5.5 — PCA finds the long axis of the cloud */
  reg("5.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var rho=0.85, sx=1.5, base=[];
    for (var i=0;i<200;i++) base.push([_randn(),_randn()]);
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-4,4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("PC1 is the direction the data actually spreads");
      var s=Math.sqrt(Math.max(0,1-rho*rho)), pts=[];
      base.forEach(function(z){ pts.push([sx*z[0], rho*z[0]+s*z[1]]); });
      pts.forEach(function(p){ if(Math.abs(p[0])<3.9&&Math.abs(p[1])<3.9){
        C.ctx.fillStyle=PAL.blue; C.ctx.globalAlpha=.7;
        C.ctx.beginPath(); C.ctx.arc(P.X(p[0]),P.Y(p[1]),2.4,0,7); C.ctx.fill(); } });
      C.ctx.globalAlpha=1;
      var n=pts.length, mx=0,my=0; pts.forEach(function(p){mx+=p[0];my+=p[1];}); mx/=n; my/=n;
      var cxx=0,cyy=0,cxy=0;
      pts.forEach(function(p){ cxx+=(p[0]-mx)*(p[0]-mx); cyy+=(p[1]-my)*(p[1]-my); cxy+=(p[0]-mx)*(p[1]-my); });
      cxx/=n; cyy/=n; cxy/=n;
      var tr=(cxx+cyy)/2, dd=Math.sqrt(Math.pow((cxx-cyy)/2,2)+cxy*cxy);
      var l1=tr+dd, l2=Math.max(tr-dd,1e-9);
      var ang=0.5*Math.atan2(2*cxy, cxx-cyy);
      var u1=[Math.cos(ang),Math.sin(ang)], u2=[-u1[1],u1[0]];
      P.arrow(mx,my,mx+u1[0]*Math.sqrt(l1)*2.1,my+u1[1]*Math.sqrt(l1)*2.1,PAL.red,2.8);
      P.arrow(mx,my,mx+u2[0]*Math.sqrt(l2)*2.1,my+u2[1]*Math.sqrt(l2)*2.1,PAL.green,2.4);
      P.txt(16,324,"variance on PC1 = "+(100*l1/(l1+l2)).toFixed(1)+"%",PAL.red,13,"left","700");
      P.txt(280,324,"on PC2 = "+(100*l2/(l1+l2)).toFixed(1)+"%  → drop it, lose little",PAL.green,13,"left","600");
    }
    slider(controls,{label:"correlation",min:-0.98,max:0.98,val:rho,step:.02,fmt:function(x){return x.toFixed(2);},on:function(x){rho=x;draw();}});
    slider(controls,{label:"x-spread",min:0.4,max:2.5,val:sx,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){sx=x;draw();}});
    draw();
  });

  /* 6.2 — the chain rule multiplies the two slopes */
  reg("6.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var k=1.4, x0=1.0;
    var g=function(x){return Math.sin(k*x);}, h=function(x){return Math.pow(g(x),2);};
    function draw(){
      var P=plot(C.ctx,600,340,-0.3,4.3,-1.25,1.35);
      P.clear(); P.grid([0,1,2,3,4],[-1,0,1]); P.axes();
      P.title("y = sin(kx)² — the tangent slope is f′(u)·g′(x)");
      P.fn(g,-0.3,4.3,PAL.ink3,1.6);
      P.fn(h,-0.3,4.3,PAL.indigo,2.6);
      var gp=k*Math.cos(k*x0), fp=2*g(x0), slope=fp*gp;
      P.poly([[x0-0.7,h(x0)-slope*0.7],[x0+0.7,h(x0)+slope*0.7]],PAL.red,2.2);
      P.dot(x0,h(x0),PAL.red,5);
      var num=(h(x0+1e-5)-h(x0-1e-5))/2e-5;
      P.txt(16,324,"g′(x) = "+gp.toFixed(3)+"    f′(u) = "+fp.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(300,324,"product = "+slope.toFixed(3)+"   (true slope "+num.toFixed(3)+") ✓",PAL.red,13,"left","700");
    }
    slider(controls,{label:"k",min:0.4,max:3,val:k,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){k=x;draw();}});
    slider(controls,{label:"x",min:0.1,max:4,val:x0,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){x0=x;draw();}});
    draw();
  });

  /* 7.5 — the mean is where the density balances */
  reg("7.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var sh=3;
    function draw(){
      var P=plot(C.ctx,600,340,-0.3,10.5,-0.03,0.42);
      P.clear(); P.axes();
      P.title("Skew the distribution — the mean is dragged toward the long tail");
      function lg(z){var g=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
        if(z<0.5) return Math.log(Math.PI/Math.sin(Math.PI*z))-lg(1-z);
        z-=1; var x=0.99999999999980993; for(var i=0;i<8;i++) x+=g[i]/(z+i+1);
        var t=z+7.5; return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(x);}
      var f=function(x){ return x<=0?0:Math.exp((sh-1)*Math.log(x)-x-lg(sh)); };
      var poly=[[0,0]]; for(var x=0.01;x<=10.4;x+=0.04) poly.push([x,f(x)]); poly.push([10.4,0]);
      P.fill(poly,"rgba(42,111,151,0.20)");
      P.fn(f,0.01,10.4,PAL.blue,2.5);
      var mean=sh, mode=Math.max(sh-1,0);
      P.line(mean,0,mean,f(mean)+0.05,PAL.red,1.8,[4,3]);
      if (mode>0) P.line(mode,0,mode,f(mode),PAL.green,1.4,[3,3]);
      var bx=P.X(mean), by=P.Y(0);
      C.ctx.fillStyle=PAL.red; C.ctx.beginPath();
      C.ctx.moveTo(bx,by+2); C.ctx.lineTo(bx-10,by+18); C.ctx.lineTo(bx+10,by+18); C.ctx.closePath(); C.ctx.fill();
      P.txt(16,324,"peak (mode) = "+mode.toFixed(2),PAL.green,13,"left","600");
      P.txt(240,324,"mean (balance point) = "+mean.toFixed(2),PAL.red,13,"left","700");
    }
    slider(controls,{label:"shape",min:1.2,max:8,val:sh,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){sh=x;draw();}});
    draw();
  });

  /* 8.4 — momentum vs plain gradient descent */
  reg("8.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var eta=0.14, beta=0.85, b=9;
    function run(useM){ var x=[4.6,1.4], v=[0,0], path=[[x[0],x[1]]];
      for (var i=0;i<40;i++){ var g=[x[0], b*x[1]];
        if (useM){ v=[beta*v[0]-eta*g[0], beta*v[1]-eta*g[1]]; x=[x[0]+v[0], x[1]+v[1]]; }
        else x=[x[0]-eta*g[0], x[1]-eta*g[1]];
        if (!isFinite(x[0])||Math.abs(x[0])>9||Math.abs(x[1])>4) break;
        path.push([x[0],x[1]]); }
      return path; }
    function draw(){
      var P=plot(C.ctx,600,340,-5.2,5.2,-2.6,2.6);
      P.clear(); P.grid([-4,-2,0,2,4],[-2,-1,0,1,2]); P.axes();
      P.title("Momentum rolls along the valley; plain GD bounces across it");
      [1.2,4.5,10].forEach(function(c,i){ var pts=[];
        for(var t=0;t<=140;t++){var a=t/140*2*Math.PI;
          pts.push([Math.sqrt(2*c)*Math.cos(a), Math.sqrt(2*c/b)*Math.sin(a)]);}
        P.poly(pts,[PAL.indigo,PAL.blue,PAL.blueF][i],1.6); });
      [[run(false),PAL.red],[run(true),PAL.green]].forEach(function(pr){
        P.poly(pr[0],pr[1],1.8);
        pr[0].forEach(function(p){ P.dot(p[0],p[1],pr[1],2.0); }); });
      P.txt(16,324,"— plain GD",PAL.red,12.5,"left","700");
      P.txt(130,324,"— momentum (β = "+beta.toFixed(2)+")",PAL.green,12.5,"left","700");
      P.txt(370,324,"η = "+eta.toFixed(2)+(eta*b>2?"  diverging!":""),eta*b>2?PAL.red:PAL.ink2,12.5,"left","600");
    }
    slider(controls,{label:"η",min:0.02,max:0.24,val:eta,step:.01,fmt:function(x){return x.toFixed(2);},on:function(x){eta=x;draw();}});
    slider(controls,{label:"β",min:0,max:0.95,val:beta,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){beta=x;draw();}});
    draw();
  });

  /* 8.5 — at the optimum the gradients line up */
  reg("8.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var th=45;
    function draw(){
      var P=plot(C.ctx,600,340,-1.8,1.8,-1.8,1.8,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-1,0,1],[-1,0,1]); P.axes();
      P.title("Walk the circle — the best point is where f's contour just kisses it");
      var pts=[]; for(var i=0;i<=160;i++){var a=i/160*2*Math.PI; pts.push([Math.cos(a),Math.sin(a)]);}
      P.poly(pts,PAL.blue,2.3);
      var t=th*Math.PI/180, p=[Math.cos(t),Math.sin(t)];
      var val=p[0]+p[1];
      [-1.2,0,1.2].forEach(function(c){ P.poly([[-1.8,c+1.8],[1.8,c-1.8]],PAL.ink3,1.1,[4,3]); });
      P.poly([[-1.8,val+1.8],[1.8,val-1.8]],PAL.indigo,2);
      var gf=[1,1], gl=Math.SQRT2, gg=[2*p[0],2*p[1]], ggl=Math.hypot(gg[0],gg[1]);
      P.arrow(p[0],p[1],p[0]+gg[0]/ggl*0.75,p[1]+gg[1]/ggl*0.75,PAL.green,2.2);
      P.arrow(p[0],p[1],p[0]+gf[0]/gl*0.5,p[1]+gf[1]/gl*0.5,PAL.red,2.6);
      P.dot(p[0],p[1],PAL.red,5);
      var cosang=(gf[0]*gg[0]+gf[1]*gg[1])/(gl*ggl);
      var aligned=cosang>0.999;
      P.txt(16,324,"f = x+y = "+val.toFixed(3),PAL.indigo,13,"left","600");
      P.txt(220,324, aligned ? "∇f ∥ ∇g — this is the maximum (√2)" : "gradients not parallel — keep walking",
            aligned?PAL.green:PAL.ink3,13,"left","700");
    }
    slider(controls,{label:"position",min:0,max:360,val:th,step:1,fmt:function(x){return x+"°";},on:function(x){th=x;draw();}});
    draw();
  });

  /* 9.2 — conditioning shrinks the world to B */
  reg("9.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var d=90;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Slide the overlap — P(A|B) is A's share of B, not of everything");
      var cx=300, cy=150, r=78;
      var ax=cx-d/2, bx=cx+d/2;
      ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1.4; ctx.strokeRect(110,60,380,180);
      ctx.fillStyle=PAL.green; ctx.globalAlpha=.30;
      ctx.beginPath(); ctx.arc(bx,cy,r,0,7); ctx.fill();
      ctx.fillStyle=PAL.blue; ctx.globalAlpha=.28;
      ctx.beginPath(); ctx.arc(ax,cy,r,0,7); ctx.fill();
      ctx.globalAlpha=1;
      ctx.strokeStyle=PAL.blue; ctx.beginPath(); ctx.arc(ax,cy,r,0,7); ctx.stroke();
      ctx.strokeStyle=PAL.green; ctx.beginPath(); ctx.arc(bx,cy,r,0,7); ctx.stroke();
      // circle-circle overlap area
      var R=r, dd=Math.abs(bx-ax), inter=0;
      if (dd < 2*R){ var q=Math.acos(Math.min(1,dd/(2*R)));
        inter = 2*R*R*q - 0.5*dd*Math.sqrt(Math.max(0,4*R*R-dd*dd)); }
      var areaB=Math.PI*R*R, cond=inter/areaB;
      P.txt(ax-30,cy,"A",PAL.blue,15,"left","700");
      P.txt(bx+16,cy,"B",PAL.green,15,"left","700");
      P.txt(495,72,"Ω",PAL.ink3,13,"left","700");
      P.txt(16,300,"overlap / B = P(A | B) = "+cond.toFixed(3),PAL.ink,13.5,"left","700");
      P.txt(330,300, cond>0.85?"B almost guarantees A":(cond<0.05?"B nearly rules A out":"partial evidence"),PAL.ink3,12.5,"left");
    }
    slider(controls,{label:"separation",min:0,max:170,val:d,step:2,fmt:function(x){return ""+x;},on:function(x){d=x;draw();}});
    draw();
  });

  /* 10.4 — covariance tilts the ellipse */
  reg("10.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var sx=1.3, sy=0.8, rho=0.6;
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-4,4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("Σ sets the tilt and the stretch of the contours");
      var cxx=sx*sx, cyy=sy*sy, cxy=rho*sx*sy;
      var tr=(cxx+cyy)/2, dd=Math.sqrt(Math.pow((cxx-cyy)/2,2)+cxy*cxy);
      var l1=tr+dd, l2=Math.max(tr-dd,1e-9), ang=0.5*Math.atan2(2*cxy,cxx-cyy);
      var u1=[Math.cos(ang),Math.sin(ang)], u2=[-u1[1],u1[0]];
      [1,2,3].forEach(function(kk,i){ var pts=[];
        for(var t=0;t<=160;t++){ var a=t/160*2*Math.PI;
          var e1=Math.sqrt(l1)*kk*Math.cos(a), e2=Math.sqrt(l2)*kk*Math.sin(a);
          pts.push([u1[0]*e1+u2[0]*e2, u1[1]*e1+u2[1]*e2]); }
        P.poly(pts,[PAL.indigo,PAL.blue,PAL.blueF][i],1.9); });
      P.arrow(0,0,u1[0]*Math.sqrt(l1)*2,u1[1]*Math.sqrt(l1)*2,PAL.red,2.4);
      P.arrow(0,0,u2[0]*Math.sqrt(l2)*2,u2[1]*Math.sqrt(l2)*2,PAL.green,2.2);
      P.txt(16,324,"Σ = ["+cxx.toFixed(2)+", "+cxy.toFixed(2)+"; "+cxy.toFixed(2)+", "+cyy.toFixed(2)+"]",PAL.ink2,12.5,"left","600");
      P.txt(340,324,"tilt = "+(ang*180/Math.PI).toFixed(0)+"°   axes √λ = "+Math.sqrt(l1).toFixed(2)+", "+Math.sqrt(l2).toFixed(2),PAL.indigo,12.5,"left","600");
    }
    slider(controls,{label:"σx",min:0.3,max:2,val:sx,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){sx=x;draw();}});
    slider(controls,{label:"σy",min:0.3,max:2,val:sy,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){sy=x;draw();}});
    slider(controls,{label:"ρ",min:-0.95,max:0.95,val:rho,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){rho=x;draw();}});
    draw();
  });

  /* 11.3 — a stronger prior pulls the estimate */
  reg("11.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var strength=2, k=7, n=10;
    function draw(){
      var P=plot(C.ctx,600,340,-0.02,1.02,-0.06,1.16);
      P.clear(); P.grid([0,0.25,0.5,0.75,1],[0,0.5,1]); P.axes();
      P.title("Strengthen the prior — MAP slides from the data toward the prior");
      var a0=strength, b0=strength;
      function norm(f){ var mx=0; for(var x=0.002;x<1;x+=0.002) mx=Math.max(mx,f(x));
        return function(x){ return f(x)/(mx||1); }; }
      var prior=norm(function(t){return Math.pow(t,a0-1)*Math.pow(1-t,b0-1);});
      var lik  =norm(function(t){return Math.pow(t,k)*Math.pow(1-t,n-k);});
      var post =norm(function(t){return Math.pow(t,a0-1+k)*Math.pow(1-t,b0-1+n-k);});
      P.fn(prior,0.002,0.998,PAL.green,2.1);
      P.fn(lik,0.002,0.998,PAL.blue,2.1);
      P.fn(post,0.002,0.998,PAL.indigo,2.8);
      var mle=k/n, map=(a0-1+k)/(a0+b0-2+n);
      P.line(mle,0,mle,1,PAL.blue,1.2,[4,3]);
      P.line(map,0,map,1,PAL.indigo,1.6,[4,3]);
      P.txt(16,324,"MLE = "+mle.toFixed(3),PAL.blue,13,"left","600");
      P.txt(180,324,"MAP = "+map.toFixed(3),PAL.indigo,13,"left","700");
      P.txt(340,324,"prior pulls it "+((mle-map)>0?"down":"up")+" toward 0.5",PAL.green,12.5,"left","600");
    }
    slider(controls,{label:"prior strength",min:1,max:30,val:strength,step:1,fmt:function(x){return x.toFixed(0);},on:function(x){strength=x;draw();}});
    slider(controls,{label:"heads k",min:0,max:10,val:k,step:1,fmt:function(x){return x+"/10";},on:function(x){k=x;draw();}});
    draw();
  });

  /* 12.4 — Jensen's gap */
  reg("12.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var ax=0.6, bx=3.4, w=0.5;
    var f=function(x){return 0.35*x*x;};
    function draw(){
      var P=plot(C.ctx,600,340,-0.3,4.3,-0.3,4.6);
      P.clear(); P.grid([0,1,2,3,4],[0,1,2,3,4]); P.axes();
      P.title("Convex f: the average of f sits ABOVE f of the average");
      P.fn(f,-0.3,3.6,PAL.indigo,2.6);
      var A=[ax,f(ax)], B=[bx,f(bx)];
      P.line(A[0],A[1],B[0],B[1],PAL.amber,2.2);
      P.dot(A[0],A[1],PAL.ink,4.5); P.dot(B[0],B[1],PAL.ink,4.5);
      var EX=w*ax+(1-w)*bx, Efx=w*f(ax)+(1-w)*f(bx), fEX=f(EX);
      P.line(EX,fEX,EX,Efx,PAL.red,2,[3,2]);
      P.dot(EX,Efx,PAL.amber,5); P.dot(EX,fEX,PAL.red,5);
      P.txt(16,324,"E[f(X)] = "+Efx.toFixed(3),PAL.amber,13,"left","700");
      P.txt(190,324,"f(E[X]) = "+fEX.toFixed(3),PAL.red,13,"left","700");
      P.txt(370,324,"gap = "+(Efx-fEX).toFixed(3)+" ≥ 0  (the ELBO)",PAL.ink2,12.5,"left","600");
    }
    slider(controls,{label:"weight",min:0.05,max:0.95,val:w,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){w=x;draw();}});
    slider(controls,{label:"spread",min:0.2,max:1.8,val:1,step:.1,fmt:function(x){return x.toFixed(1);},
      on:function(x){ ax=2-x; bx=2+x*0.9; draw(); }});
    draw();
  });

  /* 13.2 — subtract the max or exp() explodes */
  reg("13.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var big=90;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Raise the logits — naive exp() overflows; shifted stays safe");
      var logits=[big-2, big-1, big];
      var naive=logits.map(function(v){return Math.exp(v);});
      var shifted=logits.map(function(v){return Math.exp(v-big);});
      var sum=shifted.reduce(function(a,b){return a+b;},0);
      var soft=shifted.map(function(v){return v/sum;});
      function chip(x,y,txt,fill,stroke,w){ var W=w||96;
        ctx.fillStyle=fill; ctx.strokeStyle=stroke; ctx.lineWidth=1.2;
        ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x,y,W,30,7) : ctx.rect(x,y,W,30); ctx.fill(); ctx.stroke();
        P.txt(x+W/2,y+20,txt,stroke,12.5,"center","600"); }
      P.txt(40,68,"logits",PAL.ink2,12.5,"left","700");
      logits.forEach(function(v,i){ chip(40+i*104,80,v.toFixed(0),PAL.blueF,PAL.blue); });
      P.txt(40,150,"naive  exp(x)",PAL.red,12.5,"left","700");
      naive.forEach(function(v,i){ var ovf=!isFinite(v);
        chip(40+i*104,162, ovf?"overflow":v.toExponential(1), ovf?PAL.redF:PAL.blueF, ovf?PAL.red:PAL.blue); });
      P.txt(40,232,"exp(x − max)",PAL.green,12.5,"left","700");
      shifted.forEach(function(v,i){ chip(40+i*104,244,v.toFixed(3),PAL.greenF,PAL.green); });
      var tx=380;
      P.txt(tx,150,"softmax (either way)",PAL.ink2,12,"left","700");
      soft.forEach(function(v,i){ P.txt(tx,174+i*22,"p"+i+" = "+v.toFixed(4),PAL.indigo,12.5,"left","600"); });
      P.txt(tx,252, naive.some(function(v){return !isFinite(v);}) ? "naive route already broke" : "naive still OK — keep sliding",
            naive.some(function(v){return !isFinite(v);})?PAL.red:PAL.ink3,12,"left","600");
    }
    slider(controls,{label:"logit size",min:1,max:900,val:big,step:1,fmt:function(x){return x.toFixed(0);},on:function(x){big=x;draw();}});
    draw();
  });


  /* 1.2 — Σ is just a running total */
  reg("1.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var n=5, a=[3,1,4,1,5,9,2,6];
    function draw(){
      var P=plot(C.ctx,600,340,-0.6,8.6,0,22);
      P.clear(); P.axes();
      P.title("Slide the index — Σ adds one more term at each step");
      var run=0;
      for (var i=0;i<8;i++){
        var on=i<n, h=a[i];
        var X=P.X(i)-20, Y=P.Y(h), W=40, H=P.Y(0)-P.Y(h);
        C.ctx.fillStyle= on?PAL.blueF:"rgba(0,0,0,0.04)";
        C.ctx.fillRect(X,Y,W,H);
        C.ctx.strokeStyle= on?PAL.blue:PAL.line||PAL.ink3; C.ctx.lineWidth=1.2; C.ctx.strokeRect(X,Y,W,H);
        P.txt(P.X(i),P.Y(h)-8,""+h, on?PAL.blue:PAL.ink3,12,"center","600");
        P.txt(P.X(i),P.Y(0)+16,"a"+(i+1), on?PAL.ink2:PAL.ink3,10.5,"center");
        if (on) run+=h;
      }
      var X2=P.X(7.6)-20, Y2=P.Y(run);
      C.ctx.fillStyle=PAL.indigoF; C.ctx.fillRect(X2,Y2,40,P.Y(0)-P.Y(run));
      C.ctx.strokeStyle=PAL.indigo; C.ctx.lineWidth=1.6; C.ctx.strokeRect(X2,Y2,40,P.Y(0)-P.Y(run));
      P.txt(16,324,"Σ from i=1 to "+n+"  =  "+a.slice(0,n).join(" + ")+"  =  "+run,PAL.indigo,13,"left","700");
    }
    slider(controls,{label:"n",min:1,max:8,val:n,step:1,fmt:function(x){return ""+x;},on:function(x){n=x;draw();}});
    draw();
  });

  /* 1.3 — order matters in a composition */
  reg("1.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,300), controls=controlsIn(wrap);
    var x=3, fg=true;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,300,0,1,0,1); P.clear();
      P.title(fg ? "f(g(x)) — add 1 first, then square" : "g(f(x)) — square first, then add 1");
      var y=140;
      function box(cx,lab,sub,fill,stroke){
        ctx.fillStyle=fill; ctx.strokeStyle=stroke; ctx.lineWidth=1.6;
        ctx.beginPath(); ctx.roundRect? ctx.roundRect(cx-46,y-32,92,64,11) : ctx.rect(cx-46,y-32,92,64);
        ctx.fill(); ctx.stroke();
        P.txt(cx,y-4,lab,stroke,16,"center","700"); P.txt(cx,y+18,sub,PAL.ink3,11,"center"); }
      var first = fg? ["g","add 1"] : ["f","square"];
      var second= fg? ["f","square"] : ["g","add 1"];
      var mid = fg? (x+1) : (x*x);
      var out = fg? (x+1)*(x+1) : (x*x)+1;
      P.txt(60,y+5,"x = "+x,PAL.ink,16,"left","700");
      ctx.strokeStyle=PAL.axis; ctx.lineWidth=2;
      [[125,178],[268,322],[412,466]].forEach(function(seg){
        ctx.beginPath(); ctx.moveTo(seg[0],y); ctx.lineTo(seg[1],y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(seg[1],y); ctx.lineTo(seg[1]-8,y-4); ctx.lineTo(seg[1]-8,y+4); ctx.closePath();
        ctx.fillStyle=PAL.axis; ctx.fill(); });
      box(224,first[0],first[1],PAL.blueF,PAL.blue);
      box(368,second[0],second[1],PAL.greenF,PAL.green);
      P.txt(295,y-16,""+mid,PAL.ink2,12,"center","600");
      P.txt(516,y+5,""+out,PAL.indigo,18,"left","700");
      P.txt(16,268,"f(g("+x+")) = "+((x+1)*(x+1))+"     g(f("+x+")) = "+((x*x)+1)+"   → different!",PAL.ink2,13,"left","600");
    }
    slider(controls,{label:"x",min:0,max:8,val:x,step:1,fmt:function(v){return ""+v;},on:function(v){x=v;draw();}});
    button(controls,"Swap order",function(){ fg=!fg; draw(); });
    draw();
  });

  /* 3.3 — independent or stuck on a line? */
  reg("3.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,360), controls=controlsIn(wrap);
    var u=[2.2,0.7], v=[0.7,2.2], cfg={xmin:-3.4,xmax:3.4,ymin:-3.4,ymax:3.4}, P;
    function draw(){
      P=plot(C.ctx,600,360,cfg.xmin,cfg.xmax,cfg.ymin,cfg.ymax,{l:170,r:170,t:26,b:54});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      var det=u[0]*v[1]-u[1]*v[0], indep=Math.abs(det)>0.08;
      P.title(indep ? "Independent — together they reach every point" : "Collinear — they only reach one line");
      if (!indep){ var L=Math.hypot(u[0],u[1])||1;
        P.poly([[-3.3*u[0]/L,-3.3*u[1]/L],[3.3*u[0]/L,3.3*u[1]/L]],PAL.red,1.4,[5,4]); }
      else { P.fill([[0,0],u,[u[0]+v[0],u[1]+v[1]],v],"rgba(79,70,229,0.10)"); }
      P.arrow(0,0,u[0],u[1], indep?PAL.blue:PAL.red, 2.7);
      P.arrow(0,0,v[0],v[1], indep?PAL.green:PAL.red, 2.7);
      P.dot(v[0],v[1], indep?PAL.green:PAL.red, 5);
      P.txt(16,344,"det = "+det.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(190,344, indep?"basis ✓ — spans the plane":"det ≈ 0 → NOT a basis", indep?PAL.green:PAL.red,13,"left","700");
    }
    draw();
    function fm(e){var r=C.cv.getBoundingClientRect();
      v=[Math.max(cfg.xmin,Math.min(cfg.xmax,P.invX(e.clientX-r.left))),Math.max(cfg.ymin,Math.min(cfg.ymax,P.invY(e.clientY-r.top)))];draw();}
    var dg=false;
    C.cv.addEventListener("mousedown",function(e){dg=true;fm(e);});
    C.cv.addEventListener("mousemove",function(e){if(dg)fm(e);});
    window.addEventListener("mouseup",function(){dg=false;});
    slider(controls,{label:"v.x",min:-3,max:3,val:v[0],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){v[0]=x;draw();}});
    slider(controls,{label:"v.y",min:-3,max:3,val:v[1],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){v[1]=x;draw();}});
  });

  /* 3.5 — Gram-Schmidt straightens the pair */
  reg("3.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,360), controls=controlsIn(wrap);
    var a1=[3,1], a2=[1,2.5], cfg={xmin:-1.4,xmax:4.2,ymin:-1.4,ymax:3.6}, P;
    function draw(){
      P=plot(C.ctx,600,360,cfg.xmin,cfg.xmax,cfg.ymin,cfg.ymax);
      P.clear(); P.grid([-1,0,1,2,3,4],[-1,0,1,2,3]); P.axes();
      P.title("Drag a₂ — q₂ is whatever is left after removing the q₁ part");
      var l1=Math.hypot(a1[0],a1[1]), q1=[a1[0]/l1,a1[1]/l1];
      var proj=a2[0]*q1[0]+a2[1]*q1[1], pv=[proj*q1[0],proj*q1[1]];
      var u2=[a2[0]-pv[0],a2[1]-pv[1]], l2=Math.hypot(u2[0],u2[1])||1e-9;
      var q2=[u2[0]/l2,u2[1]/l2];
      P.arrow(0,0,a1[0],a1[1],PAL.ink3,1.5); P.arrow(0,0,a2[0],a2[1],PAL.ink3,1.5);
      P.line(a2[0],a2[1],pv[0],pv[1],PAL.amber,1.6,[4,3]);
      P.arrow(0,0,q1[0],q1[1],PAL.blue,3);
      P.arrow(0,0,q2[0],q2[1],PAL.green,3);
      P.dot(a2[0],a2[1],PAL.ink3,5);
      P.txt(16,344,"q₁·q₂ = "+(q1[0]*q2[0]+q1[1]*q2[1]).toFixed(6),PAL.green,13,"left","700");
      P.txt(250,344,"‖q₁‖ = "+Math.hypot(q1[0],q1[1]).toFixed(3)+"   ‖q₂‖ = "+Math.hypot(q2[0],q2[1]).toFixed(3),PAL.ink2,13,"left","600");
    }
    draw();
    function fm(e){var r=C.cv.getBoundingClientRect();
      a2=[Math.max(cfg.xmin,Math.min(cfg.xmax,P.invX(e.clientX-r.left))),Math.max(cfg.ymin,Math.min(cfg.ymax,P.invY(e.clientY-r.top)))];draw();}
    var dg=false;
    C.cv.addEventListener("mousedown",function(e){dg=true;fm(e);});
    C.cv.addEventListener("mousemove",function(e){if(dg)fm(e);});
    window.addEventListener("mouseup",function(){dg=false;});
    slider(controls,{label:"a₂.x",min:-1,max:4,val:a2[0],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){a2[0]=x;draw();}});
    slider(controls,{label:"a₂.y",min:-1,max:3.5,val:a2[1],step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){a2[1]=x;draw();}});
  });

  /* 4.3 — squash the square and the inverse dies */
  reg("4.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var d=1.3;
    function draw(){
      var P=plot(C.ctx,600,340,-1,3.4,-1,3.4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([0,1,2,3],[0,1,2,3]); P.axes();
      var a=1.5,b=0.5,c=0.4;
      var det=a*d-b*c, sing=Math.abs(det)<0.06;
      P.title(sing ? "det ≈ 0 — the square collapsed to a line; no way back"
                   : "det ≠ 0 — area survives, so the map can be undone");
      P.poly([[0,0],[1,0],[1,1],[0,1],[0,0]],PAL.ink3,1.3);
      var img=[[0,0],[a,c],[a+b,c+d],[b,d],[0,0]];
      if(!sing) P.fill(img.slice(0,4),"rgba(79,70,229,0.14)");
      P.poly(img, sing?PAL.red:PAL.indigo, 2.2);
      P.arrow(0,0,a,c,PAL.blue,2.2); P.arrow(0,0,b,d,PAL.green,2.2);
      P.txt(16,324,"det = "+det.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(180,324, sing?"singular — NOT invertible":"invertible ✓  (area ×"+Math.abs(det).toFixed(2)+")",
            sing?PAL.red:PAL.green,13,"left","700");
    }
    slider(controls,{label:"d",min:0.05,max:2.5,val:d,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){d=x;draw();}});
    draw();
  });

  /* 5.2 — symmetric ⇒ perpendicular eigen-axes */
  reg("5.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var a=2, b=1, c=2;
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-4,4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("Symmetric matrix — the circle's image lines up with the eigen-axes");
      var circ=[],ell=[];
      for(var i=0;i<=150;i++){var t=i/150*2*Math.PI,cx=Math.cos(t),sy=Math.sin(t);
        circ.push([cx,sy]); ell.push([a*cx+b*sy, b*cx+c*sy]);}
      P.poly(circ,PAL.ink3,1.4,[4,3]); P.poly(ell,PAL.indigo,2.4);
      var tr=(a+c)/2, dd=Math.sqrt(Math.pow((a-c)/2,2)+b*b);
      var l1=tr+dd, l2=tr-dd, ang=0.5*Math.atan2(2*b,a-c);
      var u1=[Math.cos(ang),Math.sin(ang)], u2=[-u1[1],u1[0]];
      P.arrow(0,0,u1[0]*l1,u1[1]*l1,PAL.red,2.5);
      P.arrow(0,0,u2[0]*l2,u2[1]*l2,PAL.green,2.3);
      P.txt(16,324,"λ₁ = "+l1.toFixed(2)+"   λ₂ = "+l2.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(270,324,"v₁·v₂ = "+(u1[0]*u2[0]+u1[1]*u2[1]).toFixed(6)+"  → always perpendicular",PAL.green,12.5,"left","700");
    }
    slider(controls,{label:"a",min:-3,max:3,val:a,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){a=x;draw();}});
    slider(controls,{label:"b",min:-3,max:3,val:b,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){b=x;draw();}});
    slider(controls,{label:"c",min:-3,max:3,val:c,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){c=x;draw();}});
    draw();
  });

  /* 7.2 — area grows at the rate of the height */
  reg("7.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var x0=2.4;
    var f=function(x){return 0.6+0.9*x-0.13*x*x;};
    var F=function(x){return 0.6*x+0.45*x*x-0.13*x*x*x/3;};
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,-0.2,5.2,-0.3,3.4,{l:52,r:20,t:26,b:150});
      P.clear();
      P.title("Slide x — the shaded area's growth rate IS the curve's height");
      var poly=[[0,0]]; for(var t=0;t<=x0;t+=0.02) poly.push([t,f(t)]); poly.push([x0,0]);
      P.fill(poly,"rgba(42,111,151,0.22)");
      P.fn(f,-0.2,5.2,PAL.indigo,2.5); P.axes();
      P.line(x0,0,x0,f(x0),PAL.red,1.8);
      var cy0=232,cH=76,cx0=P.px,cW=P.pw, mxF=F(5);
      ctx.strokeStyle=PAL.axis; ctx.lineWidth=1.2;
      ctx.beginPath(); ctx.moveTo(cx0,cy0+cH); ctx.lineTo(cx0+cW,cy0+cH); ctx.stroke();
      ctx.strokeStyle=PAL.green; ctx.lineWidth=2.2; ctx.beginPath();
      for(var i=0;i<=200;i++){ var xx=5.4*i/200, yy=F(xx)/mxF;
        var px=cx0+xx/5.4*cW, py=cy0+cH-yy*cH; i?ctx.lineTo(px,py):ctx.moveTo(px,py); }
      ctx.stroke();
      var mx=cx0+x0/5.4*cW, my=cy0+cH-F(x0)/mxF*cH;
      ctx.fillStyle=PAL.red; ctx.beginPath(); ctx.arc(mx,my,4.5,0,7); ctx.fill();
      P.txt(cx0,cy0-6,"A(x) = area so far",PAL.green,11.5,"left","700");
      P.txt(16,324,"A(x) = "+F(x0).toFixed(3),PAL.green,13,"left","700");
      P.txt(190,324,"slope of A at x = "+((F(x0+1e-4)-F(x0-1e-4))/2e-4).toFixed(3)+"   =   f(x) = "+f(x0).toFixed(3),PAL.red,13,"left","700");
    }
    slider(controls,{label:"x",min:0.1,max:5,val:x0,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){x0=v;draw();}});
    draw();
  });

  /* 8.1 — hunt the flat spots */
  reg("8.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var x0=-1.2;
    var f=function(x){return 0.2*Math.pow(x,4)-1.1*x*x+0.1*x;};
    var fp=function(x){return 0.8*x*x*x-2.2*x+0.1;};
    var fpp=function(x){return 2.4*x*x-2.2;};
    function draw(){
      var P=plot(C.ctx,600,340,-2.8,2.8,-2.2,1.8);
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1]); P.axes();
      P.title("Slide along — a flat tangent means a critical point");
      P.fn(f,-2.8,2.8,PAL.indigo,2.6,undefined);
      var s=fp(x0), y=f(x0);
      P.poly([[x0-0.85,y-s*0.85],[x0+0.85,y+s*0.85]],PAL.red,2.2);
      P.dot(x0,y,PAL.red,5);
      var flat=Math.abs(s)<0.09, kind = fpp(x0)>0?"minimum":"maximum";
      P.txt(16,324,"f′(x) = "+s.toFixed(3),flat?PAL.green:PAL.ink2,13,"left","600");
      P.txt(200,324, flat ? ("flat! f″ = "+fpp(x0).toFixed(2)+" → local "+kind) : "not flat — keep sliding",
            flat?PAL.green:PAL.ink3,13,"left","700");
    }
    slider(controls,{label:"x",min:-2.6,max:2.6,val:x0,step:.02,fmt:function(v){return v.toFixed(2);},on:function(v){x0=v;draw();}});
    draw();
  });

  /* 9.4 — expectation is the balance point of the bars */
  reg("9.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var tilt=0;
    function draw(){
      var P=plot(C.ctx,600,340,-0.7,5.7,0,0.42);
      P.clear();
      P.title("Tilt the distribution — E[X] slides to keep it balanced");
      var w=[],s=0,i;
      for(i=0;i<6;i++){ var v=Math.exp(tilt*(i-2.5)); w.push(v); s+=v; }
      for(i=0;i<6;i++) w[i]/=s;
      var m=0,v2=0;
      for(i=0;i<6;i++) m+=i*w[i];
      for(i=0;i<6;i++) v2+=w[i]*(i-m)*(i-m);
      for(i=0;i<6;i++){
        var X=P.X(i)-24, Y=P.Y(w[i]);
        C.ctx.fillStyle=PAL.blueF; C.ctx.fillRect(X,Y,48,P.Y(0)-Y);
        C.ctx.strokeStyle=PAL.blue; C.ctx.lineWidth=1.2; C.ctx.strokeRect(X,Y,48,P.Y(0)-Y);
        P.txt(P.X(i),P.Y(0)+16,""+i,PAL.ink3,11,"center");
        P.txt(P.X(i),Y-7,w[i].toFixed(2),PAL.blue,10.5,"center","600");
      }
      P.axes();
      P.line(m,0,m,0.40,PAL.red,1.8,[4,3]);
      var bx=P.X(m), by=P.Y(0);
      C.ctx.fillStyle=PAL.red; C.ctx.beginPath();
      C.ctx.moveTo(bx,by+2); C.ctx.lineTo(bx-11,by+19); C.ctx.lineTo(bx+11,by+19); C.ctx.closePath(); C.ctx.fill();
      P.txt(16,324,"E[X] = "+m.toFixed(3),PAL.red,13,"left","700");
      P.txt(200,324,"Var = "+v2.toFixed(3)+"    sd = "+Math.sqrt(v2).toFixed(3),PAL.ink2,13,"left","600");
    }
    slider(controls,{label:"tilt",min:-1,max:1,val:tilt,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){tilt=x;draw();}});
    draw();
  });

  /* 11.1 — bias and variance on a dartboard */
  reg("11.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var bias=0.25, sd=0.25, shots=[];
    function reshoot(){ shots=[]; for(var i=0;i<12;i++) shots.push([_randn(),_randn()]); }
    reshoot();
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,-2.4,2.4,-2.4,2.4,{l:160,r:160,t:26,b:34});
      P.clear(); P.axes();
      P.title("Bias moves the cluster off-centre; variance spreads it out");
      [2.0,1.35,0.7].forEach(function(r,i){
        ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1;
        ctx.beginPath(); ctx.arc(P.X(0),P.Y(0),(P.X(r)-P.X(0)),0,7); ctx.stroke(); });
      ctx.fillStyle=PAL.amber; ctx.beginPath(); ctx.arc(P.X(0),P.Y(0),4,0,7); ctx.fill();
      var mse=bias*bias+sd*sd;
      shots.forEach(function(z){
        var x=bias+sd*z[0], y=bias*0.6+sd*z[1];
        ctx.fillStyle=PAL.red; ctx.beginPath(); ctx.arc(P.X(x),P.Y(y),4,0,7); ctx.fill(); });
      P.txt(16,324,"bias² = "+(bias*bias).toFixed(3)+"    variance = "+(sd*sd).toFixed(3),PAL.ink2,13,"left","600");
      P.txt(330,324,"MSE = bias² + var = "+mse.toFixed(3),PAL.red,13,"left","700");
    }
    slider(controls,{label:"bias",min:0,max:1.4,val:bias,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){bias=x;draw();}});
    slider(controls,{label:"variance",min:0.05,max:1,val:sd,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){sd=x;draw();}});
    button(controls,"Re-shoot",function(){ reshoot(); draw(); });
    draw();
  });


  /* 2.5 — one bias row, broadcast to every row */
  reg("2.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,320), controls=controlsIn(wrap);
    var rows=3;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,320,0,1,0,1); P.clear();
      P.title("Add more rows — the same bias row is reused, never copied by hand");
      var cw=30, ch=26, x0=40, y0=76;
      function grid(x,y,r,c,fill,stroke){ for(var i=0;i<r;i++) for(var j=0;j<c;j++){
        ctx.fillStyle=fill; ctx.fillRect(x+j*cw,y+i*ch,cw,ch);
        ctx.strokeStyle=stroke; ctx.lineWidth=1.1; ctx.strokeRect(x+j*cw,y+i*ch,cw,ch); } }
      grid(x0,y0,rows,4,PAL.blueF,PAL.blue);
      P.txt(x0+60,y0-12,"X  ("+rows+" × 4)",PAL.blue,12,"center","700");
      P.txt(x0+140,y0+rows*ch/2+5,"+",PAL.ink,20,"center","700");
      var bx=x0+168;
      grid(bx,y0+(rows-1)*ch/2,1,4,PAL.amberF,PAL.amber);
      P.txt(bx+60,y0-12,"b  (1 × 4)",PAL.amber,12,"center","700");
      for(var i=0;i<rows;i++){
        ctx.strokeStyle=PAL.amber; ctx.setLineDash([3,3]); ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(bx+60,y0+(rows-1)*ch/2+ch);
        ctx.lineTo(bx+60, y0+i*ch+ch/2); ctx.stroke(); ctx.setLineDash([]); }
      P.txt(x0+300,y0+rows*ch/2+5,"=",PAL.ink,20,"center","700");
      grid(x0+330,y0,rows,4,PAL.greenF,PAL.green);
      P.txt(x0+390,y0-12,"X + b",PAL.green,12,"center","700");
      P.txt(16,300,"still one bias vector — NumPy stretches it across all "+rows+" rows, no loop, no copy",PAL.ink3,12.5,"left");
    }
    slider(controls,{label:"rows",min:1,max:6,val:rows,step:1,fmt:function(x){return ""+x;},on:function(x){rows=x;draw();}});
    draw();
  });

  /* 3.2 — rank collapse: column space and null space */
  reg("3.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var k=0.6;
    function draw(){
      var P=plot(C.ctx,600,340,-3.4,3.4,-3.4,3.4,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      var c1=[1,2], c2=[k*1, k*2 + (1-k)*(-1)];   // k=1 → collinear with c1
      var det=c1[0]*c2[1]-c1[1]*c2[0], full=Math.abs(det)>0.08;
      P.title(full ? "rank 2 — the columns reach the whole plane"
                   : "rank 1 — outputs collapse onto a single line");
      if(!full){ var L=Math.hypot(c1[0],c1[1]);
        P.poly([[-3.2*c1[0]/L,-3.2*c1[1]/L],[3.2*c1[0]/L,3.2*c1[1]/L]],PAL.blue,2.2);
        var n=[c1[1],-c1[0]], nl=Math.hypot(n[0],n[1]);
        P.poly([[-3.2*n[0]/nl,-3.2*n[1]/nl],[3.2*n[0]/nl,3.2*n[1]/nl]],PAL.green,2,[5,4]);
        P.txt(P.X(2.4*n[0]/nl),P.Y(2.4*n[1]/nl),"null space",PAL.green,11.5,"center","700");
      } else P.fill([[-3.3,-3.3],[3.3,-3.3],[3.3,3.3],[-3.3,3.3]],"rgba(42,111,151,0.07)");
      P.arrow(0,0,c1[0],c1[1],PAL.blue,2.6);
      P.arrow(0,0,c2[0],c2[1],PAL.indigo,2.6);
      P.txt(16,324,"det = "+det.toFixed(3)+"    rank = "+(full?2:1),PAL.ink2,13,"left","600");
      P.txt(240,324, full?"null space = {0} only":"a whole line maps to zero", full?PAL.green:PAL.red,13,"left","700");
    }
    slider(controls,{label:"collapse",min:0,max:1,val:k,step:.02,fmt:function(x){return x.toFixed(2);},on:function(x){k=x;draw();}});
    draw();
  });

  /* 7.3 — integration by parts splits the rectangle */
  reg("7.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var u2=2.7;
    var g=function(u){return 0.35+0.26*u*u;};
    function draw(){
      var P=plot(C.ctx,600,340,-0.1,3.5,-0.1,3.5,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([0,1,2,3],[0,1,2,3]); P.axes(); 
      P.title("uv splits into ∫v du (under) + ∫u dv (left)");
      var u1=0.5, v1=g(u1), v2=g(u2), us=[];
      for(var u=u1;u<=u2;u+=0.03) us.push(u);
      var under=[[u1,0]].concat(us.map(function(u){return [u,g(u)];})).concat([[u2,0]]);
      P.fill(under,"rgba(42,111,151,0.22)");
      var left=[[0,v1]].concat(us.map(function(u){return [u,g(u)];})).concat([[0,v2]]);
      P.fill(left,"rgba(58,125,68,0.20)");
      P.poly([[0,v2],[u2,v2],[u2,0]],PAL.ink3,1.2);
      P.fn(g,0.12,3.2,PAL.indigo,2.4);
      var A=0,B=0;
      for(var i=0;i<us.length-1;i++){ var du=us[i+1]-us[i];
        A += g(us[i])*du; B += us[i]*(g(us[i+1])-g(us[i])); }
      P.txt(P.X(1.7),P.Y(0.45),"∫ v du",PAL.blue,13,"center","700");
      P.txt(P.X(0.4),P.Y(2.3),"∫ u dv",PAL.green,13,"left","700");
      P.txt(16,324,"∫v du = "+A.toFixed(3)+"   ∫u dv = "+B.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(300,324,"sum = "+(A+B).toFixed(3)+"   uv−u₁v₁ = "+(u2*v2-u1*v1).toFixed(3)+" ✓",PAL.indigo,13,"left","700");
    }
    slider(controls,{label:"upper u",min:1.2,max:3.1,val:u2,step:.05,fmt:function(x){return x.toFixed(2);},on:function(x){u2=x;draw();}});
    draw();
  });

  /* 9.1 — events as areas, and inclusion–exclusion */
  reg("9.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var d=95;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Slide the events apart — P(A∪B) = P(A) + P(B) − P(A∩B)");
      var R=74, cy=152, ax=300-d/2, bx=300+d/2;
      ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1.4; ctx.strokeRect(110,64,380,176);
      ctx.fillStyle=PAL.blue; ctx.globalAlpha=.26; ctx.beginPath(); ctx.arc(ax,cy,R,0,7); ctx.fill();
      ctx.fillStyle=PAL.green; ctx.globalAlpha=.26; ctx.beginPath(); ctx.arc(bx,cy,R,0,7); ctx.fill();
      ctx.globalAlpha=1;
      ctx.strokeStyle=PAL.blue; ctx.beginPath(); ctx.arc(ax,cy,R,0,7); ctx.stroke();
      ctx.strokeStyle=PAL.green; ctx.beginPath(); ctx.arc(bx,cy,R,0,7); ctx.stroke();
      var box=380*176, circ=Math.PI*R*R, dd=Math.abs(bx-ax), inter=0;
      if (dd<2*R){ var q=Math.acos(Math.min(1,dd/(2*R)));
        inter=2*R*R*q-0.5*dd*Math.sqrt(Math.max(0,4*R*R-dd*dd)); }
      var pA=circ/box, pB=circ/box, pI=inter/box, pU=pA+pB-pI;
      P.txt(ax-26,cy,"A",PAL.blue,15,"left","700");
      P.txt(bx+14,cy,"B",PAL.green,15,"left","700");
      P.txt(496,76,"Ω",PAL.ink3,13,"left","700");
      P.txt(16,286,"P(A) = "+pA.toFixed(3)+"    P(B) = "+pB.toFixed(3)+"    P(A∩B) = "+pI.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(16,310,"P(A∪B) = "+pA.toFixed(3)+" + "+pB.toFixed(3)+" − "+pI.toFixed(3)+" = "+pU.toFixed(3),PAL.indigo,13,"left","700");
    }
    slider(controls,{label:"separation",min:0,max:175,val:d,step:2,fmt:function(x){return ""+x;},on:function(x){d=x;draw();}});
    draw();
  });

  /* 10.5 — z = μ + σ·ε keeps the randomness in one place */
  reg("10.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var mu=1.4, sg=1.2, eps=0.8;
    function draw(){
      var P=plot(C.ctx,600,340,-5,6.5,-0.03,0.48);
      P.clear(); P.axes();
      P.title("One fixed ε; μ and σ move it — that's why gradients can flow");
      var n=function(x,m,s){return Math.exp(-Math.pow((x-m)/s,2)/2)/(s*Math.sqrt(2*Math.PI));};
      P.fn(function(x){return n(x,0,1);},-5,6.5,PAL.ink3,2,[5,3]);
      P.fn(function(x){return n(x,mu,sg);},-5,6.5,PAL.indigo,2.6);
      P.dot(eps,n(eps,0,1),PAL.ink3,5);
      var z=mu+sg*eps;
      P.dot(z,n(z,mu,sg),PAL.red,5.5);
      P.arrow(eps,n(eps,0,1)+0.03,z,n(z,mu,sg)+0.03,PAL.amber,1.8);
      P.txt(16,324,"ε = "+eps.toFixed(2)+"   →   z = μ + σ·ε = "+mu.toFixed(2)+" + "+sg.toFixed(2)+"×"+eps.toFixed(2)+" = "+z.toFixed(2),PAL.red,13,"left","700");
      P.txt(430,324,"∂z/∂μ = 1,  ∂z/∂σ = ε",PAL.ink2,12.5,"left","600");
    }
    slider(controls,{label:"μ",min:-2,max:4,val:mu,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){mu=x;draw();}});
    slider(controls,{label:"σ",min:0.3,max:2.2,val:sg,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){sg=x;draw();}});
    slider(controls,{label:"ε",min:-2,max:2,val:eps,step:.1,fmt:function(x){return x.toFixed(1);},on:function(x){eps=x;draw();}});
    draw();
  });

  /* 12.3 — mutual information is the shared middle */
  reg("12.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var dep=0.5;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("More dependence → more overlap → higher mutual information");
      var HX=1, HY=1, I=dep*Math.min(HX,HY);
      var R=78, sep=(1-dep)*(2*R)*0.98, cy=150, ax=300-sep/2, bx=300+sep/2;
      ctx.fillStyle=PAL.blue; ctx.globalAlpha=.26; ctx.beginPath(); ctx.arc(ax,cy,R,0,7); ctx.fill();
      ctx.fillStyle=PAL.green; ctx.globalAlpha=.26; ctx.beginPath(); ctx.arc(bx,cy,R,0,7); ctx.fill();
      ctx.globalAlpha=1;
      ctx.strokeStyle=PAL.blue; ctx.lineWidth=1.6; ctx.beginPath(); ctx.arc(ax,cy,R,0,7); ctx.stroke();
      ctx.strokeStyle=PAL.green; ctx.beginPath(); ctx.arc(bx,cy,R,0,7); ctx.stroke();
      P.txt(ax-40,cy,"H(X|Y)",PAL.blue,11.5,"left","700");
      P.txt(bx+2,cy,"H(Y|X)",PAL.green,11.5,"left","700");
      if (sep < 2*R) P.txt(300,cy,"I(X;Y)",PAL.ink,11.5,"center","700");
      P.txt(ax-16,60,"H(X)",PAL.blue,13,"left","700");
      P.txt(bx-16,60,"H(Y)",PAL.green,13,"left","700");
      P.txt(16,300,"I(X;Y) ≈ "+I.toFixed(3)+" bits",PAL.indigo,13.5,"left","700");
      P.txt(230,300, dep<0.05?"independent — knowing Y tells you nothing":(dep>0.95?"identical — Y determines X":"partly predictable"),PAL.ink3,12.5,"left");
    }
    slider(controls,{label:"dependence",min:0,max:1,val:dep,step:.02,fmt:function(x){return x.toFixed(2);},on:function(x){dep=x;draw();}});
    draw();
  });

  /* 13.1 — float spacing doubles every octave */
  reg("13.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,320), controls=controlsIn(wrap);
    var e=0;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,320,0,1,0,1); P.clear();
      P.title("Zoom to a power of two — the gap between neighbours doubles each time");
      var lo=Math.pow(2,e), hi=lo*2, x0=60, x1=540, y=150;
      ctx.strokeStyle=PAL.ink2; ctx.lineWidth=1.6;
      ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
      for (var i=0;i<=16;i++){
        var px=x0+(x1-x0)*i/16;
        ctx.strokeStyle=PAL.blue; ctx.lineWidth=1.3;
        ctx.beginPath(); ctx.moveTo(px,y-9); ctx.lineTo(px,y+9); ctx.stroke(); }
      P.txt(x0,y+30,lo.toString(),PAL.ink3,12,"center");
      P.txt(x1,y+30,hi.toString(),PAL.ink3,12,"center");
      var ulp=Math.pow(2,e-23);
      P.txt(16,238,"between "+lo+" and "+hi+", float32 steps by 2^("+e+"−23)",PAL.ink2,13,"left","600");
      P.txt(16,264,"gap = "+ulp.toExponential(3)+"   →  numbers this close are indistinguishable",PAL.red,13,"left","700");
      P.txt(16,290,"that's why 1e8 + 1 == 1e8 in float32",PAL.ink3,12,"left");
    }
    slider(controls,{label:"exponent",min:-10,max:27,val:e,step:1,fmt:function(x){return "2^"+x;},on:function(x){e=x;draw();}});
    draw();
  });

  /* 13.3 — which factorization, and what does it cost? */
  reg("13.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,330), controls=controlsIn(wrap);
    var n=500;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,330,0,1,0,1); P.clear();
      P.title("Same solve, different cost — pick the cheapest that fits your matrix");
      var rows=[["Cholesky (sym. pos-def)", n*n*n/3, PAL.green],
                ["LU (any square)",        2*n*n*n/3, PAL.blue],
                ["QR (least squares)",     4*n*n*n/3, PAL.amber]];
      var mx=rows[2][1], x0=210, y0=76, W=300, bh=34, gap=22;
      rows.forEach(function(r,i){
        var y=y0+i*(bh+gap), w=Math.max(3, r[1]/mx*W);
        ctx.fillStyle=r[2]; ctx.globalAlpha=.85; ctx.fillRect(x0,y,w,bh); ctx.globalAlpha=1;
        P.txt(x0-12,y+bh/2+1,r[0],r[2],12,"right","700");
        P.txt(x0+w+10,y+bh/2+1,(r[1]/1e9).toFixed(2)+" Gflop",PAL.ink2,12,"left","600"); });
      P.txt(16,300,"n = "+n+"   → Cholesky is 2× cheaper than LU, 4× cheaper than QR",PAL.ink3,12.5,"left","600");
    }
    slider(controls,{label:"matrix size n",min:50,max:2000,val:n,step:50,fmt:function(x){return ""+x;},on:function(x){n=x;draw();}});
    draw();
  });

  /* 13.4 — one forward sweep, one backward sweep */
  reg("13.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var x=1.2, w=0.8, b=-0.3;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Values flow right; gradients flow back by multiplying local slopes");
      var u=w*x, z=u+b, s=1/(1+Math.exp(-z)), L=0.5*s*s;
      var dL=s, ds=s*(1-s), dz=dL*ds, du=dz, dx=du*w, dw=du*x, db=dz;
      var xs=[70,205,340,475], y=132;
      var vals=[["x",x],["×w",u],["+b",z],["σ",s]];
      var grads=[dx,du,dz,dL];
      vals.forEach(function(v,i){
        ctx.fillStyle=[PAL.neutralF,PAL.blueF,PAL.blueF,PAL.greenF][i];
        ctx.strokeStyle=[PAL.ink2,PAL.blue,PAL.blue,PAL.green][i]; ctx.lineWidth=1.6;
        ctx.beginPath(); ctx.arc(xs[i],y,30,0,7); ctx.fill(); ctx.stroke();
        P.txt(xs[i],y+4,v[0],ctx.strokeStyle,13,"center","700");
        P.txt(xs[i],y-46,v[1].toFixed(3),PAL.ink,12,"center","600");
        P.txt(xs[i],y+54,"∂L/∂ = "+grads[i].toFixed(3),PAL.red,11.5,"center","600"); });
      for (var i=0;i<3;i++){
        ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1.8;
        ctx.beginPath(); ctx.moveTo(xs[i]+30,y-8); ctx.lineTo(xs[i+1]-30,y-8); ctx.stroke();
        ctx.strokeStyle=PAL.red; ctx.setLineDash([4,3]);
        ctx.beginPath(); ctx.moveTo(xs[i+1]-30,y+10); ctx.lineTo(xs[i]+30,y+10); ctx.stroke(); ctx.setLineDash([]); }
      P.txt(300,84,"forward  →",PAL.ink3,11.5,"center");
      P.txt(300,196,"←  backward",PAL.red,11.5,"center");
      P.txt(16,300,"L = ½σ(wx+b)²  =  "+L.toFixed(4),PAL.indigo,13,"left","700");
      P.txt(280,300,"∂L/∂w = "+dw.toFixed(4)+"    ∂L/∂b = "+db.toFixed(4),PAL.red,13,"left","700");
    }
    slider(controls,{label:"x",min:-2,max:2,val:x,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){x=v;draw();}});
    slider(controls,{label:"w",min:-2,max:2,val:w,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){w=v;draw();}});
    slider(controls,{label:"b",min:-2,max:2,val:b,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){b=v;draw();}});
    draw();
  });


  function _cell(ctx,x,y,w,h,fill,stroke){ ctx.fillStyle=fill; ctx.fillRect(x,y,w,h);
    ctx.strokeStyle=stroke; ctx.lineWidth=1; ctx.strokeRect(x,y,w,h); }

  /* 14.1 — the shape rule */
  reg("14.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,330), controls=controlsIn(wrap);
    var n=4, m=3;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,330,0,1,0,1); P.clear();
      P.title("Gradient copies the input's shape; a Jacobian is outputs × inputs");
      var s=22;
      function grid(x,y,r,c,fill,stroke,lab,sub){
        for(var i=0;i<r;i++) for(var j=0;j<c;j++) _cell(ctx,x+j*s,y+i*s,s,s,fill,stroke);
        P.txt(x+c*s/2,y-12,lab,stroke,12.5,"center","700");
        P.txt(x+c*s/2,y+r*s+18,sub,PAL.ink3,11,"center"); }
      grid(70,90,n,1,PAL.neutralF,PAL.ink2,"x","("+n+" × 1)");
      grid(180,90,n,1,PAL.blueF,PAL.blue,"∂L/∂x","("+n+" × 1) — same as x");
      grid(330,90,m,n,PAL.greenF,PAL.green,"J","("+m+" × "+n+") outputs × inputs");
      P.txt(16,296,"scalar loss → gradient has x's shape, so w −= lr·grad is elementwise",PAL.ink3,12.5,"left","600");
    }
    slider(controls,{label:"inputs n",min:1,max:8,val:n,step:1,fmt:function(v){return ""+v;},on:function(v){n=v;draw();}});
    slider(controls,{label:"outputs m",min:1,max:6,val:m,step:1,fmt:function(v){return ""+v;},on:function(v){m=v;draw();}});
    draw();
  });

  /* 14.2 — the gradient of xᵀAx */
  reg("14.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var x=[1.1,0.7], A=[[2,1],[0.4,1.5]];
    function q(v){ return A[0][0]*v[0]*v[0] + (A[0][1]+A[1][0])*v[0]*v[1] + A[1][1]*v[1]*v[1]; }
    function draw(){
      var P=plot(C.ctx,600,340,-2.6,2.6,-2.6,2.6,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      P.title("∇(xᵀAx) = (A + Aᵀ)x — only the symmetric part survives");
      [1,3,6].forEach(function(L,i){ var pts=[],seg=[];
        for(var t=0;t<=300;t++){ var a=t/300*2*Math.PI, cx=Math.cos(a), sy=Math.sin(a);
          var qq=q([cx,sy]); if(qq<=1e-6){ if(seg.length>1)pts.push(seg); seg=[]; continue; }
          var r=Math.sqrt(L/qq); if(r>3.4){ if(seg.length>1)pts.push(seg); seg=[]; continue; }
          seg.push([r*cx,r*sy]); }
        if(seg.length>1)pts.push(seg);
        pts.forEach(function(sg){P.poly(sg,[PAL.blueF,PAL.blue,PAL.indigo][i],1.8);}); });
      var g=[(A[0][0]+A[0][0])*x[0]+(A[0][1]+A[1][0])*x[1],
             (A[0][1]+A[1][0])*x[0]+(A[1][1]+A[1][1])*x[1]];
      var h=1e-5, gn=[(q([x[0]+h,x[1]])-q([x[0]-h,x[1]]))/(2*h),(q([x[0],x[1]+h])-q([x[0],x[1]-h]))/(2*h)];
      var gl=Math.hypot(g[0],g[1])||1;
      P.arrow(x[0],x[1],x[0]+g[0]/gl*1.1,x[1]+g[1]/gl*1.1,PAL.red,2.6);
      P.dot(x[0],x[1],PAL.red,5);
      P.txt(16,324,"analytic (A+Aᵀ)x = ("+g[0].toFixed(3)+", "+g[1].toFixed(3)+")",PAL.red,12.5,"left","700");
      P.txt(320,324,"numeric = ("+gn[0].toFixed(3)+", "+gn[1].toFixed(3)+") ✓",PAL.green,12.5,"left","600");
    }
    slider(controls,{label:"x₁",min:-2,max:2,val:x[0],step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){x[0]=v;draw();}});
    slider(controls,{label:"x₂",min:-2,max:2,val:x[1],step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){x[1]=v;draw();}});
    draw();
  });

  /* 14.3 — why reverse mode wins */
  reg("14.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,330), controls=controlsIn(wrap);
    var n=1000;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,330,0,1,0,1); P.clear();
      P.title("One scalar loss, n parameters — reverse mode gets every gradient at once");
      var fwd=n, rev=1, mx=Math.max(fwd,rev);
      var rows=[["forward mode","n passes = "+n, fwd, PAL.red],
                ["reverse mode (backprop)","1 pass", rev, PAL.green]];
      var x0=230, y0=100, W=290, bh=42, gap=34;
      rows.forEach(function(r,i){ var y=y0+i*(bh+gap);
        var w=Math.max(4, Math.log10(1+r[2])/Math.log10(1+mx)*W);
        ctx.fillStyle=r[3]; ctx.globalAlpha=.85; ctx.fillRect(x0,y,w,bh); ctx.globalAlpha=1;
        P.txt(x0-12,y+bh/2+1,r[0],r[3],12.5,"right","700");
        P.txt(x0+w+10,y+bh/2+1,r[1],PAL.ink2,12.5,"left","600"); });
      P.txt(16,278,"n = "+n+" parameters",PAL.ink,13,"left","700");
      P.txt(190,278,"reverse is "+n+"× cheaper — that's the whole reason backprop exists",PAL.green,12.5,"left","600");
    }
    slider(controls,{label:"parameters n",min:1,max:1000000,val:n,step:1,fmt:function(v){return v>=1000?(v/1000).toFixed(0)+"k":""+v;},on:function(v){n=v;draw();}});
    draw();
  });

  /* 14.4 — dW is an outer product */
  reg("14.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var d1=0.8, x1=1.2;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("dW = δ xᵀ — every weight's gradient is (its error) × (its input)");
      var delta=[d1,-0.5,0.9], xv=[x1,0.6];
      var s=40, x0=250, y0=92;
      P.txt(70,y0-14,"δ (error)",PAL.red,12.5,"center","700");
      delta.forEach(function(d,i){ _cell(ctx,50,y0+i*s,s,s,PAL.redF,PAL.red);
        P.txt(50+s/2,y0+i*s+s/2+4,d.toFixed(2),PAL.red,12,"center","600"); });
      P.txt(x0+ xv.length*s/2, y0-52,"xᵀ (input)",PAL.blue,12.5,"center","700");
      xv.forEach(function(v,j){ _cell(ctx,x0+j*s,y0-42,s,s,PAL.blueF,PAL.blue);
        P.txt(x0+j*s+s/2,y0-42+s/2+4,v.toFixed(2),PAL.blue,12,"center","600"); });
      delta.forEach(function(d,i){ xv.forEach(function(v,j){
        var g=d*v, mag=Math.min(1,Math.abs(g)/1.2);
        ctx.fillStyle = g>=0 ? "rgba(79,70,229,"+(0.12+0.5*mag)+")" : "rgba(192,57,43,"+(0.12+0.5*mag)+")";
        ctx.fillRect(x0+j*s,y0+i*s,s,s);
        ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1; ctx.strokeRect(x0+j*s,y0+i*s,s,s);
        P.txt(x0+j*s+s/2,y0+i*s+s/2+4,g.toFixed(2),PAL.ink,12,"center","600"); }); });
      P.txt(x0+xv.length*s+18, y0+delta.length*s/2, "= dW", PAL.indigo,14,"left","700");
      P.txt(16,300,"big error × big input → big weight update. db = δ (just the error).",PAL.ink3,12.5,"left","600");
    }
    slider(controls,{label:"δ₁",min:-1.5,max:1.5,val:d1,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){d1=v;draw();}});
    slider(controls,{label:"x₁",min:-1.5,max:1.5,val:x1,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){x1=v;draw();}});
    draw();
  });

  /* 14.5 — softmax + cross-entropy gradient is p − y */
  reg("14.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var z=[2.0,1.0,0.2], truth=0;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("The gradient of softmax + cross-entropy is just p − y");
      var mx=Math.max.apply(null,z), ex=z.map(function(v){return Math.exp(v-mx);});
      var s=ex.reduce(function(a,b){return a+b;},0), p=ex.map(function(v){return v/s;});
      var y=[0,0,0]; y[truth]=1;
      var g=p.map(function(v,i){return v-y[i];});
      var x0=90, bw=64, gap=52, base=232, H=130;
      for (var i=0;i<3;i++){
        var bx=x0+i*(bw+gap);
        ctx.fillStyle=PAL.blueF; ctx.fillRect(bx,base-p[i]*H,bw,p[i]*H);
        ctx.strokeStyle=PAL.blue; ctx.lineWidth=1.2; ctx.strokeRect(bx,base-p[i]*H,bw,p[i]*H);
        P.txt(bx+bw/2,base-p[i]*H-9,p[i].toFixed(3),PAL.blue,12,"center","700");
        P.txt(bx+bw/2,base+18,"class "+i+(truth===i?"  ✓":""),truth===i?PAL.green:PAL.ink3,11.5,"center",truth===i?"700":"400");
        P.txt(bx+bw/2,base+40,"grad "+(g[i]>=0?"+":"")+g[i].toFixed(3),g[i]>=0?PAL.red:PAL.green,12,"center","700");
      }
      ctx.strokeStyle=PAL.axis; ctx.beginPath(); ctx.moveTo(x0-16,base); ctx.lineTo(430,base); ctx.stroke();
      P.txt(452,150,"true class ↑ gets",PAL.ink3,11.5,"left");
      P.txt(452,168,"a negative grad",PAL.green,11.5,"left","700");
      P.txt(452,190,"(push it up);",PAL.ink3,11.5,"left");
      P.txt(452,208,"others positive",PAL.red,11.5,"left","700");
      P.txt(16,308,"loss = −log p[true] = "+(-Math.log(p[truth])).toFixed(4),PAL.indigo,13,"left","700");
    }
    slider(controls,{label:"z₀",min:-3,max:4,val:z[0],step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){z[0]=v;draw();}});
    slider(controls,{label:"z₁",min:-3,max:4,val:z[1],step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){z[1]=v;draw();}});
    button(controls,"Change true class",function(){ truth=(truth+1)%3; draw(); });
    draw();
  });

  /* 17.1 — gradients vanish with depth */
  reg("17.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var L=12, act=0;
    function draw(){
      var P=plot(C.ctx,600,340,0,20,-9,0.6);
      P.clear(); P.axes();
      var names=["sigmoid (max slope 0.25)","tanh (max slope 1)","ReLU (slope 1)"];
      var fac=[0.25,0.9,1.0][act];
      P.title("Gradient magnitude after L layers — "+names[act]);
      var pts=[];
      for(var l=0;l<=L;l++) pts.push([l, Math.log10(Math.pow(fac,l)+1e-12)]);
      P.poly(pts,[PAL.red,PAL.amber,PAL.green][act],2.8);
      pts.forEach(function(p){ if(p[1]>-9) P.dot(p[0],p[1],[PAL.red,PAL.amber,PAL.green][act],3.5); });
      P.line(0,-6,20,-6,PAL.ink3,1,[4,3]);
      P.txt(P.X(14),P.Y(-5.6),"effectively zero",PAL.ink3,11,"left");
      var g=Math.pow(fac,L);
      P.txt(16,324,"after "+L+" layers, gradient × "+g.toExponential(2),
            g<1e-4?PAL.red:PAL.ink2,13,"left","700");
      P.txt(330,324, act===0?"this is why deep sigmoid nets stalled":"why ReLU/residuals unlocked depth",
            act===0?PAL.red:PAL.green,12.5,"left","600");
    }
    slider(controls,{label:"depth L",min:1,max:20,val:L,step:1,fmt:function(v){return ""+v;},on:function(v){L=v;draw();}});
    button(controls,"Switch activation",function(){ act=(act+1)%3; draw(); });
    draw();
  });

  /* 17.2 — convolution is a banded matrix multiply */
  reg("17.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var k=3, n=8;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("A convolution is a matrix with the same kernel repeated down the diagonals");
      var s=26, x0=150, y0=76, rows=n-k+1;
      for (var i=0;i<rows;i++) for (var j=0;j<n;j++){
        var inBand = (j>=i && j<i+k);
        _cell(ctx,x0+j*s,y0+i*s,s,s, inBand?PAL.blueF:"#fff", inBand?PAL.blue:PAL.line);
        if (inBand) P.txt(x0+j*s+s/2,y0+i*s+s/2+4,"w"+(j-i),PAL.blue,10.5,"center","600"); }
      P.txt(x0+n*s/2,y0-14,"Toeplitz matrix ("+rows+" × "+n+")",PAL.blue,12.5,"center","700");
      P.txt(16,300,"kernel size "+k+" → each output row sees "+k+" inputs; weights are SHARED, not "+(rows*n)+" free numbers",PAL.ink3,12,"left","600");
    }
    slider(controls,{label:"kernel k",min:1,max:5,val:k,step:1,fmt:function(v){return ""+v;},on:function(v){k=v;draw();}});
    slider(controls,{label:"input n",min:4,max:12,val:n,step:1,fmt:function(v){return ""+v;},on:function(v){n=v;draw();}});
    draw();
  });

  /* 17.3 — normalization recentres and rescales */
  reg("17.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var shift=2.5, scale=2.0;
    var base=[-1.2,-0.4,0.1,0.6,1.1,-0.8,0.9,0.3];
    function draw(){
      var P=plot(C.ctx,600,340,-8,8,-1,1.6);
      P.clear(); P.axes();
      P.title("LayerNorm: subtract the mean, divide by the spread → always centred");
      var raw=base.map(function(v){return v*scale+shift;});
      var m=raw.reduce(function(a,b){return a+b;},0)/raw.length;
      var sd=Math.sqrt(raw.reduce(function(a,b){return a+(b-m)*(b-m);},0)/raw.length)||1e-9;
      var nrm=raw.map(function(v){return (v-m)/sd;});
      raw.forEach(function(v){ P.dot(v,1.0,PAL.red,5); });
      nrm.forEach(function(v){ P.dot(v,0.2,PAL.green,5); });
      P.txt(-7.6,1.0,"before",PAL.red,12,"left","700");
      P.txt(-7.6,0.2,"after",PAL.green,12,"left","700");
      P.line(m,0.85,m,1.15,PAL.red,1.6,[3,3]);
      P.line(0,0.05,0,0.35,PAL.green,1.6,[3,3]);
      P.txt(16,324,"before: mean "+m.toFixed(2)+", sd "+sd.toFixed(2),PAL.red,13,"left","600");
      P.txt(290,324,"after: mean 0.00, sd 1.00  → stable inputs for the next layer",PAL.green,12.5,"left","700");
    }
    slider(controls,{label:"shift",min:-5,max:5,val:shift,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){shift=v;draw();}});
    slider(controls,{label:"scale",min:0.2,max:4,val:scale,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){scale=v;draw();}});
    draw();
  });

  /* 17.4 — attention weights from QK dot products */
  reg("17.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var qa=30, dk=4;
    var keys=[[1,0.2],[0.6,0.8],[-0.3,1],[-1,0.1]];
    var labels=["the","cat","sat","mat"];
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Attention: query · key → softmax → how much each token is read");
      var t=qa*Math.PI/180, q=[Math.cos(t),Math.sin(t)];
      var sc=keys.map(function(k){ return (q[0]*k[0]+q[1]*k[1])/Math.sqrt(dk); });
      var mx=Math.max.apply(null,sc), ex=sc.map(function(v){return Math.exp(v-mx);});
      var s=ex.reduce(function(a,b){return a+b;},0), w=ex.map(function(v){return v/s;});
      // mini vector plot
      var cx=118, cy=150, R=62;
      ctx.strokeStyle=PAL.line; ctx.lineWidth=1; ctx.beginPath(); ctx.arc(cx,cy,R,0,7); ctx.stroke();
      keys.forEach(function(k,i){ var L=Math.hypot(k[0],k[1]);
        ctx.strokeStyle=PAL.blue; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(cx,cy);
        ctx.lineTo(cx+k[0]/L*R, cy-k[1]/L*R); ctx.stroke();
        P.txt(cx+k[0]/L*(R+14), cy-k[1]/L*(R+14)+4, labels[i], PAL.blue,10.5,"center","600"); });
      ctx.strokeStyle=PAL.red; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(cx,cy);
      ctx.lineTo(cx+q[0]*R, cy-q[1]*R); ctx.stroke();
      P.txt(cx, cy+R+30, "query", PAL.red, 12, "center","700");
      // weight bars
      var x0=280, y0=86, bw=210;
      w.forEach(function(v,i){ var y=y0+i*40;
        ctx.fillStyle=PAL.line; ctx.fillRect(x0,y,bw,22);
        ctx.fillStyle=PAL.indigo; ctx.globalAlpha=.85; ctx.fillRect(x0,y,bw*v,22); ctx.globalAlpha=1;
        P.txt(x0-10,y+16,labels[i],PAL.ink2,12,"right","700");
        P.txt(x0+bw+10,y+16,(100*v).toFixed(0)+"%",PAL.indigo,12,"left","700"); });
      P.txt(16,300,"scores ÷ √d  (d = "+dk+") keeps the softmax from saturating as dimension grows",PAL.ink3,12,"left","600");
    }
    slider(controls,{label:"query angle",min:0,max:360,val:qa,step:2,fmt:function(v){return v+"°";},on:function(v){qa=v;draw();}});
    slider(controls,{label:"d (scaling)",min:1,max:64,val:dk,step:1,fmt:function(v){return ""+v;},on:function(v){dk=v;draw();}});
    draw();
  });

  /* 17.5 — heads split the budget */
  reg("17.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var heads=4, dmodel=64;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("Multi-head: the same width, sliced into independent views");
      var dh=Math.floor(dmodel/heads);
      var x0=60, y0=110, W=480, H=54;
      for (var i=0;i<heads;i++){
        var w=W/heads;
        ctx.fillStyle=["#dbe8fb","#e3efe5","#f7e7d2","#e5e7ff","#f7e0dc","#ece6ff","#d8f3f0","#fde8f3"][i%8];
        ctx.fillRect(x0+i*w,y0,w-3,H);
        ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1; ctx.strokeRect(x0+i*w,y0,w-3,H);
        P.txt(x0+i*w+(w-3)/2, y0+H/2+4, "head "+(i+1), PAL.ink2, 11.5,"center","700");
        P.txt(x0+i*w+(w-3)/2, y0+H+18, "d="+dh, PAL.ink3, 11,"center"); }
      P.txt(x0+W/2, y0-16, "d_model = "+dmodel, PAL.indigo, 13,"center","700");
      P.txt(16,246,heads+" heads × "+dh+" dims = "+(heads*dh)+" — same total width, "+heads+" separate attention patterns",PAL.ink2,12.5,"left","600");
      P.txt(16,274,"each head learns its own relation (syntax, coreference, position…)",PAL.ink3,12,"left");
      P.txt(16,302,"cost is unchanged: splitting is free, the views are not",PAL.green,12.5,"left","700");
    }
    slider(controls,{label:"heads",min:1,max:8,val:heads,step:1,fmt:function(v){return ""+v;},on:function(v){heads=v;draw();}});
    slider(controls,{label:"d_model",min:16,max:128,val:dmodel,step:16,fmt:function(v){return ""+v;},on:function(v){dmodel=v;draw();}});
    draw();
  });


  function _lgam(z){ var g=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
    if(z<0.5) return Math.log(Math.PI/Math.sin(Math.PI*z))-_lgam(1-z);
    z-=1; var x=0.99999999999980993; for(var i=0;i<8;i++) x+=g[i]/(z+i+1);
    var t=z+7.5; return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(x); }
  function _beta(a,b){ return function(x){ return (x<=0||x>=1)?0:
    Math.exp((a-1)*Math.log(x)+(b-1)*Math.log(1-x)+_lgam(a+b)-_lgam(a)-_lgam(b)); }; }

  /* 15.1 — conjugate updating: Beta in, Beta out */
  reg("15.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var a0=2,b0=2,k=6,n=10;
    function draw(){
      var P=plot(C.ctx,600,340,-0.02,1.02,-0.3,5.2);
      P.clear(); P.grid([0,.25,.5,.75,1],[0,1,2,3,4,5]); P.axes();
      P.title("Prior + data = posterior, and it stays a Beta (that's conjugacy)");
      P.fn(function(x){return Math.min(_beta(a0,b0)(x),5.1);},0.002,0.998,PAL.green,2.2);
      P.fn(function(x){return Math.min(_beta(a0+k,b0+n-k)(x),5.1);},0.002,0.998,PAL.indigo,2.9);
      var pm=(a0+k)/(a0+b0+n);
      P.line(pm,0,pm,5.0,PAL.red,1.4,[4,3]);
      P.txt(16,324,"prior Beta("+a0+","+b0+")  +  "+k+"/"+n+" heads",PAL.green,12.5,"left","600");
      P.txt(300,324,"posterior Beta("+(a0+k)+","+(b0+n-k)+")   mean "+pm.toFixed(3),PAL.indigo,12.5,"left","700");
    }
    slider(controls,{label:"prior a",min:1,max:12,val:a0,step:1,fmt:function(v){return ""+v;},on:function(v){a0=v;draw();}});
    slider(controls,{label:"prior b",min:1,max:12,val:b0,step:1,fmt:function(v){return ""+v;},on:function(v){b0=v;draw();}});
    slider(controls,{label:"heads k",min:0,max:40,val:k,step:1,fmt:function(v){return ""+v;},on:function(v){k=Math.min(v,n);draw();}});
    slider(controls,{label:"flips n",min:1,max:40,val:n,step:1,fmt:function(v){return ""+v;},on:function(v){n=v;if(k>n)k=n;draw();}});
    draw();
  });

  /* 15.2 — Bayesian regression: a band, not a line */
  reg("15.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var N=6, noise=0.5, pts=[];
    function gen(){ pts=[]; for(var i=0;i<20;i++){ var x=-2.5+5*i/19; pts.push([x, 0.8*x+0.4]); } }
    gen();
    function draw(){
      var P=plot(C.ctx,600,340,-3,3,-3,3);
      P.clear(); P.grid([-2,-1,0,1,2],[-2,-1,0,1,2]); P.axes();
      P.title("Few points → wide uncertainty band; more data narrows it");
      var used=pts.slice(0,N);
      var sx=0,sy=0,sxx=0,sxy=0,n=used.length;
      used.forEach(function(p){ sx+=p[0]; sy+=p[1]; sxx+=p[0]*p[0]; sxy+=p[0]*p[1]; });
      var den=(n*sxx-sx*sx)||1e-9, m=(n*sxy-sx*sy)/den, c=(sy-m*sx)/n;
      var band=noise*Math.sqrt(1/n+0)*2.2+0.25;
      var up=[],dn=[];
      for(var x=-3;x<=3;x+=0.1){ var w=band*(1+Math.abs(x)/2.5); up.push([x,m*x+c+w]); dn.push([x,m*x+c-w]); }
      P.fill(up.concat(dn.reverse()),"rgba(79,70,229,0.16)");
      P.poly([[-3,m*-3+c],[3,m*3+c]],PAL.indigo,2.6);
      used.forEach(function(p){ P.dot(p[0],p[1]+0,PAL.blue,4.5); });
      P.txt(16,324,"n = "+n+" points   slope ≈ "+m.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(300,324,"band width ≈ ±"+band.toFixed(2)+"  → shrinks like 1/√n",PAL.indigo,12.5,"left","700");
    }
    slider(controls,{label:"data points",min:2,max:20,val:N,step:1,fmt:function(v){return ""+v;},on:function(v){N=v;draw();}});
    slider(controls,{label:"noise",min:0.1,max:1.2,val:noise,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){noise=v;draw();}});
    draw();
  });

  /* 15.3 — Metropolis-Hastings in action */
  reg("15.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var step=0.8;
    function target(x){ return Math.exp(-Math.pow(x+1.2,2)/0.6) + 1.3*Math.exp(-Math.pow(x-1.4,2)/0.8); }
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-0.1,1.9);
      P.clear(); P.axes();
      P.title("Propose a jump, accept if it's better (or sometimes anyway)");
      var x=0, acc=0, samples=[];
      for(var i=0;i<4000;i++){
        var xp=x+step*_randn(), r=target(xp)/target(x);
        if (Math.random()<r){ x=xp; acc++; }
        if(i>300) samples.push(x);
      }
      var bins=48, cnt=new Array(bins).fill(0);
      samples.forEach(function(v){ var b=Math.floor((v+4)/8*bins); if(b>=0&&b<bins) cnt[b]++; });
      var mx=Math.max.apply(null,cnt)||1;
      for(var b=0;b<bins;b++){ var h=cnt[b]/mx*1.55, x0=-4+8*b/bins;
        C.ctx.fillStyle=PAL.indigo; C.ctx.globalAlpha=.55;
        C.ctx.fillRect(P.X(x0),P.Y(h),(P.X(8/bins)-P.X(0)),P.Y(0)-P.Y(h)); }
      C.ctx.globalAlpha=1;
      var tmax=0; for(var t=-4;t<=4;t+=0.02) tmax=Math.max(tmax,target(t));
      P.fn(function(v){return target(v)/tmax*1.55;},-4,4,PAL.red,2.6);
      P.txt(16,324,"proposal width = "+step.toFixed(2)+"   acceptance "+(100*acc/4000).toFixed(0)+"%",PAL.ink2,13,"left","600");
      P.txt(340,324, (acc/4000>0.15&&acc/4000<0.6)?"healthy mixing ✓":"too "+(acc/4000>0.6?"timid":"bold")+" — tune it",
            (acc/4000>0.15&&acc/4000<0.6)?PAL.green:PAL.red,12.5,"left","700");
    }
    slider(controls,{label:"proposal width",min:0.05,max:4,val:step,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){step=v;draw();}});
    button(controls,"Re-run chain",function(){ draw(); });
    draw();
  });

  /* 15.4 — the ELBO gap is the KL you can't see */
  reg("15.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var qm=0.4, qs=1.4;
    function draw(){
      var P=plot(C.ctx,600,340,-5,5,-0.03,0.62);
      P.clear(); P.axes();
      P.title("Move q onto p — the KL gap closes and the ELBO rises to log p(x)");
      var n=function(x,m,s){return Math.exp(-Math.pow((x-m)/s,2)/2)/(s*Math.sqrt(2*Math.PI));};
      P.fn(function(x){return n(x,-0.6,1.0);},-5,5,PAL.indigo,2.6);
      P.fn(function(x){return n(x,qm,qs);},-5,5,PAL.green,2.4,[5,3]);
      var kl=Math.log(1.0/qs)+(qs*qs+Math.pow(qm+0.6,2))/(2*1.0)-0.5;
      P.txt(16,324,"KL(q‖p) = "+kl.toFixed(4),kl<0.02?PAL.green:PAL.red,13,"left","700");
      P.txt(230,324,"ELBO = log p(x) − KL   → maximise ELBO = shrink KL",PAL.ink2,12.5,"left","600");
      P.txt(P.X(-0.6),P.Y(0.42),"p (true)",PAL.indigo,12,"center","700");
      P.txt(P.X(qm),P.Y(n(qm,qm,qs)+0.04),"q",PAL.green,12,"center","700");
    }
    slider(controls,{label:"q mean",min:-3,max:3,val:qm,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){qm=v;draw();}});
    slider(controls,{label:"q spread",min:0.3,max:3,val:qs,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){qs=v;draw();}});
    draw();
  });

  /* 15.5 — Gaussian process: smoothness you choose */
  reg("15.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var ell=1.0;
    var obs=[[-2,-0.6],[0,0.9],[1.6,-0.3]];
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-2.4,2.4);
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-2,-1,0,1,2]); P.axes();
      P.title("Length-scale sets how fast the function may wiggle");
      function k(a,b){ return Math.exp(-Math.pow(a-b,2)/(2*ell*ell)); }
      var n=obs.length, K=[], y=[];
      for(var i=0;i<n;i++){ K.push([]); y.push(obs[i][1]);
        for(var j=0;j<n;j++) K[i].push(k(obs[i][0],obs[j][0])+(i===j?0.02:0)); }
      // solve K a = y (3x3 gaussian elimination)
      var M=K.map(function(r,i){return r.concat([y[i]]);});
      for(i=0;i<n;i++){ var p=M[i][i]||1e-9;
        for(j=i;j<=n;j++) M[i][j]/=p;
        for(var r2=0;r2<n;r2++) if(r2!==i){ var f=M[r2][i];
          for(j=i;j<=n;j++) M[r2][j]-=f*M[i][j]; } }
      var al=M.map(function(r){return r[n];});
      var mean=[],up=[],dn=[];
      for(var x=-4;x<=4;x+=0.05){
        var m=0, kk=[]; for(i=0;i<n;i++){ kk.push(k(x,obs[i][0])); m+=kk[i]*al[i]; }
        var v=1; for(i=0;i<n;i++) v-=kk[i]*kk[i]*0.55;
        v=Math.max(0.02,v); var s=Math.sqrt(v)*1.1;
        mean.push([x,m]); up.push([x,m+s]); dn.push([x,m-s]); }
      P.fill(up.concat(dn.reverse()),"rgba(79,70,229,0.15)");
      P.poly(mean,PAL.indigo,2.6);
      obs.forEach(function(o){ P.dot(o[0],o[1],PAL.red,5); });
      P.txt(16,324,"length-scale ℓ = "+ell.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(250,324, ell<0.6?"twitchy — trusts only nearby points":(ell>2?"very smooth — long-range influence":"balanced"),PAL.indigo,12.5,"left","700");
    }
    slider(controls,{label:"length-scale",min:0.2,max:3,val:ell,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){ell=v;draw();}});
    draw();
  });

  /* 16.1 — VAE: reconstruction vs KL */
  reg("16.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var beta=1, qs=0.7;
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-0.03,0.62);
      P.clear(); P.axes();
      P.title("VAE loss = reconstruction + β·KL(q ‖ N(0,1)) — a tug of war");
      var n=function(x,m,s){return Math.exp(-Math.pow((x-m)/s,2)/2)/(s*Math.sqrt(2*Math.PI));};
      P.fn(function(x){return n(x,0,1);},-4,4,PAL.ink3,2,[5,3]);
      P.fn(function(x){return n(x,0.6,qs);},-4,4,PAL.indigo,2.6);
      var kl=Math.log(1/qs)+(qs*qs+0.36)/2-0.5;
      var rec=1.4/(qs+0.2);
      P.txt(16,324,"KL = "+kl.toFixed(3)+"   ×β("+beta.toFixed(1)+") = "+(beta*kl).toFixed(3),PAL.green,12.5,"left","600");
      P.txt(280,324,"reconstruction ≈ "+rec.toFixed(2)+"   total = "+(rec+beta*kl).toFixed(3),PAL.indigo,12.5,"left","700");
      P.txt(P.X(0),P.Y(0.45),"prior N(0,1)",PAL.ink3,11.5,"center");
      P.txt(P.X(0.6),P.Y(n(0.6,0.6,qs)+0.04),"q(z|x)",PAL.indigo,12,"center","700");
    }
    slider(controls,{label:"β",min:0,max:4,val:beta,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){beta=v;draw();}});
    slider(controls,{label:"q spread",min:0.2,max:2.5,val:qs,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){qs=v;draw();}});
    draw();
  });

  /* 16.2 — GAN: two players converging */
  reg("16.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var gm=-1.6;
    function draw(){
      var P=plot(C.ctx,600,340,-4,4,-0.05,0.72);
      P.clear(); P.axes();
      P.title("Generator chases the data; the discriminator's edge shrinks to a coin-flip");
      var n=function(x,m,s){return Math.exp(-Math.pow((x-m)/s,2)/2)/(s*Math.sqrt(2*Math.PI));};
      var pd=function(x){return n(x,1,0.8);}, pg=function(x){return n(x,gm,0.8);};
      P.fn(pd,-4,4,PAL.blue,2.6); P.fn(pg,-4,4,PAL.red,2.6);
      P.fn(function(x){ var a=pd(x),b=pg(x); return 0.62*(a/(a+b+1e-9)); },-4,4,PAL.green,2.2,[5,3]);
      var overlap=0; for(var x=-4;x<=4;x+=0.02) overlap+=Math.min(pd(x),pg(x))*0.02;
      P.txt(16,324,"generator mean "+gm.toFixed(2)+"   (data at 1.00)",PAL.red,12.5,"left","600");
      P.txt(300,324,"overlap = "+(100*overlap).toFixed(0)+"%"+(overlap>0.85?"  → D is guessing ✓":"  → D still wins"),
            overlap>0.85?PAL.green:PAL.ink3,12.5,"left","700");
      P.txt(P.X(-3.4),P.Y(0.55),"— D(x)",PAL.green,11.5,"left","700");
    }
    slider(controls,{label:"generator",min:-3,max:3,val:gm,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){gm=v;draw();}});
    draw();
  });

  /* 16.3 — forward diffusion destroys structure */
  reg("16.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var t=0.3, seed=[];
    for (var i=0;i<160;i++){ var a=i/160*2*Math.PI; seed.push([1.5*Math.cos(a)+0.06*_randn(), 1.5*Math.sin(a)+0.06*_randn(), _randn(), _randn()]); }
    function draw(){
      var P=plot(C.ctx,600,340,-3.6,3.6,-3.6,3.6,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("q(xₜ|x₀): keep √ᾱ of the data, add √(1−ᾱ) of pure noise");
      var ab=Math.max(1e-4,1-t), sa=Math.sqrt(ab), sn=Math.sqrt(1-ab);
      seed.forEach(function(p){
        var x=sa*p[0]+sn*p[2], y=sa*p[1]+sn*p[3];
        if(Math.abs(x)<3.5&&Math.abs(y)<3.5){
          C.ctx.fillStyle=PAL.indigo; C.ctx.globalAlpha=.65;
          C.ctx.beginPath(); C.ctx.arc(P.X(x),P.Y(y),2.4,0,7); C.ctx.fill(); } });
      C.ctx.globalAlpha=1;
      P.txt(16,324,"t = "+t.toFixed(2)+"    ᾱ = "+ab.toFixed(3),PAL.ink2,13,"left","600");
      P.txt(260,324, t<0.15?"still clearly a ring":(t>0.85?"pure Gaussian noise — structure gone":"melting away"),PAL.indigo,12.5,"left","700");
    }
    slider(controls,{label:"time t",min:0,max:1,val:t,step:.02,fmt:function(v){return v.toFixed(2);},on:function(v){t=v;draw();}});
    draw();
  });

  /* 16.4 — the score points uphill in density */
  reg("16.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var sig=0.8;
    function draw(){
      var P=plot(C.ctx,600,340,-3.4,3.4,-0.05,0.62);
      P.clear(); P.axes();
      P.title("The score ∇ log p points back toward the data — that's what we learn");
      var n=function(x){return 0.6*Math.exp(-Math.pow(x+1.2,2)/(2*sig*sig))/(sig*2.5)
                            +0.4*Math.exp(-Math.pow(x-1.3,2)/(2*sig*sig))/(sig*2.5);};
      P.fn(n,-3.4,3.4,PAL.indigo,2.6);
      for(var x=-3;x<=3;x+=0.5){
        var h=1e-4, s=(Math.log(n(x+h)+1e-12)-Math.log(n(x-h)+1e-12))/(2*h);
        var L=Math.max(-0.55,Math.min(0.55,s*0.12));
        P.arrow(x,0.045,x+L,0.045,PAL.red,2,6); }
      P.txt(16,324,"noise σ = "+sig.toFixed(2),PAL.ink2,13,"left","600");
      P.txt(200,324,"arrows = ∇ₓ log p(x): denoising just follows them uphill",PAL.red,12.5,"left","700");
    }
    slider(controls,{label:"noise σ",min:0.3,max:2,val:sig,step:.05,fmt:function(v){return v.toFixed(2);},on:function(v){sig=v;draw();}});
    draw();
  });

  /* 16.5 — reverse sampling, step by step */
  reg("16.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var steps=8;
    function draw(){
      var P=plot(C.ctx,600,340,-3.6,3.6,-3.6,3.6,{l:160,r:160,t:26,b:34});
      P.clear(); P.grid([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]); P.axes();
      P.title("Start from noise, denoise a little at a time — the ring reappears");
      var N=170, rng=0;
      function rnd(){ rng=(rng*9301+49297)%233280; return rng/233280; }
      function gn(){ return Math.sqrt(-2*Math.log(rnd()+1e-9))*Math.cos(2*Math.PI*rnd()); }
      for (var i=0;i<N;i++){
        var x=gn()*1.6, y=gn()*1.6;
        for (var s=0;s<steps;s++){
          var r=Math.hypot(x,y)||1e-6, pull=(1.5-r)*0.30;
          x+= x/r*pull; y+= y/r*pull;
          var jitter=0.35*Math.pow(1-s/Math.max(1,steps),1.5);
          x+=gn()*jitter; y+=gn()*jitter; }
        if(Math.abs(x)<3.5&&Math.abs(y)<3.5){
          C.ctx.fillStyle=PAL.indigo; C.ctx.globalAlpha=.65;
          C.ctx.beginPath(); C.ctx.arc(P.X(x),P.Y(y),2.4,0,7); C.ctx.fill(); } }
      C.ctx.globalAlpha=1;
      P.txt(16,324,"denoising steps = "+steps,PAL.ink2,13,"left","600");
      P.txt(230,324, steps<3?"too few — still a blob":(steps>12?"clean ring — more steps, better samples":"taking shape"),PAL.indigo,12.5,"left","700");
    }
    slider(controls,{label:"steps",min:1,max:25,val:steps,step:1,fmt:function(v){return ""+v;},on:function(v){steps=v;draw();}});
    draw();
  });

  /* C.1 — how many components do you actually need? */
  reg("C.1", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var kc=10;
    var ev=[]; for (var i=0;i<40;i++) ev.push(Math.exp(-i/6));
    var tot=ev.reduce(function(a,b){return a+b;},0);
    function draw(){
      var P=plot(C.ctx,600,340,0,40,0,1.05);
      P.clear(); P.grid([0,10,20,30,40],[0,0.25,0.5,0.75,1]); P.axes();
      P.title("Variance explained — a few components carry most of the picture");
      var cum=[],run=0;
      for (var i=0;i<40;i++){ run+=ev[i]/tot; cum.push([i+1,run]); }
      P.poly(cum,PAL.indigo,2.8);
      for (i=0;i<40;i++){ var h=ev[i]/tot*3.2;
        C.ctx.fillStyle= i<kc?PAL.blue:PAL.line; C.ctx.globalAlpha=.7;
        C.ctx.fillRect(P.X(i+0.6),P.Y(h),(P.X(0.8)-P.X(0)),P.Y(0)-P.Y(h)); }
      C.ctx.globalAlpha=1;
      var got=cum[kc-1][1];
      P.line(kc,0,kc,got,PAL.red,1.6,[4,3]); P.dot(kc,got,PAL.red,5);
      P.txt(16,324,"k = "+kc+" components",PAL.ink2,13,"left","600");
      P.txt(200,324,"keeps "+(100*got).toFixed(1)+"% of the variance — dropping the rest costs little",PAL.indigo,12.5,"left","700");
    }
    slider(controls,{label:"components k",min:1,max:40,val:kc,step:1,fmt:function(v){return ""+v;},on:function(v){kc=v;draw();}});
    draw();
  });

  /* C.2 — three optimizers on one valley */
  reg("C.2", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var eta=0.12, b=9;
    function run(kind){ var x=[4.6,1.5], v=[0,0], m=[0,0], s=[0,0], path=[[x[0],x[1]]];
      for (var i=1;i<=44;i++){ var g=[x[0], b*x[1]];
        if (kind===0) x=[x[0]-eta*g[0], x[1]-eta*g[1]];
        else if (kind===1){ v=[0.85*v[0]-eta*g[0], 0.85*v[1]-eta*g[1]]; x=[x[0]+v[0],x[1]+v[1]]; }
        else { for(var d=0;d<2;d++){ m[d]=0.9*m[d]+0.1*g[d]; s[d]=0.999*s[d]+0.001*g[d]*g[d];
              var mh=m[d]/(1-Math.pow(0.9,i)), sh=s[d]/(1-Math.pow(0.999,i));
              x[d]-= (eta*2.2)*mh/(Math.sqrt(sh)+1e-8); } }
        if(!isFinite(x[0])||Math.abs(x[0])>9||Math.abs(x[1])>4) break;
        path.push([x[0],x[1]]); }
      return path; }
    function draw(){
      var P=plot(C.ctx,600,340,-5.2,5.2,-2.6,2.6);
      P.clear(); P.grid([-4,-2,0,2,4],[-2,-1,0,1,2]); P.axes();
      P.title("Same valley, three optimizers — watch who reaches the middle first");
      [1.2,4.5,10].forEach(function(c,i){ var pts=[];
        for(var t=0;t<=140;t++){var a=t/140*2*Math.PI;
          pts.push([Math.sqrt(2*c)*Math.cos(a), Math.sqrt(2*c/b)*Math.sin(a)]);}
        P.poly(pts,[PAL.indigo,PAL.blue,PAL.blueF][i],1.5); });
      [[0,PAL.red,"GD"],[1,PAL.amber,"momentum"],[2,PAL.green,"Adam"]].forEach(function(o){
        var pa=run(o[0]); P.poly(pa,o[1],1.9);
        P.dot(pa[pa.length-1][0],pa[pa.length-1][1],o[1],4); });
      P.txt(16,324,"— GD",PAL.red,12.5,"left","700");
      P.txt(90,324,"— momentum",PAL.amber,12.5,"left","700");
      P.txt(230,324,"— Adam",PAL.green,12.5,"left","700");
      P.txt(340,324,"η = "+eta.toFixed(2)+(eta*b>2?"  GD diverging":""),eta*b>2?PAL.red:PAL.ink2,12.5,"left","600");
    }
    slider(controls,{label:"η",min:0.02,max:0.22,val:eta,step:.01,fmt:function(v){return v.toFixed(2);},on:function(v){eta=v;draw();}});
    draw();
  });

  /* C.3 — a Bayesian classifier's boundary */
  reg("C.3", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var prior=0.5;
    function draw(){
      var P=plot(C.ctx,600,340,-5,5,-0.03,0.52);
      P.clear(); P.axes();
      P.title("Shift the prior — the decision boundary slides, even with the same data");
      var n=function(x,m){return Math.exp(-Math.pow(x-m,2)/2)/Math.sqrt(2*Math.PI);};
      P.fn(function(x){return prior*n(x,-1.3);},-5,5,PAL.blue,2.5);
      P.fn(function(x){return (1-prior)*n(x,1.3);},-5,5,PAL.red,2.5);
      // boundary where prior*n(x,-1.3) = (1-prior)*n(x,1.3)
      var bnd=Math.log((1-prior)/prior)/(2*1.3)*1;
      bnd = (Math.log(prior/(1-prior)))/(2*1.3)*-1;
      var xb = Math.log((1-prior)/Math.max(prior,1e-9))/(2*1.3);
      P.line(xb,0,xb,0.48,PAL.ink,2,[4,3]);
      P.txt(16,324,"P(class A) = "+prior.toFixed(2),PAL.blue,13,"left","600");
      P.txt(200,324,"boundary at x = "+xb.toFixed(3),PAL.ink,13,"left","700");
      P.txt(390,324,"rarer class needs stronger evidence",PAL.ink3,12,"left");
    }
    slider(controls,{label:"prior for A",min:0.05,max:0.95,val:prior,step:.01,fmt:function(v){return v.toFixed(2);},on:function(v){prior=v;draw();}});
    draw();
  });

  /* C.4 — entropy, cross-entropy, KL in one picture */
  reg("C.4", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var q1=0.25;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("H(p) is unavoidable; KL is the extra you pay for a wrong model");
      var p=[0.6,0.4], q=[q1,1-q1];
      var H=-(p[0]*Math.log2(p[0])+p[1]*Math.log2(p[1]));
      var CE=-(p[0]*Math.log2(q[0])+p[1]*Math.log2(q[1]));
      var KL=CE-H;
      var x0=90, W=420, y=140, h=44;
      ctx.fillStyle=PAL.blueF; ctx.fillRect(x0,y,W*(H/CE),h);
      ctx.strokeStyle=PAL.blue; ctx.lineWidth=1.4; ctx.strokeRect(x0,y,W*(H/CE),h);
      ctx.fillStyle=PAL.redF; ctx.fillRect(x0+W*(H/CE),y,W*(KL/CE),h);
      ctx.strokeStyle=PAL.red; ctx.strokeRect(x0+W*(H/CE),y,W*(KL/CE),h);
      P.txt(x0+W*(H/CE)/2,y+28,"H(p) = "+H.toFixed(3),PAL.blue,13,"center","700");
      if (KL/CE>0.09) P.txt(x0+W*(H/CE)+W*(KL/CE)/2,y+28,"KL = "+KL.toFixed(3),PAL.red,12.5,"center","700");
      P.txt(x0,y-16,"cross-entropy H(p,q) = "+CE.toFixed(3)+" bits",PAL.ink,13,"left","700");
      P.txt(16,270,"true p = [0.60, 0.40]     model q = ["+q[0].toFixed(2)+", "+q[1].toFixed(2)+"]",PAL.ink2,13,"left","600");
      P.txt(16,300, KL<0.005?"q matches p — you pay only H(p), the irreducible floor ✓":"training pushes q → p, driving KL to 0",
            KL<0.005?PAL.green:PAL.ink3,12.5,"left","600");
    }
    slider(controls,{label:"model q₁",min:0.02,max:0.98,val:q1,step:.01,fmt:function(v){return v.toFixed(2);},on:function(v){q1=v;draw();}});
    draw();
  });

  /* C.5 — a tiny autograd tape */
  reg("C.5", function (mount) {
    var wrap=card(mount), C=canvasIn(wrap,600,340), controls=controlsIn(wrap);
    var a=1.5, b=-0.8;
    function draw(){
      var ctx=C.ctx, P=plot(C.ctx,600,340,0,1,0,1); P.clear();
      P.title("L = (a·b + a)² — the tape replays backwards to get both gradients");
      var c=a*b, d=c+a, L=d*d;
      var dL=1, dd=2*d, dc=dd, da1=dd, da2=dc*b, db=dc*a, da=da1+da2;
      var nodes=[["a",a,da],["b",b,db],["c = a·b",c,dc],["d = c+a",d,dd],["L = d²",L,dL]];
      var xs=[70,70,240,400,530], ys=[105,205,155,155,155];
      nodes.forEach(function(nd,i){
        ctx.fillStyle=i<2?PAL.neutralF:(i===4?PAL.indigoF:PAL.blueF);
        ctx.strokeStyle=i<2?PAL.ink2:(i===4?PAL.indigo:PAL.blue); ctx.lineWidth=1.6;
        ctx.beginPath(); ctx.arc(xs[i],ys[i],34,0,7); ctx.fill(); ctx.stroke();
        P.txt(xs[i],ys[i]-4,nd[0],ctx.strokeStyle,11.5,"center","700");
        P.txt(xs[i],ys[i]+13,nd[1].toFixed(2),PAL.ink,11.5,"center","600");
        P.txt(xs[i],ys[i]+54,"∂L/∂ = "+nd[2].toFixed(2),PAL.red,11,"center","600"); });
      [[0,2],[1,2],[2,3],[0,3],[3,4]].forEach(function(e){
        ctx.strokeStyle=PAL.ink3; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.moveTo(xs[e[0]]+34,ys[e[0]]); ctx.lineTo(xs[e[1]]-34,ys[e[1]]); ctx.stroke(); });
      P.txt(16,300,"a feeds TWO paths, so its gradient is the SUM: "+da1.toFixed(2)+" + "+da2.toFixed(2)+" = "+da.toFixed(2),PAL.red,12.5,"left","700");
    }
    slider(controls,{label:"a",min:-2,max:2,val:a,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){a=v;draw();}});
    slider(controls,{label:"b",min:-2,max:2,val:b,step:.1,fmt:function(v){return v.toFixed(1);},on:function(v){b=v;draw();}});
    draw();
  });

})();
