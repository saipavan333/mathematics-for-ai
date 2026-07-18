/* interactive lesson widgets — vanilla JS canvas, no external libs, theme-safe
   (each draws on a fixed light plate like the diagrams). window.WIDGETS[id](mount) */
window.WIDGETS = {};
(function () {
  var DPR = (typeof window !== "undefined" && window.devicePixelRatio) ? window.devicePixelRatio : 1;
  var PAL = { ink:"#1f2a44", ink2:"#4a5878", ink3:"#6b7891", axis:"#9aa6bd", grid:"#e7ecf4",
    blue:"#2a6f97", green:"#3a7d44", indigo:"#4f46e5", amber:"#b9722a", red:"#c0392b",
    accent:"#4f46e5", plate:"#f8fafc",
    blueF:"#dbe8fb", greenF:"#e3efe5", redF:"#f7e0dc", amberF:"#f7e7d2",
    indigoF:"#e5e7ff", neutralF:"#eef2f7" };

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

})();
