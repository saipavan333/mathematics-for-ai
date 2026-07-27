/* Service worker — offline shell for "Mathematics for AI Engineers".
   Strategy: precache the app shell on install; network-first for page navigations
   (so deploys flow through), cache-first-then-network for same-origin static assets.
   Cross-origin requests (KaTeX / Pyodide CDN, the assistant Worker) bypass the SW. */
var CACHE = "mathai-v1";
var CORE = [
  "./", "index.html", "styles.css", "features.css", "app.js", "features.js",
  "curriculum.js", "assistant-data.js", "assistant-config.js",
  "gd-assistant.js", "gd-assistant.css", "gd-extras.js", "gd-extras.css",
  "manifest.webmanifest", "icon-192.png", "icon-512.png",
  "content/track01.js", "content/track02.js", "content/track03.js", "content/track04.js",
  "content/track05.js", "content/track06.js", "content/track07.js", "content/track08.js",
  "content/track09.js", "content/track10.js", "content/track11.js", "content/track12.js",
  "content/track13.js", "content/track14.js", "content/track15.js", "content/track16.js",
  "content/track17.js", "content/capstones.js", "content/nutshells.js",
  "content/diagrams.js", "content/widgets.js"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) {
      /* add individually so one missing file can't abort the whole precache */
      return Promise.all(CORE.map(function (u) { return c.add(u).catch(function () { }); }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (req.method !== "GET") return;
  var url;
  try { url = new URL(req.url); } catch (_) { return; }
  if (url.origin !== self.location.origin) return;               // let CDN / Worker go straight to network

  if (req.mode === "navigate") {                                  // pages: fresh first, cache as fallback
    e.respondWith(
      fetch(req).then(function (r) {
        var cp = r.clone(); caches.open(CACHE).then(function (c) { c.put(req, cp); });
        return r;
      }).catch(function () { return caches.match(req).then(function (m) { return m || caches.match("index.html"); }); })
    );
    return;
  }
  e.respondWith(                                                  // static: cache first, then network
    caches.match(req).then(function (m) {
      return m || fetch(req).then(function (r) {
        if (r && r.ok) { var cp = r.clone(); caches.open(CACHE).then(function (c) { c.put(req, cp); }); }
        return r;
      }).catch(function () { return m; });
    })
  );
});
