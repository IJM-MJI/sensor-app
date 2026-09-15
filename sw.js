/* Sensor Data PWA service worker.
   Navigation (index.html): network-first so a new deploy shows when online,
   with a cached fallback for offline. Everything else (versioned model JS, the
   OpenCV.js CDN bundle, icons): cache-first with a background fill, so after the
   first online load the app runs fully offline. Bump CACHE to invalidate. */
const CACHE = 'sensor-data-v1';

self.addEventListener('install', function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) { return c.addAll(['./', './index.html']); })
      .then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE; })
        .map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;

  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req).then(function (resp) {
        var cp = resp.clone();
        caches.open(CACHE).then(function (c) { c.put(req, cp); });
        return resp;
      }).catch(function () {
        return caches.match(req).then(function (r) { return r || caches.match('./index.html'); });
      })
    );
    return;
  }

  e.respondWith(
    caches.match(req).then(function (cached) {
      return cached || fetch(req).then(function (resp) {
        try {
          var cp = resp.clone();
          caches.open(CACHE).then(function (c) { c.put(req, cp); });
        } catch (_) {}
        return resp;
      });
    })
  );
});
