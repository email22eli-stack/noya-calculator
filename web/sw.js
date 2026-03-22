const CACHE = 'noya-v2';
const ASSETS = [
  '.',
  'index.html',
  'manifest.json',
  'Noya.png',
  'https://cdn.jsdelivr.net/npm/mathjs@13/package.json',
  'https://unpkg.com/nerdamer/nerdamer.core.js',
  'https://unpkg.com/nerdamer/Calculus.js',
  'https://unpkg.com/nerdamer/Algebra.js',
  'https://unpkg.com/nerdamer/Solve.js',
  'https://cdn.plot.ly/plotly-2.35.2.min.js'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
