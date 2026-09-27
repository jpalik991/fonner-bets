// Network first, so updates show up right away; the saved copy is used only when offline.
const CACHE = 'fonner-bets-v5';
const FILES = ['./', 'index.html', 'fonner.js', 'model.json', 'manifest.webmanifest', 'icon.png', 'jszip.min.js',
  'fonts/nunito-sans-latin-400-normal.woff2', 'fonts/nunito-sans-latin-500-normal.woff2', 'fonts/nunito-sans-latin-600-normal.woff2',
  'fonts/nunito-sans-latin-700-normal.woff2', 'fonts/nunito-sans-latin-800-normal.woff2'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => Promise.all(FILES.map(f => c.add(f).catch(() => null)))));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request).then(r => {
      if (r && r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
      return r;
    }).catch(() => caches.match(e.request, { ignoreSearch: true }))
  );
});
