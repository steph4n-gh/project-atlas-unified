/**
 * GossetGate: Offline Service Worker
 * 
 * Caches static shell assets (HTML, CSS, JS), external dependencies (Three.js, Fonts),
 * and model weight binaries for full offline functionality.
 */

const CACHE_NAME = 'gossetgate-v1';
const PRECACHE_ASSETS = [
  '/',
  '/static/index.css',
  '/static/js/app.js',
  '/static/js/api.js',
  '/static/js/components/sidebar.js',
  '/static/js/components/telemetry.js',
  '/static/js/components/chat.js',
  '/static/js/components/e8_viz_3d.js',
  '/static/js/components/e8_radar_overlay.js',
  '/static/js/components/audit.js',
  '/static/js/web_uce_runner.js',
  
  // External Libraries & Google Fonts
  'https://unpkg.com/three@0.160.0/build/three.module.js',
  'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js',
  'https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Outfit:wght@300;400;600;700&display=swap',
  
  // Local model files for offline client-side runner
  '/data/uce_e4b_distilled.safetensors',
  '/data/uce_e4b_distilled.meta.json'
];

// Install Event - Pre-cache critical app shell and model assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Pre-caching static assets and model weights...');
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate Event - Clean up obsolete caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cache => {
          if (cache !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache:', cache);
            return caches.delete(cache);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event - Serve with appropriate strategies
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // For heavy model binary weights and metadata, use Cache-First to save bandwidth
  if (url.pathname.endsWith('.safetensors') || url.pathname.endsWith('.meta.json')) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        if (cachedResponse) {
          console.log('[Service Worker] Serving model file from cache:', url.pathname);
          return cachedResponse;
        }
        return fetch(event.request).then(networkResponse => {
          if (networkResponse.status === 200) {
            const responseClone = networkResponse.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, responseClone);
            });
          }
          return networkResponse;
        });
      })
    );
    return;
  }

  // Stale-While-Revalidate strategy for static JS, CSS, and HTML pages
  if (
    event.request.mode === 'navigate' ||
    url.pathname.startsWith('/static/') ||
    url.hostname.includes('unpkg.com') ||
    url.hostname.includes('fonts.googleapis.com') ||
    url.hostname.includes('fonts.gstatic.com')
  ) {
    event.respondWith(
      caches.open(CACHE_NAME).then(cache => {
        return cache.match(event.request).then(cachedResponse => {
          const fetchPromise = fetch(event.request).then(networkResponse => {
            if (networkResponse.status === 200) {
              cache.put(event.request, networkResponse.clone());
            }
            return networkResponse;
          }).catch(() => {
            // Offline fallback for navigations
            if (event.request.mode === 'navigate') {
              return caches.match('/');
            }
          });
          return cachedResponse || fetchPromise;
        });
      })
    );
    return;
  }

  // Network-First for API requests (like /api/chat/stream or /api/model/load)
  // If the server is offline, fallback to the local WebUCERunner offline simulation!
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request).catch(error => {
        console.log('[Service Worker] API network request failed. Serving simulated offline JSON status.');
        
        // Provide mock offline status/SSE data if requesting model check
        if (url.pathname.includes('/api/model/load')) {
          return new Response(JSON.stringify({
            type: 'complete',
            message: 'Switched to local client-side WebUCERunner (Offline Mode)'
          }), { headers: { 'Content-Type': 'application/json' } });
        }
        
        // Return standard offline error so frontend knows to switch to WebUCERunner
        return new Response(JSON.stringify({
          error: 'Offline',
          message: 'Local server unreachable. Switch to Offline simulation.'
        }), { status: 503, headers: { 'Content-Type': 'application/json' } });
      })
    );
  }
});
