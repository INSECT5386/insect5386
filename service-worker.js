const CACHE_NAME = "interactgpt-cache-v3";
const urlsToCache = [
  "./",
  "./index.html",
  "./manifest.json",
  "./1icon-192.png",
  "./1icon-512.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(urlsToCache))
  );
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request))
  );
});
