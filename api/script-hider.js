  // existing SW registration…
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
    .then(() => console.log('Service Worker registered'))
    .catch(err => console.error('SW registration failed:', err));
  }

  // hide the intro banner on first typing
  (function(){
    const once = document.querySelector('.onceonly');
    const input = document.querySelector('input[type="text"]');
    let hidden = false;
    input.addEventListener('input', () => {
      if (hidden) return;
      hidden = true;
      // fade out
      once.style.transition = 'opacity 0.4s ease';
      once.style.opacity = '0';
      setTimeout(() => {
        once.style.display = 'none';
      }, 400);
    });
  })();
