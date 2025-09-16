<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport"
        content="width=device-width,
                 initial-scale=1,
                 maximum-scale=1,
                 user-scalable=no">
 
  <!-- PWA Web App Manifest -->
  <link rel="manifest" href="/manifest.json">
  <link rel="apple-touch-icon" href="/images/main-logo.png">

  <!-- iOS support -->
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="AINARA">
  <link rel="apple-touch-icon" href="../images/main-logo.png">

  <title>Document</title>
  <link rel="stylesheet" href="../styles/main-style.css">
</head>
<body>
  <nav>
    <button class="back" onclick="window.history.back();"><img src="../icons/left.png" alt=""></button>
    <h2>AINARA</h2>
    <button id="menuBtn"><img src="../icons/menu.png" alt=""></button>
  </nav>

  <div class="onceonly">
    <img src="../images/logo.png" alt="">
    <h1>Beyond Roma Knowledge</h1>
  </div>

  <div class="chatsystem">
    <!-- messages will be injected here -->
  </div>

  <div class="inputbox">
    <div class="input">
      <div class="text">
        <input type="text" placeholder="Ask me anything">
      </div>
      <div class="controls">
        <div class="tools">
          <button><img src="../icons/brain.png" alt=""></button>
          <button><img src="../icons/telescope.png" alt=""></button>
          <button><img src="../icons/image.png" alt=""></button>
        </div>
        <div class="sendcon">
          <button><img src="../icons/mic.png" alt=""></button>
          <button class="sendbtn"><img src="../icons/up.png" alt=""></button>
        </div>
      </div>
    </div>
  </div>

  <div class="conversations" id="conversationsPanel">
    <nav>
      <button class="back"><img src="../icons/left.png" alt=""></button>
      <h2>Chats</h2>
    </nav>
    <center>
      <div class="search">
        <button><img src="../icons/newchat.png" alt="">Start New Chat</button>
        <button><img src="../icons/search.png" alt="">Search Past Conversations</button>
      </div>
      <div class="pastchats">
        <!-- past chats -->
      </div>
    </center>
  </div>

  <script>
    // side‑panel
    const menuBtn = document.getElementById('menuBtn');
    const panel   = document.getElementById('conversationsPanel');
    const backBtn = panel.querySelector('.back');
    menuBtn.addEventListener('click', () => {
      panel.style.display = 'block';
      setTimeout(() => panel.classList.add('visible'), 10);
    });
    backBtn.addEventListener('click', () => {
      panel.classList.remove('visible');
      setTimeout(() => panel.style.display = 'none', 400);
    });

    // chat logic
    const chatSystem = document.querySelector('.chatsystem');
    const userInput  = document.querySelector('.inputbox .text input');
    const sendBtn    = document.querySelector('.sendbtn');

    // hide splash on first input
    userInput.addEventListener('input', () => {
      const splash = document.querySelector('.onceonly');
      if (splash && !splash.classList.contains('hidden')) {
        splash.classList.add('hidden');
      }
    });

    function appendUserBubble(text) {
      const div = document.createElement('div');
      div.className = 'chatfromuser';
      div.innerHTML = `<p>${text}</p>`;
      chatSystem.appendChild(div);
      chatSystem.scrollTop = chatSystem.scrollHeight;
    }

    function showSpinner() {
      const wrap = document.createElement('div');
      wrap.id = 'loadingSpin';
      wrap.className = 'chatfromainara-wrapper';
      wrap.innerHTML = `
        <div class="chatfromainara-glow"></div>
        <div class="chatfromainara"><div class="spinner"></div></div>
      `;
      chatSystem.appendChild(wrap);
      chatSystem.scrollTop = chatSystem.scrollHeight;
    }

    function removeSpinner() {
      const s = document.getElementById('loadingSpin');
      if (s) s.remove();
    }

    function typeAIResponse(html) {
      const wrap = document.createElement('div');
      wrap.className = 'chatfromainara-wrapper';
      wrap.innerHTML = `
        <div class="chatfromainara-glow"></div>
        <div class="chatfromainara"></div>
      `;
      chatSystem.appendChild(wrap);

      const msgDiv = wrap.querySelector('.chatfromainara');
      let i = 0, speed = 30; // ms per char
      function type() {
        if (i < html.length) {
          msgDiv.innerHTML += html.charAt(i++);
          chatSystem.scrollTop = chatSystem.scrollHeight;
          setTimeout(type, speed);
        } else {
          // feedback buttons
          const fb = document.createElement('div');
          fb.className = 'feedback';
          fb.innerHTML = `
            <button><img src="../icons/copy.png" alt=""></button>
            <button><img src="../icons/like.png" alt=""></button>
            <button><img src="../icons/dislike.png" alt=""></button>
          `;
          chatSystem.appendChild(fb);
          chatSystem.scrollTop = chatSystem.scrollHeight;
        }
      }
      type();
    }

    async function sendMessage() {
      const msg = userInput.value.trim();
      if (!msg) return;
      appendUserBubble(msg);
      userInput.value = '';
      showSpinner();

      try {
        const fd  = new FormData();
        fd.append('query', msg);
        const res = await fetch('../../fridaygui/chat_api.php', { method: 'POST', body: fd });
        let data  = await res.text();
        removeSpinner();
        // strip wrapping <pre> if any
        data = data.replace(/^<pre.*?>|<\/pre>$/g, '');
        typeAIResponse(data);
      } catch (e) {
        removeSpinner();
        typeAIResponse('⚠️ Failed to fetch response.');
        console.error(e);
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') sendMessage();
    });
  </script>

  <script src="../api/script-hider.js"></script>
</body>
</html>
