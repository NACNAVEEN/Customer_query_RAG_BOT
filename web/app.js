/**
 * InstaParkAI Chatbot Widget — Frontend Logic
 * Connects to the FastAPI backend at /api/query
 */

const API_BASE = 'http://localhost:8000/api';

// ─── State ────────────────────────────────────────────────────────
let sessionId = null;
let isOpen = false;
let isLoading = false;

// ─── DOM refs ─────────────────────────────────────────────────────
const chatFab      = document.getElementById('chatFab');
const chatWindow   = document.getElementById('chatWindow');
const chatMessages = document.getElementById('chatMessages');
const chatForm     = document.getElementById('chatForm');
const chatInput    = document.getElementById('chatInput');
const chatSend     = document.getElementById('chatSend');
const chatClose    = document.getElementById('chatClose');
const chatRefresh  = document.getElementById('chatRefresh');
const suggestions  = document.getElementById('chatSuggestions');
const navbar       = document.getElementById('navbar');

// ─── Helpers ──────────────────────────────────────────────────────
function timeNow() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
}

// ─── Toggle Chat Window ───────────────────────────────────────────
function toggleChat() {
  isOpen = !isOpen;
  chatFab.classList.toggle('open', isOpen);
  chatWindow.classList.toggle('open', isOpen);
  if (isOpen) {
    // Show welcome message on first open
    if (chatMessages.children.length === 0) {
      addBotMessage("Hi! I'm <strong>InstaParkAI Assistant</strong>.<br>How can I help you today?");
    }
    setTimeout(() => chatInput.focus(), 350);
  }
}

chatFab.addEventListener('click', toggleChat);
chatClose.addEventListener('click', toggleChat);

// ─── New Conversation ─────────────────────────────────────────────
chatRefresh.addEventListener('click', async () => {
  // Clear server session
  if (sessionId) {
    try { await fetch(`${API_BASE}/sessions/${sessionId}/clear`, { method: 'POST' }); } catch (_) {}
  }
  sessionId = null;
  chatMessages.innerHTML = '';
  suggestions.classList.remove('hidden');
  addBotMessage("Hi! I'm <strong>InstaParkAI Assistant</strong>.<br>How can I help you today?");
});

// ─── Suggestion Chips ─────────────────────────────────────────────
suggestions.addEventListener('click', (e) => {
  const chip = e.target.closest('.suggestion-chip');
  if (!chip) return;
  const query = chip.dataset.query;
  sendMessage(query);
});

// ─── Form Submit ──────────────────────────────────────────────────
chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text || isLoading) return;
  sendMessage(text);
});

// ─── Send Message ─────────────────────────────────────────────────
async function sendMessage(text) {
  if (isLoading) return;

  // Hide suggestions after first message
  suggestions.classList.add('hidden');

  // Add user bubble
  addUserMessage(text);
  chatInput.value = '';
  chatInput.focus();

  // Show typing indicator
  isLoading = true;
  chatSend.disabled = true;
  const typingEl = addTypingIndicator();
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, session_id: sessionId }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const data = await res.json();

    // Store session
    if (data.session_id) sessionId = data.session_id;

    // Remove typing, add bot response
    typingEl.remove();
    addBotMessage(data.answer, data.citations, data.latency_ms, data.cache_hit);

  } catch (err) {
    typingEl.remove();
    addBotMessage("Sorry, I couldn't connect to the server. Please try again.");
    console.error('Chat error:', err);
  } finally {
    isLoading = false;
    chatSend.disabled = false;
  }
}

// ─── Render Messages ──────────────────────────────────────────────
function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-avatar">🧑</div>
    <div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
      <div class="msg-time">${timeNow()}</div>
    </div>`;
  chatMessages.appendChild(row);
  scrollToBottom();
}

function addBotMessage(html, citations, latencyMs, cacheHit) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';

  let extra = '';

  // Citations
  if (citations && citations.length > 0) {
    const pills = citations
      .map(c => `<span class="cite-pill" style="display: inline-block; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-right: 4px; margin-top: 6px; color: var(--text-muted);">📖 ${escapeHtml(c.display || c.source || '')}</span>`)
      .join('');
    extra += `<div class="msg-citations" style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px;">${pills}</div>`;
  }

  // Latency
  if (latencyMs) {
    let tag = `⚡ ${Math.round(latencyMs)}ms`;
    if (cacheHit) tag += ' · cached';
    extra += `<div class="msg-latency" style="margin-top: 6px; font-size: 0.75rem; opacity: 0.7; color: var(--text-muted);">${tag}</div>`;
  }

  row.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div>
      <div class="msg-bubble">${html}${extra}</div>
      <div class="msg-time">${timeNow()}</div>
    </div>`;
  chatMessages.appendChild(row);
  scrollToBottom();
}

function addTypingIndicator() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble typing-dots"><span></span><span></span><span></span></div>`;
  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ─── Navbar Scroll Effect ─────────────────────────────────────────
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
});

// ─── Smooth Nav Links ─────────────────────────────────────────────
document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', (e) => {
    document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  });
});
