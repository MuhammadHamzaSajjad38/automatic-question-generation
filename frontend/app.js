/* ════════════════════════════════════════════════════
   QuestionForge · app.js
   Connects the frontend UI to the Flask/Gradio backend
   ════════════════════════════════════════════════════ */

'use strict';

// ── Config ─────────────────────────────────────────────────────────
// The Flask backend runs on port 5000 by default (app.py).
// Change this if your backend is on a different host/port.
const API_BASE = 'http://127.0.0.1:5000';

// ── Sample passage ──────────────────────────────────────────────────
const SAMPLE_TEXT = `Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity, but he also made important contributions to quantum mechanics. His mass-energy equivalence formula E = mc², which arises from relativity theory, has been called "the world's most famous equation". He received the Nobel Prize in Physics in 1921 for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect. His work is also known for its influence on the philosophy of science. In 1999, he was named Time magazine's "Person of the Century".`;

// ── DOM refs ────────────────────────────────────────────────────────
const inputText        = document.getElementById('input-text');
const charCount        = document.getElementById('char-count');
const wordCount        = document.getElementById('word-count');
const maxQSlider       = document.getElementById('max-questions');
const maxQDisplay      = document.getElementById('max-q-display');
const toggleMCQ        = document.getElementById('toggle-mcq');
const toggleAnswers    = document.getElementById('toggle-answers');
const btnGenerate      = document.getElementById('btn-generate');
const generateLabel    = document.getElementById('generate-label');
const btnSample        = document.getElementById('btn-sample');
const btnCopy          = document.getElementById('btn-copy');
const btnPdf           = document.getElementById('btn-pdf');

const outputPlaceholder = document.getElementById('output-placeholder');
const outputLoading     = document.getElementById('output-loading');
const outputResults     = document.getElementById('output-results');
const outputActions     = document.getElementById('output-actions');
const resultStatus      = document.getElementById('result-status');
const questionsList     = document.getElementById('questions-list');
const qCountBadge       = document.getElementById('q-count-badge');

const step1 = document.getElementById('step-1');
const step2 = document.getElementById('step-2');
const step3 = document.getElementById('step-3');

const toast = document.getElementById('toast');

// ── State ───────────────────────────────────────────────────────────
let generatedQA = [];
let loadingTimer = null;

// ── Char / word counter ─────────────────────────────────────────────
function updateCounter() {
  const text = inputText.value;
  charCount.textContent = text.length.toLocaleString();
  wordCount.textContent = text.trim() ? text.trim().split(/\s+/).length.toLocaleString() + ' words' : '0 words';
}
inputText.addEventListener('input', updateCounter);
updateCounter();

// ── Slider label ────────────────────────────────────────────────────
function updateSlider() {
  const val = maxQSlider.value;
  maxQDisplay.textContent = val;
  const pct = ((val - maxQSlider.min) / (maxQSlider.max - maxQSlider.min)) * 100;
  maxQSlider.style.background = `linear-gradient(to right, var(--c-primary) 0%, var(--c-primary) ${pct}%, var(--c-surface-3) ${pct}%)`;
}
maxQSlider.addEventListener('input', updateSlider);
updateSlider();

// ── Sample text ─────────────────────────────────────────────────────
btnSample.addEventListener('click', () => {
  inputText.value = SAMPLE_TEXT;
  updateCounter();
  inputText.focus();
});

// ── Toast helper ────────────────────────────────────────────────────
let toastTimeout;
function showToast(msg, type = '') {
  clearTimeout(toastTimeout);
  toast.textContent = msg;
  toast.className = 'toast show ' + type;
  toastTimeout = setTimeout(() => { toast.className = 'toast'; }, 3200);
}

// ── UI state transitions ─────────────────────────────────────────────
function setUIState(state) {
  outputPlaceholder.style.display = 'none';
  outputLoading.style.display     = 'none';
  outputResults.style.display     = 'none';
  outputActions.style.display     = 'none';

  if (state === 'placeholder') {
    outputPlaceholder.style.display = 'flex';
  } else if (state === 'loading') {
    outputLoading.style.display = 'flex';
  } else if (state === 'results') {
    outputResults.style.display = 'flex';
    outputActions.style.display = 'flex';
  }
}

// ── Animate loading steps ───────────────────────────────────────────
function animateLoadingSteps() {
  const steps = [step1, step2, step3];
  steps.forEach(s => { s.className = 'step'; });
  step1.classList.add('step-active');

  let current = 0;
  loadingTimer = setInterval(() => {
    if (current < steps.length) {
      steps[current].classList.remove('step-active');
      steps[current].classList.add('step-done');
      current++;
      if (current < steps.length) steps[current].classList.add('step-active');
    }
  }, 900);
}
function stopLoadingSteps() {
  clearInterval(loadingTimer);
}

// ── Render MCQ options ──────────────────────────────────────────────
function renderMCQ(mcq) {
  if (!mcq || !mcq.options) return '';
  const letters = ['A','B','C','D','E'];
  const opts = mcq.options.map((opt, i) => {
    const isCorrect = opt === mcq.correct;
    return `
      <div class="mcq-option ${isCorrect ? 'correct' : ''}">
        <span class="mcq-option-letter">${letters[i] || i+1}</span>
        <span>${escHtml(opt)}</span>
        ${isCorrect ? '<span class="mcq-correct-badge">✓ Correct</span>' : ''}
      </div>`;
  }).join('');
  return `
    <div class="q-mcq">
      <p class="q-mcq-label">Multiple Choice Options</p>
      <div class="mcq-options">${opts}</div>
    </div>`;
}

// ── Render a single question card ───────────────────────────────────
function renderQuestionCard(qa, idx, delay) {
  const nerTag = qa.label ? `<span class="q-ner-tag">${escHtml(qa.label)}</span>` : '';
  const mcqHTML = toggleMCQ.checked && qa.mcq ? renderMCQ(qa.mcq) : '';
  const answerHTML = toggleAnswers.checked ? `
    <div class="q-answer-row">
      <span class="q-answer-icon">✅</span>
      <div>
        <div class="q-answer-label">Answer</div>
        <div class="q-answer-text">${escHtml(qa.answer)}</div>
      </div>
    </div>` : '';
  const contextHTML = qa.context ? `
    <div class="q-context-row">"${escHtml(truncate(qa.context, 140))}"</div>` : '';

  return `
    <div class="q-card" style="animation-delay:${delay}ms">
      <div class="q-card-header">
        <div class="q-number">${idx}</div>
        <p class="q-text">${escHtml(qa.question)}</p>
        ${nerTag}
      </div>
      <div class="q-card-body">
        ${answerHTML}
        ${contextHTML}
        ${mcqHTML}
      </div>
    </div>`;
}

// ── Generate (calls Flask backend or falls back to mock) ────────────
btnGenerate.addEventListener('click', async () => {
  const text = inputText.value.trim();
  if (!text) { showToast('⚠️ Please enter or paste a text passage first.', 'error'); return; }

  // Lock UI
  btnGenerate.disabled = true;
  generateLabel.textContent = 'Generating…';
  setUIState('loading');
  animateLoadingSteps();

  const maxQ   = parseInt(maxQSlider.value, 10);
  const doMCQ  = toggleMCQ.checked;

  try {
    /* ── Try the real Flask backend ─────────────────────────────── */
    const response = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, generate_mcq: doMCQ, max_questions: maxQ }),
    });

    if (!response.ok) throw new Error(`Server responded ${response.status}`);
    const data = await response.json();

    if (data.error) throw new Error(data.error);

    generatedQA = data.qa_list || [];
    displayResults(generatedQA, data.status || `Generated ${generatedQA.length} questions.`);

  } catch (err) {
    /* ── Fallback: client-side mock demo mode ──────────────────── */
    console.warn('Backend unavailable — using demo mode:', err.message);
    generatedQA = generateDemoQuestions(text, maxQ, doMCQ);
    displayResults(generatedQA,
      `⚡ Demo mode (backend offline) — showing ${generatedQA.length} sample questions.`);
  }

  // Unlock UI
  stopLoadingSteps();
  btnGenerate.disabled = false;
  generateLabel.textContent = 'Generate Questions';
});

// ── Display results ─────────────────────────────────────────────────
function displayResults(qaList, status) {
  if (!qaList || qaList.length === 0) {
    setUIState('placeholder');
    showToast('No questions could be generated. Try a longer passage.', 'error');
    return;
  }

  resultStatus.textContent = '✅ ' + status;
  qCountBadge.textContent = `${qaList.length} question${qaList.length !== 1 ? 's' : ''}`;
  questionsList.innerHTML = qaList
    .map((qa, i) => renderQuestionCard(qa, i + 1, i * 60))
    .join('');

  setUIState('results');
  showToast(`✅ ${qaList.length} questions generated!`, 'success');
}

// ── Copy all questions ──────────────────────────────────────────────
btnCopy.addEventListener('click', () => {
  if (!generatedQA.length) return;
  const text = generatedQA.map((qa, i) =>
    `Q${i+1}. ${qa.question}\nA: ${qa.answer}\n`
  ).join('\n');
  navigator.clipboard.writeText(text)
    .then(() => showToast('📋 Questions copied to clipboard!', 'success'))
    .catch(() => showToast('Could not copy to clipboard.', 'error'));
});

// ── Export PDF (calls backend) ──────────────────────────────────────
btnPdf.addEventListener('click', async () => {
  if (!generatedQA.length) return;
  try {
    const response = await fetch(`${API_BASE}/export_pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        qa_list: generatedQA,
        include_answers: toggleAnswers.checked,
        include_mcq: toggleMCQ.checked,
      }),
    });
    if (!response.ok) throw new Error('PDF export failed');
    const blob = await response.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'question_paper.pdf'; a.click();
    URL.revokeObjectURL(url);
    showToast('📄 PDF downloaded!', 'success');
  } catch {
    showToast('⚠️ Could not export PDF — backend may be offline.', 'error');
  }
});

// ── Utility helpers ─────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function truncate(str, n) {
  return str && str.length > n ? str.slice(0, n) + '…' : str;
}

// ── Demo mode: client-side question generation ──────────────────────
// Used when the Flask backend is not running — extracts noun phrases
// from the text using simple regex heuristics.
function generateDemoQuestions(text, maxQ, doMCQ) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const results = [];
  const starters = [
    'What is', 'Who is', 'When did', 'Where is', 'How did', 'What was', 'Which',
  ];
  const nounRegex = /\b([A-Z][a-z]+(?: [A-Z][a-z]+)*|\d{4}|[A-Z]{2,})\b/g;

  for (let si = 0; si < sentences.length && results.length < maxQ; si++) {
    const sent = sentences[si].trim();
    const matches = [...new Set((sent.match(nounRegex) || []))];
    for (const match of matches) {
      if (results.length >= maxQ) break;
      const starter = starters[results.length % starters.length];
      const q = `${starter} "${match}" in the given passage?`;
      const qa = { question: q, answer: match, context: sent, label: 'NER' };
      if (doMCQ) {
        const distractors = getDemoDistractors(match, matches);
        const options = shuffle([match, ...distractors.slice(0,3)]);
        qa.mcq = { correct: match, options };
      }
      results.push(qa);
    }
  }
  return results;
}

function getDemoDistractors(answer, pool) {
  const generics = ['None of the above', 'All of the above', 'Not mentioned'];
  const siblings = pool.filter(p => p !== answer);
  return [...siblings, ...generics].slice(0, 3);
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ── Initial state ────────────────────────────────────────────────────
setUIState('placeholder');
