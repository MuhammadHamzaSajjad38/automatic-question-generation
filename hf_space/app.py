import subprocess, sys
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import streamlit as st
import torch
import spacy
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuestionForge — AI Question Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 3rem 4rem 3rem !important;
    max-width: 1160px !important;
}
.stApp { background: #07070e; color: #f0eeff; }

/* ── Animated gradient background ────────────────── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 700px 500px at 15% 20%, rgba(100,80,220,.13) 0%, transparent 70%),
        radial-gradient(ellipse 500px 400px at 85% 75%, rgba(20,210,160,.08) 0%, transparent 70%),
        radial-gradient(ellipse 400px 350px at 60% 10%, rgba(160,100,255,.07) 0%, transparent 70%);
    pointer-events: none;
}

/* ── Typography helpers ──────────────────────────── */
.font-space { font-family: 'Space Grotesk', sans-serif !important; }
.font-mono  { font-family: 'JetBrains Mono', monospace !important; }

/* ── Hero section ────────────────────────────────── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(120,90,255,.12);
    border: 1px solid rgba(120,90,255,.28);
    color: #a48aff;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    padding: 6px 18px; border-radius: 99px;
    margin-bottom: 1.6rem;
}
.hero-eyebrow span { width: 6px; height: 6px; border-radius: 50%; background: #a48aff; display: inline-block; animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5.5vw, 4rem);
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1.08;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.1rem;
}
.hero-sub {
    font-size: 1.05rem; font-weight: 400;
    color: #7870a0; max-width: 500px;
    margin: 0 auto 2rem;
    line-height: 1.75;
}

/* Metric pills in hero */
.metric-strip {
    display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.metric-pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px; padding: 8px 16px;
}
.metric-pill-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px; font-weight: 600;
    background: linear-gradient(135deg, #c4b5fd, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-pill-lbl { font-size: 11px; color: #5a5475; font-weight: 500; }

/* ── Divider ─────────────────────────────────────── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(120,90,255,.35), rgba(103,232,249,.2), transparent);
    margin: 1.5rem 0 2rem;
}

/* ── Section labels ──────────────────────────────── */
.section-label {
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: #4a4465; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 10px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(255,255,255,.06);
}

/* ── Textarea ────────────────────────────────────── */
textarea {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 14px !important;
    color: #f0eeff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.75 !important;
    padding: 1rem 1.2rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
textarea:focus {
    border-color: rgba(120,90,255,.5) !important;
    box-shadow: 0 0 0 3px rgba(120,90,255,.12) !important;
    outline: none !important;
}
textarea::placeholder { color: #3a3555 !important; }

/* ── Selectbox ───────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 12px !important;
    color: #f0eeff !important;
}
.stSelectbox > div > div:hover { border-color: rgba(120,90,255,.4) !important; }

/* ── Slider ──────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(120,90,255,.25) !important; }
.stSlider > div > div > div > div { background: #8264ff !important; box-shadow: 0 0 8px rgba(130,100,255,.5) !important; }
.stSlider label { color: #7870a0 !important; font-size: 13px !important; }

/* ── Generate button ────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c5af7 0%, #5b3ee0 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important; font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 6px 28px rgba(120,90,255,.35) !important;
    transition: all .2s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(120,90,255,.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ─────────────────────────────── */
.stDownloadButton > button {
    background: rgba(103,232,249,.08) !important;
    border: 1px solid rgba(103,232,249,.25) !important;
    color: #67e8f9 !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important; font-weight: 600 !important;
    width: 100% !important; transition: all .2s !important;
}
.stDownloadButton > button:hover {
    background: rgba(103,232,249,.16) !important;
    border-color: rgba(103,232,249,.45) !important;
}

/* ── Spinner ─────────────────────────────────────── */
.stSpinner > div { border-top-color: #8264ff !important; }

/* ── Question cards ──────────────────────────────── */
.q-outer {
    border-radius: 16px;
    padding: 1px;               /* border via gradient trick */
    background: linear-gradient(135deg, rgba(120,90,255,.25), rgba(103,232,249,.12));
    margin-bottom: 14px;
    transition: all .22s;
}
.q-outer:hover {
    background: linear-gradient(135deg, rgba(120,90,255,.5), rgba(103,232,249,.3));
    transform: translateX(3px);
}
.q-inner {
    background: rgba(10,9,20,.85);
    border-radius: 15px;
    padding: 1.1rem 1.4rem;
    display: flex; gap: 14px; align-items: flex-start;
}
.q-num-badge {
    width: 30px; height: 30px; border-radius: 9px;
    background: linear-gradient(135deg, #7c5af7, #5b3ee0);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; font-weight: 700; color: #fff;
    flex-shrink: 0; margin-top: 1px;
    box-shadow: 0 3px 10px rgba(120,90,255,.4);
}
.q-text {
    font-size: 15.5px; font-weight: 500;
    color: #e8e4ff; line-height: 1.6; flex: 1;
}

/* ── Empty state ─────────────────────────────────── */
.empty-state {
    border: 1px dashed rgba(255,255,255,.07);
    border-radius: 18px;
    padding: 4rem 1rem;
    text-align: center;
    color: #2e2a45;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 14px; opacity: .5; }
.empty-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; letter-spacing: 2px;
    text-transform: uppercase; color: #2e2a45;
}

/* ── Result header bar ───────────────────────────── */
.result-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.2rem;
}
.result-count {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; font-weight: 600;
    background: rgba(120,90,255,.15);
    border: 1px solid rgba(120,90,255,.25);
    color: #a48aff; padding: 5px 14px; border-radius: 99px;
}

/* ── Info/warning tweaks ─────────────────────────── */
.stAlert { border-radius: 12px !important; }

/* ── Labels ──────────────────────────────────────── */
label { color: #7870a0 !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow"><span></span>&nbsp;T5 · SQuAD 1.1 · NLP · MPhil Data Science&nbsp;<span></span></div>
    <div class="hero-title">Automatic Question<br>Generation</div>
    <p class="hero-sub">Paste any educational passage and generate exam-ready questions instantly using a fine-tuned T5 Transformer.</p>
    <div class="metric-strip">
        <div class="metric-pill"><span class="metric-pill-val">0.4509</span><span class="metric-pill-lbl">ROUGE-1</span></div>
        <div class="metric-pill"><span class="metric-pill-val">0.2445</span><span class="metric-pill-lbl">ROUGE-2</span></div>
        <div class="metric-pill"><span class="metric-pill-val">0.4167</span><span class="metric-pill-lbl">ROUGE-L</span></div>
        <div class="metric-pill"><span class="metric-pill-val">0.1581</span><span class="metric-pill-lbl">BLEU</span></div>
        <div class="metric-pill"><span class="metric-pill-val">60M</span><span class="metric-pill-lbl">Parameters</span></div>
    </div>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    MODEL_ID  = "Hamzasajjad38/t5-small-qg"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device)
    nlp       = spacy.load("en_core_web_sm")
    return tokenizer, model, device, nlp

with st.spinner("⚡ Loading T5 model from Hugging Face Hub…"):
    tokenizer, model, device, nlp = load_model()


# ══════════════════════════════════════════════════════════════════════
# SAMPLE PASSAGES
# ══════════════════════════════════════════════════════════════════════
SAMPLES = {
    "— paste your own —": "",
    "🌿 Biology — Photosynthesis": (
        "Photosynthesis is a process used by plants and other organisms to convert light energy into "
        "chemical energy stored in glucose. This process takes place inside chloroplasts, where the "
        "green pigment chlorophyll absorbs sunlight. Carbon dioxide from the air and water from the "
        "soil are the main raw materials used during photosynthesis. Oxygen is released as a byproduct "
        "of this process, which is essential for most life on Earth."
    ),
    "⚙️ History — Industrial Revolution": (
        "The Industrial Revolution began in Britain in the late 18th century and transformed "
        "manufacturing, agriculture, and transportation. The invention of the steam engine by James "
        "Watt in 1769 played a pivotal role, enabling factories to operate machinery and railroads to "
        "carry goods across the country. Workers migrated from rural areas to cities, leading to rapid "
        "urbanization and significant social changes across Europe and North America."
    ),
    "🤖 AI — Machine Learning": (
        "Machine learning is a subset of artificial intelligence that enables systems to learn from "
        "data and improve their performance without being explicitly programmed. Supervised learning, "
        "unsupervised learning, and reinforcement learning are the three main paradigms. Neural networks "
        "form the foundation of deep learning, revolutionizing fields like computer vision, speech "
        "recognition, and natural language processing. Geoffrey Hinton, Yann LeCun, and Yoshua Bengio "
        "are widely regarded as the pioneers of deep learning."
    ),
    "🌍 Geography — Amazon River": (
        "The Amazon River in South America is the largest river in the world by water discharge, "
        "carrying about 20 percent of all freshwater that flows into the oceans. It flows through "
        "Brazil, Peru, and Colombia, spanning over 6,400 kilometres. Its basin hosts the world's "
        "largest tropical rainforest, home to millions of plant and animal species. The Amazon plays "
        "a critical role in regulating Earth's climate by absorbing vast amounts of carbon dioxide."
    ),
    "🔬 Physics — Relativity": (
        "Albert Einstein published his special theory of relativity in 1905 while working as a patent "
        "clerk in Bern, Switzerland. The theory introduced the famous equation E = mc², demonstrating "
        "that mass and energy are interchangeable. In 1915, Einstein extended this with the general "
        "theory of relativity, which described gravity as the curvature of spacetime caused by mass. "
        "In 1921, he received the Nobel Prize in Physics for the discovery of the law of the "
        "photoelectric effect."
    ),
}


# ══════════════════════════════════════════════════════════════════════
# LAYOUT — two columns
# ══════════════════════════════════════════════════════════════════════
col_left, gap_col, col_right = st.columns([11, 1, 12])

# ── LEFT — INPUT ─────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">01 — Passage</div>', unsafe_allow_html=True)

    sample_key = st.selectbox(
        "Load a sample",
        list(SAMPLES.keys()),
        label_visibility="collapsed",
    )
    default_text = SAMPLES[sample_key]

    passage = st.text_area(
        "passage",
        value=default_text,
        height=240,
        placeholder="Paste any educational paragraph here…",
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label" style="margin-top:1.6rem">02 — Settings</div>', unsafe_allow_html=True)

    num_q = st.slider(
        "Number of questions",
        min_value=1, max_value=10, value=5,
        help="How many questions to generate (1 – 10)"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    gen_clicked = st.button("⚡  Generate Questions")


# ══════════════════════════════════════════════════════════════════════
# GENERATION LOGIC
# ══════════════════════════════════════════════════════════════════════
def generate_questions(passage: str, num_q: int):
    """
    Extract candidate answer spans (NER + noun chunks),
    then run them through the fine-tuned T5 model.
    Returns a list of question strings (no answers).
    """
    doc = nlp(passage)

    # Priority 1: Named entities
    candidates = [e.text.strip() for e in doc.ents
                  if len(e.text.strip().split()) <= 5 and len(e.text.strip()) > 1]

    # Priority 2: Noun chunks (fallback / top-up)
    if len(candidates) < num_q:
        for chunk in doc.noun_chunks:
            c = chunk.text.strip()
            if 1 < len(c.split()) <= 5 and c not in candidates:
                candidates.append(c)

    # De-duplicate while preserving order
    seen, unique = set(), []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    candidates = unique[:num_q]
    if not candidates:
        return []

    questions = []
    seen_q = set()
    for answer in candidates:
        # Build highlighted input
        highlighted = passage.replace(answer, f"<hl> {answer} <hl>", 1)
        inp = tokenizer(
            f"generate question: {highlighted}",
            return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        out = model.generate(
            **inp,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        q = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # Deduplicate questions
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            questions.append(q)

    return questions


# ── RIGHT — OUTPUT ────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">03 — Generated Questions</div>', unsafe_allow_html=True)

    if gen_clicked:
        if not passage or len(passage.strip()) < 40:
            st.warning("Please enter a more detailed passage (at least 40 characters) for best results.")
        else:
            with st.spinner("Generating questions with T5…"):
                questions = generate_questions(passage.strip(), num_q)

            if not questions:
                st.error("Could not extract answer candidates. Try a more detailed or factual passage.")
            else:
                # ── Result header ──────────────────────────────────
                st.markdown(f"""
                <div class="result-header">
                    <span class="result-count">⚡ {len(questions)} question{"s" if len(questions) != 1 else ""} generated</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Question cards (no answers) ────────────────────
                download_text = ""
                for i, q in enumerate(questions, 1):
                    st.markdown(f"""
                    <div class="q-outer">
                        <div class="q-inner">
                            <div class="q-num-badge">{i}</div>
                            <div class="q-text">{q}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    download_text += f"Q{i}. {q}\n\n"

                # ── Download ───────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📄  Download questions (.txt)",
                    data=download_text.strip(),
                    file_name="questions.txt",
                    mime="text/plain",
                )

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-label">Questions appear here</div>
            <p style="font-size:13px; color:#2e2a45; margin-top:10px; line-height:1.6">
                Choose a sample or paste your passage,<br>set the number of questions, then click generate.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="glow-divider" style="margin-top:3rem"></div>
<p style="text-align:center; color:#2e2a45; font-size:12px; padding:0.5rem 0 1rem; line-height:1.8">
    MPhil Data Science &nbsp;·&nbsp; NLP Course Project &nbsp;·&nbsp;
    <strong style="color:#3d3760">M. Hamza Sajjad &amp; Shanza Gul</strong> &nbsp;·&nbsp;
    Supervised by Dr. Adnan Abid<br>
    Model: <a href="https://huggingface.co/Hamzasajjad38/t5-small-qg"
              style="color:#5a547a; text-decoration:none">Hamzasajjad38/t5-small-qg</a>
    &nbsp;·&nbsp; Dataset:
    <a href="https://huggingface.co/datasets/rajpurkar/squad"
       style="color:#5a547a; text-decoration:none">SQuAD 1.1</a>
</p>
""", unsafe_allow_html=True)
