"""
Automatic Question Generation System — Main Application

Provides:
  - Gradio web interface for interactive question generation
  - Flask API endpoints for programmatic access

Usage:
    python app.py
"""
import os
import json
import tempfile
import gradio as gr

from preprocessing import get_answer_spans, segment_sentences
from question_generator import generate_questions_from_spans
from mcq_generator import generate_mcqs
from pdf_exporter import export_questions_pdf


# ═══════════════════════════════════════════════════════════════════════════
# Core Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    text: str,
    generate_mcq: bool = False,
    max_questions: int = 10,
):
    """
    Full pipeline: text → answer spans → questions → (optional MCQ) → output.

    Returns:
        formatted_text: Markdown-formatted Q&A output
        qa_list:        Raw list of Q&A dicts
        status:         Status message
    """
    if not text or not text.strip():
        return "⚠️ Please enter some text.", [], "No input provided."

    text = text.strip()

    # Step 1: Extract answer spans
    spans = get_answer_spans(text)
    if not spans:
        return (
            "⚠️ Could not extract any answer spans from the text. "
            "Try providing a longer or more factual passage.",
            [],
            "No answer spans found.",
        )

    # Limit spans
    spans = spans[:max_questions]

    # Step 2: Generate questions
    qa_list = generate_questions_from_spans(spans)
    if not qa_list:
        return "⚠️ No questions could be generated.", [], "Generation failed."

    # Step 3: Optional MCQ
    if generate_mcq:
        qa_list = generate_mcqs(qa_list, all_spans=spans)

    # Step 4: Format output as Markdown
    lines = []
    for idx, qa in enumerate(qa_list, 1):
        lines.append(f"### Q{idx}. {qa['question']}")
        lines.append(f"**Answer:** {qa['answer']}")
        lines.append(f"**Context:** _{qa['context']}_")

        if generate_mcq and "mcq" in qa:
            lines.append("\n**Options:**")
            for opt_idx, option in enumerate(qa["mcq"]["options"]):
                letter = chr(65 + opt_idx)
                marker = " ✅" if option == qa["mcq"]["correct"] else ""
                lines.append(f"- **{letter})** {option}{marker}")

        lines.append("---")

    formatted = "\n\n".join(lines)
    status = f"✅ Generated {len(qa_list)} questions from {len(spans)} answer spans."

    return formatted, qa_list, status


def export_to_pdf(qa_json: str, include_answers: bool, include_mcq: bool):
    """Export the generated Q&A to a PDF file."""
    if not qa_json:
        return None, "⚠️ No questions to export. Generate questions first."

    try:
        qa_list = json.loads(qa_json) if isinstance(qa_json, str) else qa_json
    except (json.JSONDecodeError, TypeError):
        return None, "⚠️ Invalid Q&A data."

    if not qa_list:
        return None, "⚠️ No questions to export."

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="questions_")
    path = export_questions_pdf(
        qa_list,
        filepath=tmp.name,
        include_answers=include_answers,
        include_mcq=include_mcq,
    )
    return path, f"✅ PDF exported: {os.path.basename(path)}"


# ═══════════════════════════════════════════════════════════════════════════
# Gradio Interface
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_TEXT = """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity, but he also made important contributions to quantum mechanics. His mass-energy equivalence formula E = mc², which arises from relativity theory, has been called "the world's most famous equation". He received the Nobel Prize in Physics in 1921 for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect. His work is also known for its influence on the philosophy of science. In 1999, he was named Time magazine's "Person of the Century"."""

CSS = """
/* ── Global Theme ────────────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ── Header ──────────────────────────────────────────────────────────── */
#header-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}
#header-subtitle {
    text-align: center;
    color: #6b7280 !important;
    font-size: 1rem !important;
    margin-top: 4px !important;
}

/* ── Cards / Panels ──────────────────────────────────────────────────── */
.panel {
    border-radius: 12px !important;
    border: 1px solid rgba(102, 126, 234, 0.15) !important;
    padding: 16px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
#generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    transition: all 0.3s ease !important;
}
#generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}
#export-btn {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
}
#clear-btn {
    border-radius: 10px !important;
}
"""

def build_ui():
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(css=CSS, title="Auto Question Generator | NLP Project", theme=gr.themes.Soft()) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown("# 🧠 Automatic Question Generation System", elem_id="header-title")
        gr.Markdown(
            "Powered by **T5 Transformer** · spaCy NER · WordNet MCQs · PDF Export",
            elem_id="header-subtitle",
        )

        # Hidden state for Q&A JSON
        qa_state = gr.State([])

        # ── Input Section ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    label="📝 Input Text Passage",
                    placeholder="Paste a paragraph or passage here...",
                    lines=10,
                    value=SAMPLE_TEXT,
                )
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                mcq_toggle = gr.Checkbox(label="Generate MCQs", value=True)
                max_q_slider = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Max Questions",
                )
                generate_btn = gr.Button("🚀 Generate Questions", elem_id="generate-btn", variant="primary")
                clear_btn = gr.ClearButton([input_text], value="🗑️ Clear", elem_id="clear-btn")

        # ── Status ──────────────────────────────────────────────────────
        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)

        # ── Output Section ──────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## 📋 Generated Questions")
        output_md = gr.Markdown(value="_Questions will appear here after generation..._")

        # ── PDF Export Section ──────────────────────────────────────────
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Export to PDF")
                with gr.Row():
                    pdf_answers = gr.Checkbox(label="Include Answer Key", value=True)
                    pdf_mcq = gr.Checkbox(label="Include MCQ Options", value=True)
                export_btn = gr.Button("📥 Download PDF", elem_id="export-btn", variant="secondary")
            with gr.Column(scale=1):
                pdf_status = gr.Textbox(label="Export Status", interactive=False, max_lines=1)
                pdf_file = gr.File(label="📎 Download", type="filepath")

        # ── Extracted Spans Preview ─────────────────────────────────────
        with gr.Accordion("🔍 Extracted Answer Spans (Debug)", open=False):
            spans_json = gr.JSON(label="Answer Spans")

        # ── Event Handlers ──────────────────────────────────────────────

        def on_generate(text, mcq, max_q):
            formatted, qa_list, status = run_pipeline(text, mcq, max_q)
            spans = get_answer_spans(text) if text.strip() else []
            return formatted, qa_list, status, spans

        generate_btn.click(
            fn=on_generate,
            inputs=[input_text, mcq_toggle, max_q_slider],
            outputs=[output_md, qa_state, status_box, spans_json],
        )

        def on_export(qa_list, inc_ans, inc_mcq):
            qa_json = json.dumps(qa_list) if isinstance(qa_list, list) else qa_list
            path, msg = export_to_pdf(qa_json, inc_ans, inc_mcq)
            return path, msg

        export_btn.click(
            fn=on_export,
            inputs=[qa_state, pdf_answers, pdf_mcq],
            outputs=[pdf_file, pdf_status],
        )

        # ── Footer ──────────────────────────────────────────────────────
        gr.Markdown(
            "<center style='color:#9ca3af; margin-top:20px;'>"
            "MPhil Data Science · NLP Course Project · M. Hamza Sajjad & Shanza Gul · "
            "Supervised by Dr. Adnan Abid"
            "</center>"
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Automatic Question Generation System")
    print("  NLP Course Project — MPhil Data Science")
    print("=" * 60)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
