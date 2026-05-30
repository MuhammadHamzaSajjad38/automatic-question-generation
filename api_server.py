"""
Automatic Question Generation System — REST API Server
=======================================================
Exposes JSON endpoints that the HTML frontend (frontend/index.html) calls.

Endpoints:
  POST /generate       — generate questions from a text passage
  POST /export_pdf     — export Q&A list to a PDF file (returns file stream)
  GET  /health         — simple health/status check

Usage:
    pip install flask flask-cors
    python api_server.py
"""

import os
import json
import tempfile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from preprocessing import get_answer_spans
from question_generator import generate_questions_from_spans
from mcq_generator import generate_mcqs
from pdf_exporter import export_questions_pdf

# ── App setup ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow the frontend (any origin) to call these endpoints


# ═══════════════════════════════════════════════════════════════════
# Helper: shared pipeline
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(text: str, generate_mcq: bool = False, max_questions: int = 10):
    """
    text → answer spans → questions → (optional MCQ)
    Returns (qa_list, status_message)
    """
    text = text.strip()
    if not text:
        return [], "No input provided."

    # Step 1: Extract answer spans (NER + noun chunks fallback)
    spans = get_answer_spans(text)
    if not spans:
        return [], "Could not extract any answer spans from the text."

    # Cap to max_questions
    spans = spans[:max_questions]

    # Step 2: Generate questions via fine-tuned T5 model
    qa_list = generate_questions_from_spans(spans)
    if not qa_list:
        return [], "No questions could be generated."

    # Step 3: Optionally enrich with MCQ distractors
    if generate_mcq:
        qa_list = generate_mcqs(qa_list, all_spans=spans)

    status = f"Generated {len(qa_list)} questions from {len(spans)} answer spans."
    return qa_list, status


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check — the frontend polls this on load."""
    return jsonify({"status": "ok", "model": "Hamzasajjad38/t5-small-qg"})


@app.route("/generate", methods=["POST"])
def generate():
    """
    Request body (JSON):
      {
        "text":          "<passage>",
        "generate_mcq":  true | false,
        "max_questions": 10
      }

    Response (JSON):
      {
        "qa_list": [ { question, answer, context, label, mcq? }, ... ],
        "status":  "<message>"
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    text          = data.get("text", "")
    generate_mcq  = bool(data.get("generate_mcq", False))
    max_questions = int(data.get("max_questions", 10))

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        qa_list, status = run_pipeline(text, generate_mcq, max_questions)
        return jsonify({"qa_list": qa_list, "status": status})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    """
    Request body (JSON):
      {
        "qa_list":         [ ... ],
        "include_answers": true | false,
        "include_mcq":     true | false
      }

    Response: PDF file download stream.
    """
    data = request.get_json(force=True, silent=True) or {}
    qa_list         = data.get("qa_list", [])
    include_answers = bool(data.get("include_answers", True))
    include_mcq     = bool(data.get("include_mcq", False))

    if not qa_list:
        return jsonify({"error": "No Q&A data to export."}), 400

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="questions_")
        path = export_questions_pdf(
            qa_list,
            filepath=tmp.name,
            include_answers=include_answers,
            include_mcq=include_mcq,
        )
        return send_file(
            path,
            as_attachment=True,
            download_name="question_paper.pdf",
            mimetype="application/pdf",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AQG API Server — QuestionForge Frontend Backend")
    print("  Model: Hamzasajjad38/t5-small-qg")
    print("  Listening on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
