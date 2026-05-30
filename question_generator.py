"""
Question Generation module using a T5 transformer model.

Pipeline:
    1. Receive (answer, context) pairs from preprocessing.
    2. Format input as  "answer: <answer> context: <context>"
    3. Generate questions using beam-search decoding.
"""
from transformers import T5ForConditionalGeneration, T5Tokenizer
from config import (
    QG_MODEL_NAME, DEVICE,
    MAX_INPUT_LENGTH, MAX_QUESTION_LENGTH,
    NUM_BEAMS, NUM_RETURN_SEQUENCES,
)

# ── Model lazy-loading ──────────────────────────────────────────────────────
_model = None
_tokenizer = None


def load_model(model_name: str = QG_MODEL_NAME):
    """Load T5 model and tokenizer (downloads on first run)."""
    global _model, _tokenizer
    if _model is None:
        print(f"[INFO] Loading model: {model_name} → {DEVICE}")
        _tokenizer = T5Tokenizer.from_pretrained(model_name)
        _model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        _model.eval()
    return _model, _tokenizer


def generate_question(answer: str, context: str) -> str:
    """
    Generate a single question given an answer span and its context.

    Args:
        answer:  The answer text (entity or noun-chunk).
        context: The surrounding sentence / paragraph.

    Returns:
        Generated question string.
    """
    model, tokenizer = load_model()

    # Highlight-based input format expected by valhalla/t5-small-qg-hl
    highlighted_context = context.replace(
        answer, f"<hl> {answer} <hl>"
    )
    input_text = f"generate question: {highlighted_context}"

    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(DEVICE)

    outputs = model.generate(
        inputs,
        max_length=MAX_QUESTION_LENGTH,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question.strip()


def generate_questions_from_spans(spans: list[dict]) -> list[dict]:
    """
    Generate questions for a batch of answer spans.

    Args:
        spans: List of dicts with 'answer' and 'sentence' keys
               (output of preprocessing.get_answer_spans).

    Returns:
        List of dicts with keys: answer, context, question, label.
    """
    results = []
    seen_questions = set()

    for span in spans:
        try:
            question = generate_question(span["answer"], span["sentence"])
        except Exception as e:
            print(f"[WARN] Skipping span {span['answer']!r}: {e}")
            continue

        # De-duplicate
        q_lower = question.lower()
        if q_lower in seen_questions:
            continue
        seen_questions.add(q_lower)

        results.append({
            "question": question,
            "answer": span["answer"],
            "context": span["sentence"],
            "label": span.get("label", ""),
        })

    return results


def generate_questions_from_text(text: str) -> list[dict]:
    """
    End-to-end: take raw text → return list of generated Q&A dicts.
    """
    from preprocessing import get_answer_spans
    spans = get_answer_spans(text)
    return generate_questions_from_spans(spans)


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Albert Einstein was a German-born theoretical physicist. "
        "He developed the theory of relativity in 1905 while working "
        "at the Swiss Patent Office in Bern."
    )
    for qa in generate_questions_from_text(sample):
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}  (label: {qa['label']})")
        print()
