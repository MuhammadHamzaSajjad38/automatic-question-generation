"""
Configuration constants for the Automatic Question Generation System.
"""
import torch

# ── Model Settings ──────────────────────────────────────────────────────────
# Pre-fine-tuned T5 model for question generation (trained on SQuAD)
QG_MODEL_NAME = "valhalla/t5-small-qg-hl"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Generation Hyper-parameters ─────────────────────────────────────────────
MAX_INPUT_LENGTH = 512
MAX_QUESTION_LENGTH = 64
NUM_BEAMS = 4
NUM_RETURN_SEQUENCES = 1     # questions per answer span

# ── spaCy Model ─────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── NER Labels to consider as answer candidates ─────────────────────────────
ANSWER_NER_LABELS = {
    "PERSON", "ORG", "GPE", "DATE", "EVENT",
    "WORK_OF_ART", "LOC", "NORP", "FAC", "PRODUCT",
    "MONEY", "QUANTITY", "PERCENT", "TIME", "CARDINAL", "ORDINAL",
}

# ── MCQ Settings ────────────────────────────────────────────────────────────
MCQ_NUM_DISTRACTORS = 3

# ── PDF Export Settings ─────────────────────────────────────────────────────
PDF_TITLE = "Generated Question Paper"
PDF_FONT_FAMILY = "Helvetica"
PDF_FONT_SIZE = 12
