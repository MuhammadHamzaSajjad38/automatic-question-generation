"""
Preprocessing module:
  - Sentence segmentation using spaCy
  - Named Entity Recognition (NER) to extract answer spans
"""
import spacy
from config import SPACY_MODEL, ANSWER_NER_LABELS


def load_spacy_model():
    """Load spaCy model, downloading it if necessary."""
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"[INFO] Downloading spaCy model: {SPACY_MODEL} ...")
        from spacy.cli import download
        download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
    return nlp


# Global spaCy model instance
_nlp = None


def get_nlp():
    """Lazy-load and return the spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = load_spacy_model()
    return _nlp


def segment_sentences(text: str) -> list[str]:
    """
    Split text into sentences using spaCy's sentence segmenter.

    Args:
        text: Raw input paragraph or document.

    Returns:
        List of sentence strings.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def extract_answer_spans(text: str) -> list[dict]:
    """
    Use NER to extract potential answer spans from the text.

    Each span is a dict with keys:
        - answer:  the entity text
        - label:   the NER label (e.g. PERSON, ORG)
        - start:   character start offset
        - end:     character end offset
        - sentence: the enclosing sentence text

    Args:
        text: Input text passage.

    Returns:
        List of answer-span dicts.
    """
    nlp = get_nlp()
    doc = nlp(text)

    spans = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in ANSWER_NER_LABELS:
            continue
        key = (ent.text.strip(), ent.sent.text.strip())
        if key in seen:
            continue
        seen.add(key)
        spans.append({
            "answer": ent.text.strip(),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "sentence": ent.sent.text.strip(),
        })

    return spans


def extract_noun_chunks(text: str) -> list[dict]:
    """
    Fallback answer extraction using noun chunks when NER yields
    too few candidates.

    Returns:
        List of answer-span dicts (label is 'NOUN_CHUNK').
    """
    nlp = get_nlp()
    doc = nlp(text)

    spans = []
    seen = set()
    for chunk in doc.noun_chunks:
        clean = chunk.text.strip()
        if len(clean.split()) < 2:
            continue  # skip single-word chunks
        sent = chunk.sent.text.strip()
        key = (clean, sent)
        if key in seen:
            continue
        seen.add(key)
        spans.append({
            "answer": clean,
            "label": "NOUN_CHUNK",
            "start": chunk.start_char,
            "end": chunk.end_char,
            "sentence": sent,
        })

    return spans


def get_answer_spans(text: str, min_spans: int = 3) -> list[dict]:
    """
    Get answer spans from a passage.  Uses NER first; falls back to
    noun-chunk extraction if fewer than `min_spans` entities are found.

    Args:
        text:       Input text.
        min_spans:  Minimum number of spans before falling back.

    Returns:
        List of answer-span dicts.
    """
    spans = extract_answer_spans(text)
    if len(spans) < min_spans:
        spans += extract_noun_chunks(text)
    return spans


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Albert Einstein was a German-born theoretical physicist. "
        "He developed the theory of relativity in 1905 while working "
        "at the Swiss Patent Office in Bern."
    )
    print("Sentences:", segment_sentences(sample))
    print()
    for span in get_answer_spans(sample):
        print(f"  [{span['label']}] {span['answer']!r}  →  {span['sentence']!r}")
