"""
MCQ (Multiple Choice Question) Generator.

Generates distractor options for each answer using:
  - WordNet synonyms / hypernyms / hyponyms
  - NER siblings (other entities of the same label in the text)
"""
import random
import nltk
from nltk.corpus import wordnet as wn
from config import MCQ_NUM_DISTRACTORS

# Ensure WordNet data is available
try:
    wn.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def _wordnet_distractors(word: str, n: int = 10) -> list[str]:
    """
    Collect candidate distractors from WordNet relations.
    """
    candidates = set()
    for syn in wn.synsets(word):
        # Synonyms
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                candidates.add(name)
        # Hypernyms
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    candidates.add(name)
        # Hyponyms
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    candidates.add(name)

    return list(candidates)[:n]


def _ner_sibling_distractors(
    answer: str, label: str, all_spans: list[dict]
) -> list[str]:
    """
    Get other entities with the same NER label as distractors.
    """
    return [
        s["answer"]
        for s in all_spans
        if s["label"] == label and s["answer"].lower() != answer.lower()
    ]


def generate_mcq_options(
    answer: str,
    label: str = "",
    all_spans: list[dict] | None = None,
    num_distractors: int = MCQ_NUM_DISTRACTORS,
) -> dict:
    """
    Generate MCQ options for a given answer.

    Returns:
        dict with keys:
            - correct: the correct answer
            - distractors: list of distractor strings
            - options: shuffled list of all options (correct + distractors)
    """
    distractors: list[str] = []

    # 1. Try NER siblings first
    if all_spans and label:
        distractors += _ner_sibling_distractors(answer, label, all_spans)

    # 2. Fill remaining from WordNet
    if len(distractors) < num_distractors:
        # Try each word in the answer
        for token in answer.split():
            distractors += _wordnet_distractors(token)
            if len(distractors) >= num_distractors:
                break

    # 3. Fallback generic distractors
    fallbacks = ["None of the above", "All of the above", "Not mentioned", "Cannot be determined"]
    while len(distractors) < num_distractors:
        fb = fallbacks.pop(0) if fallbacks else f"Option {len(distractors)+1}"
        if fb not in distractors:
            distractors.append(fb)

    # De-duplicate and trim
    seen = {answer.lower()}
    unique = []
    for d in distractors:
        if d.lower() not in seen:
            seen.add(d.lower())
            unique.append(d)
        if len(unique) == num_distractors:
            break
    distractors = unique

    options = [answer] + distractors
    random.shuffle(options)

    return {
        "correct": answer,
        "distractors": distractors,
        "options": options,
    }


def generate_mcqs(qa_list: list[dict], all_spans: list[dict] | None = None) -> list[dict]:
    """
    Add MCQ options to each Q&A dict.

    Args:
        qa_list:   Output of question_generator.generate_questions_from_spans
        all_spans: All extracted answer spans (for sibling distractors)

    Returns:
        Enriched list with 'mcq' key added to each dict.
    """
    results = []
    for qa in qa_list:
        mcq = generate_mcq_options(
            answer=qa["answer"],
            label=qa.get("label", ""),
            all_spans=all_spans,
        )
        results.append({**qa, "mcq": mcq})
    return results


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_qa = {
        "question": "Where was Einstein born?",
        "answer": "Germany",
        "context": "Albert Einstein was born in Germany.",
        "label": "GPE",
    }
    result = generate_mcq_options("Germany", label="GPE")
    print("Correct:", result["correct"])
    print("Distractors:", result["distractors"])
    print("Options:", result["options"])
