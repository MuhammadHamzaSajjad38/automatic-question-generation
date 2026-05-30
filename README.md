# ⚡ Automatic Question Generation System
### NLP Course Project · MPhil Data Science

> A full-stack NLP application that automatically generates exam-ready questions from any educational passage, powered by a **T5-small Transformer** fine-tuned on **SQuAD 1.1**.

---

## 🌐 Live Demo & Model

| Resource | Link |
|---|---|
| 🤗 Fine-tuned Model | [Hamzasajjad38/t5-small-qg](https://huggingface.co/Hamzasajjad38/t5-small-qg) |
| 🚀 HF Space (Demo) | [huggingface.co/spaces/Hamzasajjad38/...](#) |
| 📦 Dataset | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) |

---

## 📊 Model Evaluation

Evaluated on 200 SQuAD 1.1 validation samples:

| Metric | Score |
|---|---|
| ROUGE-1 | **0.4509** |
| ROUGE-2 | **0.2445** |
| ROUGE-L | **0.4167** |
| BLEU | **0.1581** |

---

## 🗂️ Project Structure

```
📁 Automatic-Question-Generation/
│
├── 📁 frontend/                  # Standalone web UI (HTML/CSS/JS)
│   ├── index.html                # Main page
│   ├── style.css                 # Dark-mode premium design system
│   └── app.js                   # API calls & UI logic
│
├── api_server.py                 # Flask REST API (connects frontend ↔ backend)
├── app.py                        # Gradio interface (alternative UI)
├── config.py                     # Model & generation settings
├── preprocessing.py              # spaCy NER + noun chunk extraction
├── question_generator.py         # T5 model loading & question generation
├── mcq_generator.py              # MCQ distractor generation (WordNet + NER)
├── pdf_exporter.py               # PDF question paper export (fpdf2)
├── fine_tune.py                  # Fine-tuning script (full training loop)
├── NLP_Project_FineTuning_Evaluation.ipynb   # Google Colab notebook (end-to-end)
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/automatic-question-generation.git
cd automatic-question-generation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the backend API
```bash
python api_server.py
```
> The Flask server starts at `http://127.0.0.1:5000`

### 4. Open the frontend
Open `frontend/index.html` directly in your browser — no extra server needed.

---

## ⚙️ How It Works

```
User Input (text passage)
        ↓
  spaCy NER Pipeline
  (extract answer candidates)
        ↓
  T5 Transformer (fine-tuned)
  generate question: <hl> answer <hl> context
        ↓
  WordNet MCQ Distractor Generation
        ↓
  Output: Questions + MCQs + PDF Export
```

### Key Pipeline Steps
1. **Preprocessing** — spaCy extracts Named Entities (PERSON, ORG, DATE, GPE…) and noun chunks as answer candidates
2. **Question Generation** — Each answer + its sentence context is fed to the T5 model in the format `generate question: <hl> {answer} <hl> {context}`
3. **MCQ Generation** — WordNet synonyms/hypernyms and NER siblings are used as distractors
4. **Export** — Questions can be downloaded as a styled PDF

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status check |
| `POST` | `/generate` | Generate questions from text |
| `POST` | `/export_pdf` | Export Q&A to PDF |

### Example `/generate` request
```json
POST http://127.0.0.1:5000/generate
{
  "text": "Albert Einstein developed the theory of relativity in 1905...",
  "generate_mcq": true,
  "max_questions": 5
}
```

---

## 🧪 Fine-Tuning (Google Colab)

Open `NLP_Project_FineTuning_Evaluation.ipynb` in Google Colab:
- Runtime → **T4 GPU**
- Dataset streams directly from Hugging Face (no download needed)
- Model is pushed to Hugging Face Hub at the end automatically

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language Model | T5-small (60M params) |
| Framework | Hugging Face Transformers |
| NER | spaCy `en_core_web_sm` |
| MCQ | NLTK WordNet |
| Backend API | Flask + Flask-CORS |
| Frontend | HTML · CSS · Vanilla JS |
| PDF Export | fpdf2 |
| Training | Google Colab T4 GPU |

---

## 👥 Authors

- **M. Hamza Sajjad** — [Hamzasajjad38](https://huggingface.co/Hamzasajjad38)

📌 Supervised by **Dr. Adnan Abid** · MPhil Data Science

---

## 📄 License

This project is released for academic and educational purposes.
