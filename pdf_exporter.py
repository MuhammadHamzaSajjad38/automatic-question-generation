"""
PDF Exporter — generates a neatly formatted question paper as PDF.
Uses fpdf2 library.
"""
import os
from datetime import datetime
from fpdf import FPDF
from config import PDF_TITLE, PDF_FONT_FAMILY, PDF_FONT_SIZE


class QuestionPaperPDF(FPDF):
    """Custom PDF class with header and footer."""

    def __init__(self, title: str = PDF_TITLE, **kwargs):
        super().__init__(**kwargs)
        self._title = title

    def header(self):
        self.set_font(PDF_FONT_FAMILY, "B", 16)
        self.cell(0, 12, self._title, ln=True, align="C")
        self.set_font(PDF_FONT_FAMILY, "", 9)
        self.cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.ln(6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(PDF_FONT_FAMILY, "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def export_questions_pdf(
    qa_list: list[dict],
    filepath: str = "question_paper.pdf",
    title: str = PDF_TITLE,
    include_answers: bool = True,
    include_mcq: bool = False,
) -> str:
    """
    Export generated questions to a styled PDF file.

    Args:
        qa_list:          List of Q&A dicts (from question_generator).
        filepath:         Output PDF path.
        title:            Title printed on the paper.
        include_answers:  Whether to include an answer key at the end.
        include_mcq:      Whether to render MCQ options (requires 'mcq' key).

    Returns:
        Absolute path to the generated PDF.
    """
    pdf = QuestionPaperPDF(title=title)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Questions Section ───────────────────────────────────────────────
    pdf.set_font(PDF_FONT_FAMILY, "B", 13)
    pdf.cell(0, 10, "Questions", ln=True)
    pdf.ln(2)

    for idx, qa in enumerate(qa_list, 1):
        pdf.set_font(PDF_FONT_FAMILY, "B", PDF_FONT_SIZE)
        pdf.multi_cell(0, 7, f"Q{idx}. {qa['question']}")

        if include_mcq and "mcq" in qa:
            pdf.set_font(PDF_FONT_FAMILY, "", PDF_FONT_SIZE - 1)
            for opt_idx, option in enumerate(qa["mcq"]["options"]):
                letter = chr(65 + opt_idx)  # A, B, C, D
                pdf.cell(10)  # indent
                pdf.cell(0, 6, f"    {letter}) {option}", ln=True)

        pdf.ln(4)

    # ── Answer Key Section ──────────────────────────────────────────────
    if include_answers:
        pdf.add_page()
        pdf.set_font(PDF_FONT_FAMILY, "B", 13)
        pdf.cell(0, 10, "Answer Key", ln=True)
        pdf.ln(2)

        for idx, qa in enumerate(qa_list, 1):
            pdf.set_font(PDF_FONT_FAMILY, "", PDF_FONT_SIZE)
            pdf.multi_cell(0, 7, f"Q{idx}. {qa['answer']}")
            pdf.ln(2)

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    pdf.output(filepath)
    return os.path.abspath(filepath)


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
        {"question": "In which year was the theory published?", "answer": "1905"},
    ]
    path = export_questions_pdf(sample, "test_paper.pdf")
    print(f"PDF saved → {path}")
