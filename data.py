import json
import urllib.request
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium

QUESTIONS_URL = "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"
DATA = Path("data")


def load_questions():
    return [json.loads(line) for line in (DATA / "questions.jsonl").open()]


def main():
    (DATA / "pdfs").mkdir(parents=True, exist_ok=True)
    if not (DATA / "questions.jsonl").exists():
        urllib.request.urlretrieve(QUESTIONS_URL, DATA / "questions.jsonl")
    questions = load_questions()
    docs = sorted({q["doc_name"] for q in questions} | {e["doc_name"] for q in questions for e in q["evidence"]})
    rows = []
    for doc in docs:
        path = DATA / "pdfs" / f"{doc}.pdf"
        if not path.exists():
            urllib.request.urlretrieve(PDF_URL.format(doc), path)
        for i, page in enumerate(pdfium.PdfDocument(path)):
            rows.append({"doc_name": doc, "page": i, "text": page.get_textpage().get_text_bounded().replace("\r\n", "\n")})
    pages = pd.DataFrame(rows)
    pages.to_parquet(DATA / "pages.parquet")
    size = sum((DATA / "pdfs" / f"{d}.pdf").stat().st_size for d in docs) / 1e6
    print(f"{len(questions)} questions, {len(docs)} docs ({size:.0f} MB of PDFs), {len(pages)} pages")
    print(f"empty pages: {(pages.text.str.strip() == '').sum()}")


if __name__ == "__main__":
    main()
