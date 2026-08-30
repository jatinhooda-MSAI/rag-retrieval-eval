import argparse
import json
from pathlib import Path

import anthropic
from pydantic import BaseModel

from chunk import STRATEGIES
from data import load_questions
from retrieve import RETRIEVERS, load_index

MODEL = "claude-opus-5-5"

ANSWER_PROMPT = """Answer the question using only the excerpts from SEC filings below. Show the figures you rely on. If the excerpts do not contain what you need, say so instead of guessing.

{context}

Question: {question}"""

CLAIMS_PROMPT = """Below are excerpts from SEC filings and an answer that was written from them.

{context}

<answer>
{answer}
</answer>

Split the answer into its individual factual claims: each figure, fact, calculation and conclusion. For each claim, decide whether the excerpts support it. A calculation counts as supported when its inputs appear in the excerpts and the arithmetic is right. A statement that the excerpts lack some information is a claim too, supported only if that is true."""

CORRECTNESS_PROMPT = """Question: {question}

<reference>
{reference}
</reference>

<candidate>
{answer}
</candidate>

Is the candidate answer correct? The key figures and the conclusion must agree with the reference. Rounding differences and extra detail are fine."""


class Answer(BaseModel):
    answer: str


class Claim(BaseModel):
    claim: str
    supported: bool


class Claims(BaseModel):
    claims: list[Claim]


class Verdict(BaseModel):
    correct: bool
    reason: str


def ask(client, prompt, output_format):
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        output_format=output_format,
        messages=[{"role": "user", "content": prompt}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    return response.parsed_output


def context(index, rows):
    return "\n\n".join(f'<excerpt source="{index.chunks.doc_name.iat[i]}">\n{index.texts[i]}\n</excerpt>' for i in rows)


def answer_and_judge(client, index, question, retriever, scope):
    rows = RETRIEVERS[retriever](index, question["question"], question["doc_name"] if scope == "doc" else None)[:5]
    excerpts = context(index, rows)
    answer = ask(client, ANSWER_PROMPT.format(context=excerpts, question=question["question"]), Answer).answer
    claims = ask(client, CLAIMS_PROMPT.format(context=excerpts, answer=answer), Claims).claims
    verdict = ask(client, CORRECTNESS_PROMPT.format(question=question["question"], reference=question["answer"], answer=answer), Verdict)
    return {
        "id": question["financebench_id"],
        "question": question["question"],
        "reference": question["answer"],
        "answer": answer,
        "claims": [c.model_dump() for c in claims],
        "faithfulness": sum(c.supported for c in claims) / len(claims) if claims else None,
        "correct": verdict.correct,
        "reason": verdict.reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", default="structure", choices=STRATEGIES)
    parser.add_argument("--retriever", default="rerank", choices=list(RETRIEVERS))
    parser.add_argument("--scope", default="doc", choices=["corpus", "doc"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    client = anthropic.Anthropic()
    index = load_index(args.chunker)
    out = Path(f"results/answers_{args.chunker}_{args.retriever}_{args.scope}.jsonl")
    rows = []
    with out.open("w") as f:
        for question in load_questions()[:args.limit]:
            row = answer_and_judge(client, index, question, args.retriever, args.scope)
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            faithfulness = "n/a" if row["faithfulness"] is None else f"{row['faithfulness']:.2f}"
            print(f"{row['id']}  faithfulness {faithfulness}  correct {row['correct']}")
    scored = [row["faithfulness"] for row in rows if row["faithfulness"] is not None]
    print(f"faithfulness {sum(scored) / len(scored):.3f} over {len(scored)} answers with claims")
    print(f"correctness {sum(row['correct'] for row in rows) / len(rows):.3f} over {len(rows)} answers")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
