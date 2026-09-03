import re
import time
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from chunk import STRATEGIES, documents
from data import load_questions
from retrieve import RETRIEVERS, load_index

WORD = re.compile(r"[a-z0-9]+", re.I)
KS = (1, 5, 10, 20)
METRICS = [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10", "filing@10"]
SCOPES = ("corpus", "doc")
DISCOUNTS = 1 / np.log2(np.arange(2, 12))
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def locate(text, page_starts, page, evidence):
    start = page_starts[page]
    end = page_starts[page + 1] - 2 if page + 1 < len(page_starts) else len(text)
    words = [(m.start(), m.group().lower()) for m in WORD.finditer(text, start, end)]
    target = [w.lower() for w in WORD.findall(evidence)]
    blocks = SequenceMatcher(None, [w for _, w in words], target, autojunk=False).get_matching_blocks()
    positions = np.array([words[b.a + i][0] for b in blocks for i in range(b.size)])
    return positions, len(positions) / len(target)


def locate_evidence(questions):
    docs = {doc: (text, page_starts) for doc, text, page_starts in documents()}
    evidence = []
    for qi, q in enumerate(questions):
        text, page_starts = docs[q["doc_name"]]
        for ev in q["evidence"]:
            page = ev["evidence_page_num"]
            found = {off: locate(text, page_starts, page + off, ev["evidence_text"]) for off in (-1, 0, 1) if 0 <= page + off < len(page_starts)}
            evidence.append({
                "question": qi,
                "id": q["financebench_id"],
                "doc_name": q["doc_name"],
                "page": page,
                "tokens": len(WORD.findall(ev["evidence_text"])),
                "match": round(found[0][1], 3),
                "best_offset": max(found, key=lambda off: found[off][1]),
                "positions": found[0][0],
            })
    return evidence


def label(chunks, evidence, n_questions):
    doc_rows = chunks.groupby("doc_name").indices
    start, end = chunks.start.to_numpy(), chunks.end.to_numpy()
    first, last = chunks.page_start.to_numpy(), chunks.page_end.to_numpy()
    relevant = [[] for _ in range(n_questions)]
    gains = [{} for _ in range(n_questions)]
    for ev in evidence:
        rows, pos = doc_rows[ev["doc_name"]], ev["positions"]
        coverage = (np.searchsorted(pos, end[rows]) - np.searchsorted(pos, start[rows])) / len(pos)
        on_page = (first[rows] <= ev["page"]) & (ev["page"] <= last[rows])
        grade = np.where(coverage >= 0.5, 2, np.where(on_page, 1, 0))
        relevant[ev["question"]].append(set(rows[grade == 2]))
        for i, g in zip(rows[grade > 0], grade[grade > 0]):
            gains[ev["question"]][i] = max(gains[ev["question"]].get(i, 0), g)
    return relevant, gains


def score(ranking, relevant, gains):
    row = {f"recall@{k}": np.mean([bool(s & set(ranking[:k])) for s in relevant]) for k in KS}
    hits = [r for r, i in enumerate(ranking[:10], 1) if any(i in s for s in relevant)]
    row["mrr@10"] = 1 / hits[0] if hits else 0.0
    dcg = sum(gains.get(i, 0) * d for i, d in zip(ranking[:10], DISCOUNTS))
    ideal = sum(g * d for g, d in zip(sorted(gains.values(), reverse=True), DISCOUNTS))
    row["ndcg@10"] = dcg / ideal
    return row


def heatmap(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    titles = {"corpus": "Whole corpus", "doc": "Filtered to the question's filing"}
    for ax, scope in zip(axes, SCOPES):
        grid = summary[summary.scope == scope].pivot(index="chunker", columns="retriever", values="recall@10").loc[STRATEGIES, list(RETRIEVERS)]
        values = grid.to_numpy()
        ax.pcolormesh(values, cmap=BLUES, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
        for (r, c), v in np.ndenumerate(values):
            ax.text(c + 0.5, r + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=11, color="white" if v > 0.55 else INK)
        ax.set_xticks(np.arange(values.shape[1]) + 0.5, grid.columns)
        ax.set_yticks(np.arange(values.shape[0]) + 0.5, grid.index)
        ax.invert_yaxis()
        ax.tick_params(length=0, labelcolor=MUTED)
        ax.set_facecolor(SURFACE)
        ax.set_title(titles[scope], color=INK, fontsize=11, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Evidence recall@10 by chunker and retriever", color=INK, fontsize=13, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig("results/heatmap.png", dpi=150, facecolor=SURFACE)


def main():
    questions = load_questions()
    evidence = locate_evidence(questions)
    ev = pd.DataFrame(evidence).drop(columns="positions")
    ev.to_csv("results/evidence.csv", index=False)
    print(f"{len(ev)} evidence passages, best page offset counts: {ev.best_offset.value_counts().to_dict()}, token match median {ev.match.median():.2f}, lowest {ev.match.min():.2f}")
    rows, chunkers = [], []
    for strategy in STRATEGIES:
        index = load_index(strategy)
        relevant, gains = label(index.chunks, evidence, len(questions))
        filings = index.chunks.doc_name.to_numpy()
        coverable = float(np.mean([len(s) > 0 for rel in relevant for s in rel]))
        chunkers.append({"chunker": strategy, "chunks": len(index.texts), "mean_tokens": round(index.chunks.n_tokens.mean()), "coverable_evidence": round(coverable, 3)})
        print(chunkers[-1])
        for scope in SCOPES:
            for name, retrieve in RETRIEVERS.items():
                for qi, q in enumerate(questions):
                    start = time.perf_counter()
                    ranking = list(retrieve(index, q["question"], q["doc_name"] if scope == "doc" else None))
                    latency = time.perf_counter() - start
                    rows.append({"chunker": strategy, "retriever": name, "scope": scope, "id": q["financebench_id"], **score(ranking, relevant[qi], gains[qi]), "filing@10": np.mean(filings[ranking[:10]] == q["doc_name"]), "latency_ms": 1000 * latency})
                print(strategy, scope, name, f"recall@10 {np.mean([r['recall@10'] for r in rows[-len(questions):]]):.3f}")
    per_question = pd.DataFrame(rows)
    per_question.round(4).to_csv("results/per_question.csv", index=False)
    pd.DataFrame(chunkers).to_csv("results/chunkers.csv", index=False)
    groups = per_question.groupby(["chunker", "retriever", "scope"], sort=False)
    summary = groups[METRICS].mean().round(3)
    summary["p50_ms"] = groups.latency_ms.median().round(1)
    summary = summary.reset_index()
    summary.to_csv("results/retrieval.csv", index=False)
    heatmap(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
