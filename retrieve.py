import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from sentence_transformers import CrossEncoder

from chunk import STRATEGIES, embed, load_chunks

EMB = Path("data/emb")
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
STEMMER = Stemmer.Stemmer("english")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class Index(NamedTuple):
    chunks: pd.DataFrame
    texts: list[str]
    doc_rows: dict[str, np.ndarray]
    bm25: bm25s.BM25
    emb: np.ndarray


def tokenize(texts, return_ids=True):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, return_ids=return_ids, show_progress=False)


def build_bm25(texts):
    bm25 = bm25s.BM25()
    bm25.index(tokenize(texts), show_progress=False)
    return bm25


def load_index(strategy):
    chunks = load_chunks(strategy)
    texts = chunks.text.tolist()
    doc_rows = chunks.groupby("doc_name").indices
    return Index(chunks, texts, doc_rows, build_bm25(texts), np.load(EMB / f"{strategy}.npy"))


def top(index, scores, doc, k):
    if doc is None:
        return np.argsort(-scores)[:k]
    rows = index.doc_rows[doc]
    return rows[np.argsort(-scores[rows])[:k]]


def bm25(index, query, doc=None, k=100):
    return top(index, index.bm25.get_scores(tokenize([query], return_ids=False)[0]), doc, k)


def dense(index, query, doc=None, k=100):
    return top(index, index.emb @ embed([QUERY_PROMPT + query])[0], doc, k)


def hybrid(index, query, doc=None, k=100):
    fused = defaultdict(float)
    for ranking in (bm25(index, query, doc), dense(index, query, doc)):
        for rank, i in enumerate(ranking, 1):
            fused[i] += 1 / (60 + rank)
    return np.array(sorted(fused, key=fused.get, reverse=True)[:k])


def rerank(index, query, doc=None, k=50):
    candidates = hybrid(index, query, doc)[:k]
    scores = RERANKER.predict([(query, index.texts[i]) for i in candidates], batch_size=k)
    return candidates[np.argsort(-scores)]


RETRIEVERS = {"bm25": bm25, "dense": dense, "hybrid": hybrid, "rerank": rerank}


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in STRATEGIES:
        texts = load_chunks(strategy).text.tolist()
        start = time.perf_counter()
        build_bm25(texts)
        bm25_s = time.perf_counter() - start
        start = time.perf_counter()
        emb = embed(texts)
        dense_s = time.perf_counter() - start
        np.save(EMB / f"{strategy}.npy", emb)
        rows.append({"chunker": strategy, "chunks": len(texts), "bm25_build_s": round(bm25_s, 1), "dense_build_s": round(dense_s, 1)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("results/index_build.csv", index=False)


if __name__ == "__main__":
    main()
