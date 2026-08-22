import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

MODEL = "BAAI/bge-small-en-v1.5"
TOKENIZER = Tokenizer.from_pretrained(MODEL)
EMBEDDER = SentenceTransformer(MODEL)
HEADING = re.compile(r"\s*(item\s+\d{1,2}(\.\d{2})?[a-c]?\b|part\s+i{1,3}v?\b)", re.I)
CHUNKS = Path("data/chunks")
FIXED = {"fixed256": (256, 32), "fixed512": (512, 64)}
STRATEGIES = ["fixed256", "fixed512", "structure", "semantic"]


def documents():
    pages = pd.read_parquet("data/pages.parquet")
    for doc, group in pages.groupby("doc_name"):
        texts = group.sort_values("page").text.tolist()
        page_starts = np.cumsum([0] + [len(t) + 2 for t in texts[:-1]])
        yield doc, "\n\n".join(texts), page_starts


def embed(texts):
    return EMBEDDER.encode(texts, normalize_embeddings=True)


def fixed(spans, size, overlap):
    return [(i, min(i + size, len(spans))) for i in range(0, max(len(spans) - overlap, 1), size - overlap)]


def paragraphs(text, page_starts):
    page_ends = list(page_starts[1:] - 2) + [len(text)]
    for start, end in zip(page_starts, page_ends):
        lines = text[start:end].split("\n")
        width = max(map(len, lines))
        pos = para = start
        for k, line in enumerate(lines):
            pos += len(line) + 1
            if len(line) < 0.75 * width or k == len(lines) - 1:
                if text[para:pos - 1].strip():
                    yield para, pos - 1
                para = pos


def merge(units, spans, breaks, limit=400, min_size=64):
    bounds, a = [], None
    for (s, e), brk in zip(units, breaks):
        i, j = np.searchsorted(spans[:, 0], [s, e])
        if a is not None and (j - a > limit or (brk and i - a >= min_size)):
            bounds.append((a, i))
            a = None
        if a is None:
            a = i
        while j - a > limit:
            bounds.append((a, a + limit))
            a += limit
    return bounds + [(a, len(spans))]


def semantic_breaks(text, units, percentile=90):
    emb = embed([text[s:e] for s, e in units])
    dist = 1 - (emb[:-1] * emb[1:]).sum(1)
    return [False, *(dist > np.percentile(dist, percentile))]


def chunk_document(strategy, doc, text, page_starts):
    spans = np.array(TOKENIZER.encode(text, add_special_tokens=False).offsets)
    if strategy in FIXED:
        bounds = fixed(spans, *FIXED[strategy])
    else:
        units = list(paragraphs(text, page_starts))
        if strategy == "structure":
            breaks = [bool(HEADING.match(text, s)) for s, _ in units]
        else:
            breaks = semantic_breaks(text, units)
        bounds = merge(units, spans, breaks)
    rows = []
    for a, b in bounds:
        start, end = int(spans[a][0]), int(spans[b - 1][1])
        rows.append({
            "doc_name": doc,
            "start": start,
            "end": end,
            "page_start": int(np.searchsorted(page_starts, start, "right") - 1),
            "page_end": int(np.searchsorted(page_starts, end - 1, "right") - 1),
            "n_tokens": b - a,
            "text": text[start:end],
        })
    return rows


def load_chunks(strategy):
    return pd.read_parquet(CHUNKS / f"{strategy}.parquet")


def main():
    CHUNKS.mkdir(parents=True, exist_ok=True)
    docs = list(documents())
    for strategy in STRATEGIES:
        start = time.perf_counter()
        chunks = pd.DataFrame([row for doc in docs for row in chunk_document(strategy, *doc)])
        seconds = time.perf_counter() - start
        chunks.to_parquet(CHUNKS / f"{strategy}.parquet")
        n = chunks.n_tokens
        print(f"{strategy:10s} {len(chunks):6d} chunks  tokens mean {n.mean():.0f}  p10 {n.quantile(0.1):.0f}  p90 {n.quantile(0.9):.0f}  {seconds:.0f}s")


if __name__ == "__main__":
    main()
