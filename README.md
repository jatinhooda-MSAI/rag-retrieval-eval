# rag-retrieval-eval

Retrieval evaluation for RAG over SEC filings. It takes the 150 open-source FinanceBench questions and the 84 filings they cite, crosses four chunking strategies with BM25, dense, hybrid and reranked retrieval, and scores every combination on evidence recall@k, MRR@10 and nDCG@10, searching either the whole corpus or only the question's filing. An answer-and-judge step for faithfulness is included but has not been run.

## Running it

Needs Python 3.12 and uv. Everything except `answer.py` runs locally without API keys.

```
uv sync
uv run data.py       # questions from HF, the 84 cited PDFs (166 MB) from GitHub, page text -> data/pages.parquet
uv run chunk.py      # four chunkings -> data/chunks/
uv run retrieve.py   # BM25 indexes, bge-small embeddings -> data/emb/, results/index_build.csv
uv run evaluate.py   # labels, every chunker x retriever x scope -> results/
```

Each step's printed output from the run below is in `results/*.log`. Embedding takes most of the time. `results/retrieval.csv` has recall@1/5/10/20, MRR@10, nDCG@10 and p50 latency for all 32 runs, and `results/per_question.csv` has the per-question scores.

## Results

"corpus" searches all 84 filings, "filing" searches only the question's filing (`doc` in the CSVs and on the command line). "own filing in top 10" is the share of the whole-corpus top 10 that came from the question's filing. p50 is per query in the whole-corpus scope, including query embedding and the cross-encoder, on an M1 laptop (MPS).

| chunker | retriever | R@10 corpus | R@10 filing | MRR@10 corpus | MRR@10 filing | nDCG@10 corpus | nDCG@10 filing | own filing in top 10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| fixed256 | bm25 | 0.107 | 0.221 | 0.064 | 0.113 | 0.068 | 0.179 | 0.21 | 2 |
| fixed256 | dense | 0.201 | 0.284 | 0.112 | 0.183 | 0.139 | 0.250 | 0.36 | 21 |
| fixed256 | hybrid | 0.150 | 0.251 | 0.079 | 0.158 | 0.099 | 0.242 | 0.29 | 20 |
| fixed256 | rerank | 0.193 | 0.291 | 0.102 | 0.164 | 0.119 | 0.241 | 0.30 | 323 |
| fixed512 | bm25 | 0.150 | 0.363 | 0.100 | 0.202 | 0.099 | 0.225 | 0.23 | 1 |
| fixed512 | dense | 0.276 | 0.456 | 0.149 | 0.267 | 0.160 | 0.307 | 0.36 | 15 |
| fixed512 | hybrid | 0.232 | 0.462 | 0.123 | 0.278 | 0.136 | 0.310 | 0.31 | 17 |
| fixed512 | rerank | 0.282 | 0.458 | 0.150 | 0.268 | 0.156 | 0.291 | 0.33 | 549 |
| structure | bm25 | 0.113 | 0.331 | 0.092 | 0.169 | 0.087 | 0.225 | 0.21 | 2 |
| structure | dense | 0.237 | 0.374 | 0.131 | 0.236 | 0.151 | 0.286 | 0.34 | 15 |
| structure | hybrid | 0.200 | 0.364 | 0.113 | 0.225 | 0.126 | 0.281 | 0.29 | 17 |
| structure | rerank | 0.223 | 0.370 | 0.119 | 0.207 | 0.139 | 0.261 | 0.32 | 503 |
| semantic | bm25 | 0.110 | 0.308 | 0.062 | 0.167 | 0.065 | 0.209 | 0.20 | 2 |
| semantic | dense | 0.273 | 0.431 | 0.153 | 0.271 | 0.170 | 0.305 | 0.34 | 16 |
| semantic | hybrid | 0.193 | 0.418 | 0.083 | 0.222 | 0.105 | 0.282 | 0.27 | 18 |
| semantic | rerank | 0.250 | 0.440 | 0.134 | 0.226 | 0.156 | 0.281 | 0.31 | 478 |

![recall@10 heatmap](results/heatmap.png)

| chunker | chunks | mean tokens | coverable evidence | chunking s | BM25 build s | embedding s |
|---|---|---|---|---|---|---|
| fixed256 | 37348 | 256 | 0.762 | 19 | 3.4 | 345.0 |
| fixed512 | 18688 | 511 | 0.979 | 19 | 3.3 | 365.1 |
| structure | 24692 | 339 | 0.921 | 20 | 4.5 | 543.5 |
| semantic | 32022 | 261 | 0.931 | 574 | 3.0 | 515.2 |

What stands out:

- Absolute recall is low, and the filing filter is the most reliable fix. The best setup reaches 0.462 recall@10. Filtering to the question's filing raised recall@10 in all 16 chunker and retriever pairs, by 0.083 to 0.230 (fixed512 dense goes from 0.276 to 0.456). Over the whole corpus only 0.20 to 0.36 of the top 10 comes from the right filing; the rest is similar text from other companies or other years of the same one.
- Dense beat BM25 in every cell, and equal-weight hybrid didn't help. BM25 only reaches 0.107 to 0.150 recall@10 over the corpus, and fusing its list pulled whole-corpus recall below dense alone for all four chunkers. The questions are long, carry instructions, and use different words than the statements ("capital expenditure" against "Purchases of property, plant and equipment").
- The cross-encoder didn't earn its latency. Reranking the hybrid top 50 recovered what fusion lost but ended near dense alone (fixed512: 0.282 vs 0.276 over the corpus, 0.458 vs 0.456 in the filing), had lower MRR@10 than dense in the filing for three of four chunkers, and costs 323 to 549 ms per query against 15 to 21 for dense.
- Bigger chunks did best, partly by construction. fixed512 had the best recall@10 for every retriever in both scopes, while fixed256 can only hold 0.762 of the evidence passages at all. Semantic chunks beat the structure-aware ones with dense retrieval (0.431 vs 0.374 in the filing), but building them took 574 s of paragraph embedding against 20 s.

## How relevance is defined

FinanceBench gives each question one to three evidence passages, each tagged with a filing and a page number. `evidence_page_num` is 0-based: all 189 passages line up best with that page index in the pypdfium2 text, none with the page before or after (`results/evidence.csv`).

Each passage is located in our page text by aligning its word tokens with difflib (match median 1.00, lowest 0.68). That gives the character positions of the passage in the filing. A chunk's coverage of a passage is the share of those positions that fall inside the chunk's character span. Grades:

- 2 if the chunk covers at least half of the passage
- 1 if it overlaps the evidence page but covers less
- 0 otherwise

Recall@k is the share of a question's passages with a grade-2 chunk in the top k, averaged over questions. MRR@10 uses the first grade-2 chunk. nDCG@10 uses the grades as gains, with the ideal ranking built from every graded chunk in the filing.

Every chunker cuts the same page text, so all of them are scored against one set of character offsets and none gets its own gold chunks. The catch is that a chunker only scores on passages it can hold: a 256-token window can't contain half of a full-page table. The coverable column is that ceiling.

## Faithfulness step (not run)

`answer.py` takes the top 5 chunks for a question and asks Claude (`claude-opus-5-5`, Anthropic SDK) to answer from them. Two judge calls with structured output follow. One splits the answer into claims and marks each as supported or not by those five chunks (faithfulness = supported / total). The other checks the answer against FinanceBench's reference answer. Results go to `results/answers_<chunker>_<retriever>_<scope>.jsonl`, one line per question.

No API key was available when these results were produced, so it has not been run and there are no faithfulness numbers here. To run it:

```
export ANTHROPIC_API_KEY=...
uv run answer.py --chunker structure --retriever rerank --scope doc --limit 20
```

That is three API calls per question; drop `--limit` for all 150. Requests opt into server-side refusal fallback (`fallbacks="default"`).

## Design notes

- Pages are joined into one string per filing and every chunk is a character span of it, so chunks and evidence share coordinates. Chunk sizes count bge-small WordPiece tokens, which keeps 512 inside the embedding model's limit.
- pypdfium2 gives no blank lines, so a paragraph ends at a line shorter than 75% of the page's longest line. The structure chunker merges paragraphs up to 400 tokens and starts a new chunk at an `Item N` or `Part N` line once the current one has 64 tokens. The semantic chunker uses the same paragraphs but breaks where the embedding distance to the previous paragraph is above that filing's 90th percentile.
- Dense is bge-small-en-v1.5 with its query instruction, normalized vectors and a numpy dot product over all chunks. BM25 is bm25s with stemming and stopwords. Hybrid is reciprocal rank fusion (k=60) of the two top-100 lists, and rerank runs ms-marco-MiniLM-L-6-v2 over the hybrid top 50.
- The filing scope works like a metadata filter: score everything, keep the filing's chunks. BM25 IDF still comes from the whole corpus.
- Questions go in verbatim, instructions included ("Give a response to the question by relying on the details shown in the cash flow statement").

## Limitations

- 150 questions is small, so gaps of a point or two are noise.
- PDF tables extract poorly. Statement rows come out as runs of labels and numbers with the column headers far away, and the statement chunks I checked ranked low for plain-language questions with both BM25 and the small embedding model.
- Only the cited page counts. The same figure often appears elsewhere in a filing (MD&A repeats the statements), so some misses would still answer the question.
- The cross-encoder reads at most 512 tokens per pair, so it sees a truncated fixed512 chunk.
- LLM-judge bias: the same model answers and judges, judges lean lenient, and how finely an answer is split into claims moves the faithfulness ratio. Spot-check the jsonl before trusting averages.
- Timings come from one M1 laptop that was running other jobs at the same time. Treat them as relative.
