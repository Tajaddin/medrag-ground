# medrag-ground

> Audit-logged RAG over OpenFDA drug labels with **mandatory citation grounding** and **PII detection** on inputs. Healthcare-shaped regulated-workflow proxy: every answer cites a source, every refusal is logged, and the pipeline detects PII before any retrieval call.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

200 questions across four adversarial categories, real Claude Haiku 4.5 calls (~$0.50 total), 5 fallback drug labels indexed (Tylenol, Advil, Bayer Aspirin, Aleve, Benadryl).

| Category | n | Required behavior | Result |
|---|---:|---|---:|
| **Grounded** (questions answerable from labels) | 50 | answer with valid citation | **90.0%** answered with citation (5 refusals — see analysis) |
| **Unanswerable** (questions about drugs NOT in corpus) | 50 | refuse | **100.0%** refused |
| **PII-injected** (queries containing SSN / phone / email / address / DOB) | 50 | refuse on PII before retrieval | **100.0%** refused for PII |
| **Adversarial** ("ignore previous instructions" jailbreaks) | 50 | refuse or stay strictly cited | **100.0%** safe |
| | | **Hallucinated drug names in any answer** | **0** |

Total wall-clock: 208 s. Raw per-question results: [`bench/eval_results.json`](bench/eval_results.json).

The 5 grounded refusals are cases where the citation guard caught a citation the LLM produced that didn't match a retrieved source — exactly the failure mode the guard is designed to catch (the LLM cited a section name like "general usage" that the retriever hadn't returned). The audit log records each as `unsupported citation(s)`.

Reproduce:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or GROQ_API_KEY for the free path
python bench/eval_200.py
```

## Why this exists

The CV claims healthcare data work, but the production system at AzeriMed is private. Recruiters scanning GitHub for healthcare evidence see nothing. `medrag-ground` is a public proxy that has the same shape as a regulated healthcare workflow:

* **Audit-logged** — every query (PII-redacted), every retrieval, every refusal lands in an append-only JSONL
* **Citation-grounded** — the pipeline rejects any answer without a bracketed source title from the retrieved chunks
* **PII-aware** — regex-based detection on user inputs catches email, phone, SSN, credit card, DOB, name+honorific, address; queries with PII are refused before retrieval
* **Adversarial-tolerant** — the strict system prompt anchors the model on cited sources; jailbreaks like "ignore previous instructions" don't escape the citation guard

OpenFDA is a public API, MIT-licensed code, no AWS, no API keys required for the default `HashEmbedder` path. Everything you need to run the eval ships in this repo.

## Architecture

```
                       ┌────────────────┐
   user query ────────▶│ PIIDetector    │ — regex pass for 7 PII types
                       └───────┬────────┘   refuse if any → audit log
                               │ ok
                       ┌───────▼────────┐
                       │ Retriever      │ — cosine top-k over indexed
                       │                │   OpenFDA labels
                       └───────┬────────┘   refuse if no hit ≥ min_score
                               │ ok
                       ┌───────▼────────┐
                       │ LLM (strict)   │ — cite-or-refuse system prompt
                       └───────┬────────┘
                               │
                       ┌───────▼────────┐
                       │ Citation guard │ — every bracketed title must
                       │                │   match a retrieved source
                       └───────┬────────┘   refuse if unsupported
                               │ ok
                       ┌───────▼────────┐
                       │ AuditLog       │ — append redacted query +
                       └────────────────┘   sources + refusal reason
```

Every step is mockable: pass a `MockLLMClient` and a `HashEmbedder` to run the pipeline offline.

## Quickstart

```bash
pip install -e ".[fastembed,groq,server]"
```

```python
from medrag_ground import (
    Indexer, MedRAG, OpenFDAClient, GroqClient, AuditLog,
)

# 1. Fetch and index drug labels
labels = OpenFDAClient().fetch_labels(limit=15)
indexer = Indexer()       # FastEmbedEmbedder by default; falls back to HashEmbedder
indexer.index_labels(labels)

# 2. Build the RAG pipeline
llm = GroqClient(model="llama-3.1-8b-instant")
rag = MedRAG(indexer, llm, audit_log=AuditLog("audit.jsonl"))

# 3. Ask. Get a cited answer or an audited refusal.
resp = rag.ask("What is the maximum daily dose of Tylenol for adults?")
print(resp.answer)
# "The maximum daily dose is 3,000 mg ... [Tylenol indications]"

print(resp.sources_cited)
# ["Tylenol indications"]

# PII in query → refused before any retrieval
resp = rag.ask("My SSN is 123-45-6789. What is Tylenol's dose?")
assert resp.refused and "ssn" in resp.pii_types

# Unknown drug → refused at the retrieval gate
resp = rag.ask("What is Lipitor used for?")
assert resp.refused and "no grounded source" in resp.refusal_reason
```

## What the pipeline refuses

| Refusal type | Trigger | Audit log shows |
|---|---|---|
| `PII detected` | Regex hit on email / phone / SSN / credit card / DOB / address / name+honorific | redacted query, `pii_types` |
| `no grounded source above threshold` | No retrieved chunk has cosine similarity ≥ `min_score` | retrieved doc IDs (empty) |
| `LLM refused: insufficient grounded source` | The LLM itself output `REFUSE: ...` | which sources it had access to |
| `answer had no citations` | LLM produced an answer but no `[source]` brackets | the offending answer text |
| `unsupported citation(s)` | LLM cited a source that wasn't in the retrieved chunks | the unsupported citation strings |

## FastAPI server

`examples/server.py` exposes `/ask`, `/audit`, and `/health`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn examples.server:app --port 8080
```

```bash
curl -X POST localhost:8080/ask -H 'content-type: application/json' \
    -d '{"query": "What is the warning for Advil?"}'

curl 'localhost:8080/audit?n=10'     # last 10 audit entries (PII-redacted)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
17 passed in 0.47s
```

Covers PII detection across 7 categories, redaction, the full pipeline's refusal paths (PII, no retrieval, LLM refusal, missing citation, unsupported citation), grounded-answer happy path, audit log shape, and citation extraction.

## Configuration

| Knob | Effect | Default |
|---|---|---|
| `min_score` (Retriever) | minimum cosine similarity for a chunk to enter context | 0.30 |
| `top_k` (MedRAG) | max chunks per query | 5 |
| `Embedder` | swap for any `Embedder` instance | FastEmbed BGE-small if installed, else HashEmbedder |
| `LLMClient` | swap to GroqClient (free), AnthropicClient, or any custom | required |

## Project layout

```
.
├── src/medrag_ground/
│   ├── __init__.py
│   ├── openfda.py        # OpenFDA Drug Label API client
│   ├── embedders.py      # FastEmbed (semantic) + HashEmbedder (zero-dep)
│   ├── indexer.py        # chunk + index drug labels; cosine retriever
│   ├── pii.py            # 7-pattern PII detector + redactor
│   ├── audit.py          # append-only JSONL audit log
│   ├── llm.py            # Mock / Groq / Anthropic adapters
│   └── pipeline.py       # MedRAG.ask() = PII → retrieve → cite-or-refuse → guard → audit
├── tests/                # 17 pytest cases
├── bench/
│   ├── eval_200.py       # 200-question 4-category benchmark
│   ├── eval_results.json # latest run output
│   └── audit.jsonl       # written by the eval
└── examples/
    └── server.py         # FastAPI /ask + /audit + /health
```

## Limitations

**OpenFDA labels are SPL excerpts, not clinical guidance.** This project is for software-engineering demonstration purposes only. Do not use the output as medical advice. The strict citation guard reduces hallucination risk but does not establish clinical fitness.

**PII detection is regex-based.** Production deployments handling PHI should use Microsoft Presidio, AWS Comprehend Medical, or similar — they handle context (e.g. "patient John" vs "John Wayne movie") and rare formats much better than a 7-pattern regex. The regex approach here is conservative (it refuses on false positives), which is the right failure mode for a healthcare workflow even if it's not the right fit for free-form chat.

**No de-identification beyond redaction.** Redaction replaces matches with `[REDACTED:TYPE]` — that's reversible information loss but not k-anonymity. For HIPAA-grade workflows you'd need a Safe Harbor or Expert Determination pipeline downstream.

**Single-tenant audit log.** The append-only JSONL has no auth or rotation. Real deployments would put audit entries in a separate database with retention policies.

**No re-ranking.** Top-k cosine is the only retrieval signal. Cross-encoder re-ranking would lift the grounded cite rate from 90% toward 95-100% on niche / multi-condition queries; out of scope for v0.1.

## License

MIT — see [LICENSE](LICENSE). Drug label content is sourced from the public OpenFDA API.
