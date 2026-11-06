# medrag-ground

> Audit-logged RAG over OpenFDA drug labels with **mandatory citation grounding** and **PII detection** on inputs. Healthcare-shaped regulated-workflow proxy: every answer cites a source, every refusal is logged, and the pipeline detects PII before any retrieval call.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

200 questions across four adversarial categories, real Claude Haiku 4.5 calls (~$0.50 total), 5 fallback drug labels indexed (Tylenol, Advil, Bayer Aspirin, Aleve, Benadryl).

| Category | n | Required behavior | Result |
|---|---:|---|---:|
| **Grounded** (questions answerable from labels) | 50 | answer with valid citation | **90.0%** answered with citation (5 refusals — see analysis) |
| **Unanswerable** (questions about drugs NOT in corpus) | 50 | refuse | **100.0%** refused |
| **PII-injected** (queries containing SSN / phone / email / address / DOB) | 50 | refuse on PII before retrieval | **100.0%** refused for PII |
| **Adversarial** ("ignore previous instructions" jailbreaks) | 50 | refuse or stay strictly cited | **100.0%** safe |
| | | **Delivered answers naming a drug their retrieved sources don't mention** | **0 of 77** |

Total wall-clock: 208 s. Raw per-question results: [`bench/eval_results.json`](bench/eval_results.json).

The 5 grounded refusals are the LLM's own `REFUSE` (the audit log records each as `LLM refused: insufficient grounded source`); none was caught by the citation guard. In three, about what Aleve treats, the retriever never returned Aleve's indications chunk. The other two ask about warnings the labels don't contain (stomach bleeding for Benadryl, drowsiness for Tylenol), so refusing was correct.

The drug-name row was re-checked offline after the guards were tightened: `python bench/rescore_guards.py` re-applies the exact citation match and the drug-name guard to the 77 answers the run delivered, using the audit log's answer text and retrieved chunks. All 77 pass, so the rates above are unchanged. The guard knows 20 names: the 5 indexed brands, their generic names, and the 10 out-of-corpus drugs the unanswerable questions ask about; it cannot catch a drug name outside that list. The original run's check only looked for capitalised mentions of those 10 out-of-corpus drugs.

Reproduce:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or GROQ_API_KEY for the free path
python bench/eval_200.py
```

## Why this exists

The CV claims healthcare data work, but the production system at AzeriMed is private. Recruiters scanning GitHub for healthcare evidence see nothing. `medrag-ground` is a public proxy that has the same shape as a regulated healthcare workflow:

* **Audit-logged** — every query (PII-redacted), every retrieval, every refusal lands in an append-only JSONL
* **Citation-grounded** — the pipeline rejects any answer without a bracketed source title from the retrieved chunks, and any answer that names a known drug none of those chunks mention
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
                       │                │   equal a retrieved source title
                       └───────┬────────┘   refuse if unsupported
                               │ ok
                       ┌───────▼────────┐
                       │ Drug-name guard│ — every known drug named must
                       │                │   appear in a retrieved source
                       └───────┬────────┘   refuse if ungrounded
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
| `unsupported citation(s)` | LLM cited a source that wasn't in the retrieved chunks (titles must match exactly, ignoring case and spacing) | the unsupported citation strings |
| `ungrounded drug name(s)` | Answer names a known drug — any indexed brand or generic name, plus `drug_lexicon` — that no retrieved chunk mentions | the drug names |

## FastAPI server

`examples/server.py` exposes `/ask`, `/audit`, and `/health`. `/audit` needs a bearer token; without `MEDRAG_AUDIT_TOKEN` it is disabled.

```bash
export GROQ_API_KEY=gsk_...
export MEDRAG_AUDIT_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
uvicorn examples.server:build_app --factory --port 8080
```

```bash
curl -X POST localhost:8080/ask -H 'content-type: application/json' \
    -d '{"query": "What is the warning for Advil?"}'

curl -H "Authorization: Bearer $MEDRAG_AUDIT_TOKEN" 'localhost:8080/audit?n=10'  # last 10 entries (PII-redacted), n in 1..100
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
32 passed
```

Covers PII detection across 7 categories, redaction, the full pipeline's refusal paths (PII, no retrieval, LLM refusal, missing citation, unsupported or partial citation, ungrounded drug name), grounded-answer happy path, audit log shape (no raw query on disk), citation extraction, and the server's `/audit` token and bounds (needs `fastapi`, in the dev extra).

## Configuration

| Knob | Effect | Default |
|---|---|---|
| `min_score` (Retriever) | minimum cosine similarity for a chunk to enter context | 0.30 |
| `top_k` (MedRAG) | max chunks per query | 5 |
| `drug_lexicon` (MedRAG) | extra drug names the drug-name guard recognises, e.g. drugs known to be absent from the corpus; it can only catch names it knows | none |
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
├── tests/                # 32 pytest cases
├── bench/
│   ├── eval_200.py       # 200-question 4-category benchmark
│   ├── eval_results.json # latest run output
│   ├── rescore_guards.py # re-apply the post-LLM guards to the logged run
│   └── audit.jsonl       # written by the eval
└── examples/
    └── server.py         # FastAPI /ask + /audit + /health
```

## Limitations

**OpenFDA labels are SPL excerpts, not clinical guidance.** This project is for software-engineering demonstration purposes only. Do not use the output as medical advice. The strict citation guard reduces hallucination risk but does not establish clinical fitness.

**PII detection is regex-based.** Production deployments handling PHI should use Microsoft Presidio, AWS Comprehend Medical, or similar — they handle context (e.g. "patient John" vs "John Wayne movie") and rare formats much better than a 7-pattern regex. The regex approach here is conservative (it refuses on false positives), which is the right failure mode for a healthcare workflow even if it's not the right fit for free-form chat.

**No de-identification beyond redaction.** Redaction replaces matches with `[REDACTED:TYPE]` — that's reversible information loss but not k-anonymity. For HIPAA-grade workflows you'd need a Safe Harbor or Expert Determination pipeline downstream.

**Single-tenant audit log.** The append-only JSONL has no rotation, and the example server guards `/audit` with one shared token. Real deployments would put audit entries in a separate database with retention policies.

**No re-ranking.** Top-k cosine is the only retrieval signal. Cross-encoder re-ranking would lift the grounded cite rate from 90% toward 95-100% on niche / multi-condition queries; out of scope for v0.1.

## License

MIT — see [LICENSE](LICENSE). Drug label content is sourced from the public OpenFDA API.
