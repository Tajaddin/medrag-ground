"""FastAPI server exposing /ask and /audit endpoints.

Run with::

    pip install -e ".[fastembed,groq,server]"
    export GROQ_API_KEY=gsk_...
    uvicorn examples.server:app --port 8080
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from medrag_ground import (
    AuditLog,
    GroqClient,
    Indexer,
    MedRAG,
    OpenFDAClient,
    get_embedder,
)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    sources_cited: list[str]


app = FastAPI(title="medrag-ground")

# Build the index on startup. In production this would be a persistent store.
_indexer = Indexer(embedder=get_embedder())
try:
    _indexer.index_labels(OpenFDAClient().fetch_labels(limit=15))
except Exception:
    # Network unavailable — bench/eval_200.py keeps a small fallback corpus.
    from bench.eval_200 import FALLBACK_LABELS  # type: ignore[import-not-found]

    _indexer.index_labels(FALLBACK_LABELS)

_llm = GroqClient() if os.environ.get("GROQ_API_KEY") else None
_audit = AuditLog(Path("audit.jsonl"))

if _llm is None:
    raise RuntimeError("Set GROQ_API_KEY before starting the server.")

_rag = MedRAG(_indexer, _llm, audit_log=_audit)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    resp = _rag.ask(req.query)
    return AskResponse(
        answer=resp.answer,
        refused=resp.refused,
        refusal_reason=resp.refusal_reason,
        sources_cited=resp.sources_cited,
    )


@app.get("/audit")
def audit(n: int = 20) -> dict:
    """Return the last N audit entries (most recent first)."""
    entries = _audit.tail(n)
    return {
        "n": len(entries),
        "entries": [
            {
                "timestamp": e.timestamp,
                "redacted_query": e.redacted_query,
                "refused": e.refused,
                "refusal_reason": e.refusal_reason,
                "sources_cited": e.sources_cited,
                "answer_preview": (e.answer_text or "")[:120],
                "pii_types": e.pii_types,
            }
            for e in reversed(entries)
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_chunks": len(_indexer)}
