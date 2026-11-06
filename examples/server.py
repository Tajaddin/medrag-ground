"""FastAPI server exposing /ask, /audit and /health endpoints.

Run with::

    pip install -e ".[fastembed,groq,server]"
    export GROQ_API_KEY=gsk_...
    export MEDRAG_AUDIT_TOKEN=...   # enables /audit; unset leaves it disabled
    uvicorn examples.server:build_app --factory --port 8080
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from medrag_ground import (
    AuditLog,
    GroqClient,
    Indexer,
    MedRAG,
    OpenFDAClient,
    get_embedder,
)

MAX_AUDIT_ENTRIES = 100


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    sources_cited: list[str]


def create_app(rag: MedRAG, audit_log: AuditLog, *, audit_token: str | None) -> FastAPI:
    """``/audit`` requires ``Authorization: Bearer <audit_token>``; with no token it is disabled."""
    app = FastAPI(title="medrag-ground")

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        resp = rag.ask(req.query)
        return AskResponse(
            answer=resp.answer,
            refused=resp.refused,
            refusal_reason=resp.refusal_reason,
            sources_cited=resp.sources_cited,
        )

    @app.get("/audit")
    def audit(
        n: int = Query(20, ge=1, le=MAX_AUDIT_ENTRIES),
        authorization: str | None = Header(None),
    ) -> dict:
        """Return the last N audit entries (most recent first)."""
        if not audit_token or not hmac.compare_digest(
            (authorization or "").encode(), f"Bearer {audit_token}".encode()
        ):
            raise HTTPException(status_code=401, detail="audit token required")
        entries = audit_log.tail(n)
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
        return {"status": "ok", "n_chunks": len(rag.indexer)}

    return app


def build_app() -> FastAPI:
    """Index the labels, connect to Groq and wire the endpoints (``uvicorn --factory``)."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY before starting the server.")

    # Build the index on startup. In production this would be a persistent store.
    indexer = Indexer(embedder=get_embedder())
    try:
        indexer.index_labels(OpenFDAClient().fetch_labels(limit=15))
    except Exception:
        # Network unavailable — bench/eval_200.py keeps a small fallback corpus.
        from bench.eval_200 import FALLBACK_LABELS  # type: ignore[import-not-found]

        indexer.index_labels(FALLBACK_LABELS)

    audit_log = AuditLog(Path("audit.jsonl"))
    rag = MedRAG(indexer, GroqClient(), audit_log=audit_log)
    return create_app(rag, audit_log, audit_token=os.environ.get("MEDRAG_AUDIT_TOKEN"))
