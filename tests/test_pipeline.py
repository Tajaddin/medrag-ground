"""Integration tests for MedRAG with mock LLM and HashEmbedder."""

from __future__ import annotations

import pytest

from medrag_ground import (
    AuditLog,
    HashEmbedder,
    Indexer,
    MedRAG,
    MockLLMClient,
    PIIDetector,
)
from medrag_ground.openfda import DrugLabel


def _build_index() -> Indexer:
    labels = [
        DrugLabel(
            set_id="abc1",
            brand_name="Tylenol",
            generic_name="Acetaminophen",
            indications_and_usage="For temporary relief of minor aches and pain. Reduces fever.",
            warnings="Do not take with alcohol. May cause liver damage.",
            dosage_and_administration="Adults: 325-650 mg every 4 to 6 hours.",
            raw={},
        ),
        DrugLabel(
            set_id="def2",
            brand_name="Advil",
            generic_name="Ibuprofen",
            indications_and_usage="For pain relief from arthritis and headache.",
            warnings="May cause stomach bleeding. Avoid in pregnancy.",
            dosage_and_administration="Adults: 200-400 mg every 4 to 6 hours.",
            raw={},
        ),
    ]
    idx = Indexer(embedder=HashEmbedder(dim=256))
    idx.index_labels(labels)
    return idx


def test_pii_query_is_refused() -> None:
    idx = _build_index()
    rag = MedRAG(idx, MockLLMClient(handler=lambda p, s: "should not be called"))
    resp = rag.ask("My SSN is 123-45-6789, what is Tylenol for?")
    assert resp.refused is True
    assert "PII" in (resp.refusal_reason or "")
    assert "ssn" in resp.pii_types


def test_unanswerable_query_refused_no_retrieval() -> None:
    idx = _build_index()
    rag = MedRAG(idx, MockLLMClient(handler=lambda p, s: "ignored"), min_score=0.999)
    resp = rag.ask("What is the capital of France?")
    assert resp.refused is True
    assert "grounded source" in (resp.refusal_reason or "")
    assert resp.sources_cited == []


def test_llm_refusal_is_treated_as_refusal() -> None:
    idx = _build_index()
    handler = lambda p, s: "REFUSE: insufficient grounded source"
    rag = MedRAG(idx, MockLLMClient(handler=handler), min_score=0.0)
    resp = rag.ask("Anything about Tylenol?")
    assert resp.refused is True
    assert "insufficient" in (resp.refusal_reason or "")


def test_answer_without_citations_refused() -> None:
    idx = _build_index()
    handler = lambda p, s: "Tylenol is used for fever and pain. It is generally safe."
    rag = MedRAG(idx, MockLLMClient(handler=handler), min_score=0.0)
    resp = rag.ask("What is Tylenol for?")
    assert resp.refused is True
    assert "no citations" in (resp.refusal_reason or "")


def test_answer_with_unsupported_citation_refused() -> None:
    idx = _build_index()
    # Lipitor is NOT in the index; the citation guard should catch this.
    handler = lambda p, s: "Lipitor lowers cholesterol [Lipitor indications]."
    rag = MedRAG(idx, MockLLMClient(handler=handler), min_score=0.0)
    resp = rag.ask("What is Lipitor for?")
    assert resp.refused is True
    assert "unsupported" in (resp.refusal_reason or "").lower()


def test_grounded_answer_with_valid_citation_passes() -> None:
    idx = _build_index()
    handler = lambda p, s: "Acetaminophen is used for temporary relief [Tylenol indications]."
    rag = MedRAG(idx, MockLLMClient(handler=handler), min_score=0.0)
    resp = rag.ask("What is Tylenol used for?")
    assert resp.refused is False
    assert "Tylenol indications" in resp.sources_cited


def test_audit_log_records_every_call() -> None:
    idx = _build_index()
    log = AuditLog()
    handler = lambda p, s: "REFUSE"
    rag = MedRAG(idx, MockLLMClient(handler=handler), audit_log=log)

    rag.ask("My phone is 415-555-1212")  # PII path
    rag.ask("What is Lipitor for?")      # retrieval path
    assert len(log.entries) == 2
    assert log.entries[0].refused is True
    assert "ssn" not in log.entries[0].pii_types  # was phone, not ssn
    assert "phone" in log.entries[0].pii_types


def test_extract_citations_parses_brackets() -> None:
    from medrag_ground.pipeline import _extract_citations

    out = _extract_citations("foo [Tylenol indications] bar [Advil warnings] baz [Tylenol indications]")
    assert out == ["Tylenol indications", "Advil warnings"]
