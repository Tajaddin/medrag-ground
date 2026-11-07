"""/audit endpoint of the example server: token-gated and bounded."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from examples.server import create_app  # noqa: E402
from medrag_ground import AuditLog, HashEmbedder, Indexer, MedRAG, MockLLMClient  # noqa: E402
from medrag_ground.openfda import DrugLabel  # noqa: E402

TOKEN = "audit-token-for-tests"


def _client(audit_token: str | None) -> TestClient:
    idx = Indexer(embedder=HashEmbedder(dim=64))
    idx.index_labels(
        [DrugLabel(set_id="t", brand_name="Tylenol", generic_name="Acetaminophen",
                   indications_and_usage="Relieves minor aches.", warnings="", dosage_and_administration="", raw={})]
    )
    log = AuditLog()
    rag = MedRAG(idx, MockLLMClient(handler=lambda p, s: "unused"), audit_log=log)
    for i in range(30):
        rag.ask(f"My phone is 415-555-{1000 + i}")
    return TestClient(create_app(rag, audit_token=audit_token))


def test_audit_requires_the_configured_token() -> None:
    client = _client(TOKEN)
    assert client.get("/audit").status_code == 401
    assert client.get("/audit", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/audit?n=5", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200
    assert ok.json()["n"] == 5


def test_audit_token_scheme_is_case_insensitive() -> None:
    client = _client(TOKEN)
    assert client.get("/audit", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200


def test_audit_401_names_the_bearer_scheme() -> None:
    r = _client(TOKEN).get("/audit")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_audit_is_disabled_when_no_token_is_configured() -> None:
    client = _client(None)
    for header in ("Bearer ", "Bearer None"):
        assert client.get("/audit", headers={"Authorization": header}).status_code == 401


@pytest.mark.parametrize("n", [0, -5, 101])
def test_audit_rejects_out_of_range_n(n: int) -> None:
    client = _client(TOKEN)
    assert client.get(f"/audit?n={n}", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 422
