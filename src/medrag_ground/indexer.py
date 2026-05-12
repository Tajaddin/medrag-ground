"""Chunk drug-label text into searchable documents and run cosine retrieval over them.

Documents are kept in memory — this project is a regulated-workflow proxy, not
a search engine. A real deployment would persist to pgvector or similar; the
:class:`Indexer` API is small enough that swapping the storage layer is one method.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from medrag_ground.embedders import Embedder, get_embedder
from medrag_ground.openfda import DrugLabel


@dataclass
class DrugDoc:
    """One chunk of a drug-label document with provenance."""

    doc_id: str
    drug_name: str
    section: str           # "indications" | "warnings" | "dosage"
    text: str
    source_url: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Retrieval:
    doc: DrugDoc
    score: float


def _chunk(text: str, max_chars: int = 700) -> list[str]:
    """Split on sentence boundaries up to ``max_chars`` per chunk. Cheap, robust."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip() if buf else s
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out


class Indexer:
    """Builds a searchable corpus of drug-label chunks."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or get_embedder()
        self.docs: list[DrugDoc] = []
        self._embeddings: np.ndarray | None = None  # shape (n_docs, dim)

    def index_labels(self, labels: list[DrugLabel]) -> None:
        new_docs: list[DrugDoc] = []
        for label in labels:
            drug_name = label.brand_name or label.generic_name or label.set_id
            source_url = (
                f"https://api.fda.gov/drug/label.json?search=set_id:{label.set_id}"
                if label.set_id
                else ""
            )
            for section_name, section_text in (
                ("indications", label.indications_and_usage),
                ("warnings", label.warnings),
                ("dosage", label.dosage_and_administration),
            ):
                if not section_text:
                    continue
                for chunk in _chunk(section_text):
                    new_docs.append(
                        DrugDoc(
                            doc_id=f"{label.set_id}::{section_name}::{len(new_docs)}",
                            drug_name=drug_name,
                            section=section_name,
                            text=chunk,
                            source_url=source_url,
                            metadata={"generic_name": label.generic_name, "brand_name": label.brand_name},
                        )
                    )
        if not new_docs:
            return
        new_embs = self.embedder.embed([d.text for d in new_docs])
        self.docs.extend(new_docs)
        if self._embeddings is None:
            self._embeddings = new_embs
        else:
            self._embeddings = np.vstack([self._embeddings, new_embs])

    def __len__(self) -> int:
        return len(self.docs)


class Retriever:
    """Cosine-similarity top-k retrieval over an :class:`Indexer`."""

    def __init__(self, indexer: Indexer, min_score: float = 0.30) -> None:
        if indexer._embeddings is None or len(indexer.docs) == 0:
            raise ValueError("Indexer is empty. Call index_labels() first.")
        self.indexer = indexer
        self.min_score = min_score

    def search(self, query: str, k: int = 5) -> list[Retrieval]:
        q_emb = self.indexer.embedder.embed([query])[0]
        scores = (self.indexer._embeddings @ q_emb).astype(float)
        order = np.argsort(-scores)
        out: list[Retrieval] = []
        for idx in order[:k]:
            s = float(scores[idx])
            if s < self.min_score:
                break
            out.append(Retrieval(doc=self.indexer.docs[idx], score=s))
        return out

    def drug_names(self) -> set[str]:
        """Distinct drug names currently in the index. Used by the citation guard
        to detect hallucinated drug names that aren't in the corpus."""
        return {d.drug_name.lower() for d in self.indexer.docs if d.drug_name}
