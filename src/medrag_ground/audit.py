"""Append-only audit log. Every query the pipeline sees lands here."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AuditEntry:
    timestamp: float
    raw_query: str
    redacted_query: str
    refused: bool
    refusal_reason: str | None
    retrieved_doc_ids: list[str]
    sources_cited: list[str]
    answer_text: str
    pii_types: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class AuditLog:
    """Append-only JSONL log. Each ``write`` flushes immediately so an
    operator inspecting the file mid-run sees up-to-date state.

    Stored on disk by default; if ``path`` is ``None``, the log is in-memory
    only (useful for tests).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.entries: list[AuditEntry] = []

    def write(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    @staticmethod
    def now() -> float:
        return time.time()

    def tail(self, n: int = 50) -> list[AuditEntry]:
        return self.entries[-n:]
