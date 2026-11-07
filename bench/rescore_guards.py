"""Re-apply the pipeline's post-LLM guards to the answers logged by the last eval run.

The guards (citation match, drug-name check) are pure functions of the answer
text and the retrieved chunks, both of which the audit log keeps, so this
re-scores a committed run without calling an LLM again. It assumes the run
used the built-in fallback corpus, as the committed run did.

    python bench/rescore_guards.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent / "src"))
sys.path.insert(0, str(BENCH))

from eval_200 import FALLBACK_LABELS, OUT_OF_CORPUS_DRUGS  # noqa: E402

from medrag_ground import HashEmbedder, Indexer, MedRAG, MockLLMClient  # noqa: E402
from medrag_ground.indexer import Retrieval  # noqa: E402


def main() -> int:
    rows = json.loads((BENCH / "eval_results.json").read_text(encoding="utf-8"))["rows"]
    audit_lines = (BENCH / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in audit_lines[-len(rows):]]
    for row, entry in zip(rows, entries, strict=True):
        if not entry["pii_types"] and row["query"] != entry["redacted_query"]:
            raise SystemExit("audit.jsonl does not end with the run in eval_results.json")

    index = Indexer(embedder=HashEmbedder(dim=64))  # embeddings are unused here
    index.index_labels(FALLBACK_LABELS)
    by_id = {d.doc_id: d for d in index.docs}
    rag = MedRAG(index, MockLLMClient(), drug_lexicon=OUT_OF_CORPUS_DRUGS)

    outcome: Counter[tuple[str, str]] = Counter()
    for row, entry in zip(rows, entries, strict=True):
        if entry["refused"]:
            outcome[(row["category"], "refused in the run")] += 1
            continue
        retrieved = [Retrieval(doc=by_id[i], score=0.0) for i in entry["retrieved_doc_ids"]]
        reason, _ = rag.check_answer(entry["answer_text"], retrieved)
        outcome[(row["category"], reason or "passes the guards")] += 1
        if reason:
            print(f"{row['category']}: {row['query']!r} -> {reason}")

    for (category, result), n in sorted(outcome.items()):
        print(f"{category:13} {result:30} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
