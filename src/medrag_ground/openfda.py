"""Thin client for the OpenFDA Drug Label API.

Public endpoint, no API key required (rate-limited to 240 req/min without one).
We pull a small slice of human-prescription labels for the demo and benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


@dataclass
class DrugLabel:
    """Subset of OpenFDA's drug label record that this project actually uses."""

    set_id: str
    brand_name: str
    generic_name: str
    indications_and_usage: str
    warnings: str
    dosage_and_administration: str
    raw: dict


class OpenFDAClient:
    """Fetches drug-label records from the public OpenFDA API."""

    def __init__(self, timeout: float = 30.0, base_url: str = OPENFDA_LABEL_URL) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def fetch_labels(self, limit: int = 30, search: str | None = None) -> list[DrugLabel]:
        """Return up to ``limit`` drug labels. ``search`` is an OpenFDA query.

        Default search restricts to records that include brand_name (skips
        weird records that have only generic info, which are messier to chunk).
        """
        params = {
            "search": search or "openfda.brand_name:_exists_",
            "limit": min(max(limit, 1), 100),
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        out: list[DrugLabel] = []
        for r in data.get("results", []):
            openfda = r.get("openfda") or {}
            brand = self._first(openfda.get("brand_name"))
            generic = self._first(openfda.get("generic_name"))
            out.append(
                DrugLabel(
                    set_id=r.get("set_id", "") or r.get("id", ""),
                    brand_name=brand,
                    generic_name=generic,
                    indications_and_usage=self._first(r.get("indications_and_usage")),
                    warnings=self._first(r.get("warnings") or r.get("warnings_and_cautions")),
                    dosage_and_administration=self._first(r.get("dosage_and_administration")),
                    raw=r,
                )
            )
        return out

    @staticmethod
    def _first(v: object) -> str:
        if isinstance(v, list) and v:
            return str(v[0])
        if isinstance(v, str):
            return v
        return ""
