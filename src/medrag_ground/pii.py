"""Regex-based PII detector. Conservative — false positives are preferred to false negatives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4})",
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_DOB_RE = re.compile(r"\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](19|20)\d{2}\b")
# Name heuristic: "Mr/Mrs/Dr <First> <Last>" or "<First> <Last> visited..."
_NAME_PREFIXED_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Miss|Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(St|Street|Ave|Avenue|Blvd|Road|Rd|Dr|Drive|Ln|Lane|Way|Court|Ct)\b"
)


@dataclass
class PIIDetection:
    found: bool
    types: list[str] = field(default_factory=list)
    matches: list[tuple[str, str]] = field(default_factory=list)  # (pii_type, match_text)

    @property
    def types_str(self) -> str:
        return ", ".join(sorted(set(self.types)))


class PIIDetector:
    """Detect common PII patterns in free-form text.

    Production deployments should swap this for Microsoft Presidio or a
    domain-specific detector. This regex-only version covers the common
    cases that appear in user-typed healthcare queries (emails, phones,
    addresses, SSNs, credit cards, dates of birth, name+honorific).
    """

    PATTERNS: list[tuple[str, re.Pattern]] = [
        ("email", _EMAIL_RE),
        ("phone", _PHONE_RE),
        ("ssn", _SSN_RE),
        ("credit_card", _CC_RE),
        ("date_of_birth", _DOB_RE),
        ("name_with_honorific", _NAME_PREFIXED_RE),
        ("address", _ADDRESS_RE),
    ]

    def detect(self, text: str) -> PIIDetection:
        if not text:
            return PIIDetection(found=False)
        types: list[str] = []
        matches: list[tuple[str, str]] = []
        for kind, pat in self.PATTERNS:
            for m in pat.finditer(text):
                types.append(kind)
                matches.append((kind, m.group(0)))
        return PIIDetection(found=bool(types), types=types, matches=matches)

    def redact(self, text: str) -> str:
        """Return ``text`` with detected PII replaced by ``[REDACTED:TYPE]``."""
        if not text:
            return text
        out = text
        for kind, pat in self.PATTERNS:
            out = pat.sub(f"[REDACTED:{kind.upper()}]", out)
        return out
