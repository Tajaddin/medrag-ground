"""PII detector tests."""

from __future__ import annotations

import pytest

from medrag_ground import PIIDetector


@pytest.fixture
def det() -> PIIDetector:
    return PIIDetector()


def test_email_detected(det: PIIDetector) -> None:
    r = det.detect("Email me at jane.doe@example.com please")
    assert r.found is True
    assert "email" in r.types


def test_phone_us_detected(det: PIIDetector) -> None:
    r = det.detect("Call me at (415) 555-1212.")
    assert r.found is True
    assert "phone" in r.types


def test_ssn_detected(det: PIIDetector) -> None:
    r = det.detect("My SSN is 123-45-6789.")
    assert r.found is True
    assert "ssn" in r.types


def test_credit_card_detected(det: PIIDetector) -> None:
    r = det.detect("Card 4111-1111-1111-1111 expires soon.")
    assert r.found is True
    assert "credit_card" in r.types


def test_address_detected(det: PIIDetector) -> None:
    r = det.detect("Send to 1600 Pennsylvania Avenue.")
    assert r.found is True
    assert "address" in r.types


def test_name_with_honorific_detected(det: PIIDetector) -> None:
    r = det.detect("Dr. John Smith prescribed it.")
    assert r.found is True
    assert "name_with_honorific" in r.types


def test_clean_text_passes(det: PIIDetector) -> None:
    r = det.detect("What is the recommended dose of acetaminophen for adults?")
    assert r.found is False
    assert r.types == []


def test_redact_replaces_match(det: PIIDetector) -> None:
    out = det.redact("Email me at test@example.com")
    assert "test@example.com" not in out
    assert "REDACTED" in out


def test_empty_input(det: PIIDetector) -> None:
    assert det.detect("").found is False
    assert det.redact("") == ""
