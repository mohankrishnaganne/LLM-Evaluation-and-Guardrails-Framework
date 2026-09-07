"""Tests for PII and secret detection."""

from __future__ import annotations

import pytest

from llm_eval_guardrails.guardrails.pii import (
    PIIDetector,
    luhn_valid,
    redact,
    valid_iban,
    valid_ssn,
)
from llm_eval_guardrails.models import PIIEntity, PIIEntityType


class TestLuhn:
    @pytest.mark.parametrize(
        "number",
        [
            "4111111111111111",  # Visa test number
            "5500005555555559",  # Mastercard test number
            "4111 1111 1111 1111",
            "4111-1111-1111-1111",
            "378282246310005",  # Amex, 15 digits
        ],
    )
    def test_accepts_valid_cards(self, number):
        assert luhn_valid(number)

    @pytest.mark.parametrize(
        "number",
        [
            "4111111111111112",  # checksum off by one
            "1234567890123",
            "123456789012",  # too short
            "12345678901234567890",  # too long
            "",
            "not a number",
        ],
    )
    def test_rejects_invalid(self, number):
        assert not luhn_valid(number)


class TestSSN:
    @pytest.mark.parametrize("ssn", ["123-45-6789", "001-01-0001", "772019999"])
    def test_accepts_valid(self, ssn):
        assert valid_ssn(ssn)

    @pytest.mark.parametrize(
        "ssn",
        [
            "000-45-6789",  # area 000 is never issued
            "666-45-6789",  # area 666 is never issued
            "900-45-6789",  # 9xx is never issued
            "123-00-6789",  # group 00 is never issued
            "123-45-0000",  # serial 0000 is never issued
            "12345678",  # too few digits
            "1234567890",  # too many digits
        ],
    )
    def test_rejects_invalid(self, ssn):
        assert not valid_ssn(ssn)

    def test_validator_ignores_separator_placement(self):
        # The validator normalises separators; the *regex* enforces the
        # NNN-NN-NNNN shape, so these two responsibilities are tested apart.
        assert valid_ssn("12-345-6789")
        # The SSN detector requires the NNN-NN-NNNN shape, so this is not an
        # SSN hit (the broader phone heuristic may still match it).
        assert not any(
            e.entity_type is PIIEntityType.SSN for e in PIIDetector().detect("id 12-345-6789")
        )


class TestIBAN:
    @pytest.mark.parametrize(
        "iban",
        ["GB82WEST12345698765432", "DE89370400440532013000", "GB82 WEST 1234 5698 7654 32"],
    )
    def test_accepts_valid(self, iban):
        assert valid_iban(iban)

    @pytest.mark.parametrize(
        "iban",
        ["GB82WEST12345698765433", "12345678901234567", "GB8", "GB82WEST!2345698765432"],
    )
    def test_rejects_invalid(self, iban):
        assert not valid_iban(iban)


class TestPIIDetector:
    def test_detects_email(self):
        entities = PIIDetector().detect("Reach me at ada.lovelace@example.com please")
        assert [e.entity_type for e in entities] == [PIIEntityType.EMAIL]
        assert entities[0].detector == "regex"

    def test_detects_credit_card_and_validates_luhn(self):
        detector = PIIDetector()
        assert detector.detect("card 4111111111111111")
        # Same shape, invalid checksum: must not fire.
        assert not detector.detect("order 4111111111111112")

    def test_detects_ssn(self):
        entities = PIIDetector().detect("SSN 123-45-6789")
        assert entities[0].entity_type is PIIEntityType.SSN

    def test_rejects_ssn_shaped_non_ssn(self):
        assert not PIIDetector().detect("part number 000-45-6789")

    def test_detects_aws_access_key(self):
        entities = PIIDetector().detect("key AKIAIOSFODNN7EXAMPLE here")
        assert entities[0].entity_type is PIIEntityType.AWS_ACCESS_KEY

    def test_detects_aws_secret_key_only_when_labelled(self):
        detector = PIIDetector()
        labelled = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert any(e.entity_type is PIIEntityType.AWS_SECRET_KEY for e in detector.detect(labelled))
        # A bare 40-char token is indistinguishable from ordinary base64.
        bare = "wJalrXUtnFEMIaK7MDENGabPxRfiCYEXAMPLEKEY"
        assert not any(e.entity_type is PIIEntityType.AWS_SECRET_KEY for e in detector.detect(bare))

    def test_detects_jwt(self):
        token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        entities = PIIDetector().detect(f"token: {token}")
        assert entities[0].entity_type is PIIEntityType.JWT

    def test_detects_private_key_block(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nabc123\n-----END RSA PRIVATE KEY-----"
        entities = PIIDetector().detect(text)
        assert entities[0].entity_type is PIIEntityType.PRIVATE_KEY

    def test_detects_ip_address(self):
        entities = PIIDetector().detect("client 192.168.1.100 connected")
        assert entities[0].entity_type is PIIEntityType.IP_ADDRESS

    def test_rejects_out_of_range_ip(self):
        assert not any(
            e.entity_type is PIIEntityType.IP_ADDRESS
            for e in PIIDetector().detect("version 999.888.777.666")
        )

    def test_detects_date_of_birth_with_label(self):
        entities = PIIDetector().detect("Date of birth: 1985-04-12")
        assert any(e.entity_type is PIIEntityType.DATE_OF_BIRTH for e in entities)

    def test_returns_empty_for_clean_text(self):
        assert PIIDetector().detect("What is our refund policy for damaged goods?") == []

    def test_returns_empty_for_empty_text(self):
        assert PIIDetector().detect("") == []

    def test_detects_multiple_entities_sorted_by_offset(self):
        text = "Email ada@example.com or call +1 555 123 4567"
        entities = PIIDetector().detect(text)
        assert len(entities) >= 2
        assert entities == sorted(entities, key=lambda e: e.start)

    def test_spans_do_not_overlap(self):
        text = "card 4111111111111111 and ssn 123-45-6789 and ada@example.com"
        entities = PIIDetector().detect(text)
        for earlier, later in zip(entities, entities[1:], strict=False):
            assert earlier.end <= later.start

    def test_allowed_types_restricts_detection(self):
        detector = PIIDetector(allowed_types=["email"])
        entities = detector.detect("ada@example.com and 123-45-6789")
        assert {e.entity_type for e in entities} == {PIIEntityType.EMAIL}

    def test_unknown_allowed_type_is_ignored_not_fatal(self):
        detector = PIIDetector(allowed_types=["email", "not_a_real_type"])
        assert detector.detect("ada@example.com")

    def test_offsets_map_to_original_text(self):
        text = "contact ada@example.com now"
        entity = PIIDetector().detect(text)[0]
        assert text[entity.start : entity.end] == "ada@example.com"

    def test_presidio_unavailable_degrades_gracefully(self):
        # The `pii` extra is not installed in the default test environment;
        # detection must still work using regex alone.
        detector = PIIDetector(use_presidio=True)
        assert detector.detect("ada@example.com")


class TestRedact:
    def test_replaces_entity_with_token(self):
        text = "Email me at ada@example.com"
        entities = PIIDetector().detect(text)
        assert redact(text, entities) == "Email me at [REDACTED:email]"

    def test_redacts_multiple_entities_preserving_earlier_offsets(self):
        text = "ada@example.com and bob@example.org"
        result = redact(text, PIIDetector().detect(text))
        assert result == "[REDACTED:email] and [REDACTED:email]"

    def test_custom_template(self):
        text = "ada@example.com"
        result = redact(text, PIIDetector().detect(text), template="<{entity_type}>")
        assert result == "<email>"

    def test_no_entities_returns_input_unchanged(self):
        assert redact("nothing here", []) == "nothing here"

    def test_redacts_in_reverse_order_without_corrupting_text(self):
        text = "a ada@example.com b bob@example.org c"
        entities = [
            PIIEntity(entity_type=PIIEntityType.EMAIL, start=2, end=17),
            PIIEntity(entity_type=PIIEntityType.EMAIL, start=20, end=35),
        ]
        assert redact(text, entities) == "a [REDACTED:email] b [REDACTED:email] c"


class TestPIIEntityValidation:
    def test_rejects_inverted_span(self):
        with pytest.raises(ValueError, match="end .* must exceed start"):
            PIIEntity(entity_type=PIIEntityType.EMAIL, start=10, end=5)

    def test_rejects_empty_span(self):
        with pytest.raises(ValueError, match="end .* must exceed start"):
            PIIEntity(entity_type=PIIEntityType.EMAIL, start=5, end=5)
