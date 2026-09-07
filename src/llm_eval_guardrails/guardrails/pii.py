"""Low-latency PII and secret detection.

Detection is deterministic and regex-driven so that the inbound guardrail adds
microseconds, not milliseconds, to the request path. Patterns that regular
expressions alone cannot separate from ordinary numbers -- credit cards, IBANs,
US SSNs -- are additionally validated (Luhn, mod-97, allocation rules), which is
what keeps the false-positive rate low enough to redact automatically.

Named entities (people, places) genuinely need a model. When the optional
``pii`` extra is installed and ``use_presidio`` is enabled, Presidio's analyzer
augments the regex detectors; otherwise the regex layer runs alone and the
absence of NER is reported rather than silently assumed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

from ..logging_config import get_logger
from ..models import PIIEntity, PIIEntityType

__all__ = [
    "DEFAULT_PATTERNS",
    "PIIDetector",
    "PIIPattern",
    "luhn_valid",
    "redact",
    "valid_iban",
    "valid_ssn",
]

_log = get_logger(__name__)


def luhn_valid(digits: str) -> bool:
    """Validate a number against the Luhn checksum.

    Args:
        digits: The candidate number; non-digit separators are ignored.

    Returns:
        ``True`` when the digits satisfy the Luhn check and are of plausible
        payment-card length (13-19 digits).
    """
    cleaned = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(cleaned) <= 19:
        return False
    checksum = 0
    parity = len(cleaned) % 2
    for index, digit in enumerate(cleaned):
        value = digit
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def valid_ssn(candidate: str) -> bool:
    """Validate a US Social Security Number against SSA allocation rules.

    Rejects the ranges the SSA never issues, which eliminates most false
    positives from formatted numbers such as order or part identifiers.

    Args:
        candidate: The candidate SSN, with or without separators.

    Returns:
        ``True`` when the number could be a real SSN.
    """
    digits = "".join(c for c in candidate if c.isdigit())
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def valid_iban(candidate: str) -> bool:
    """Validate an IBAN using the ISO 13616 mod-97 checksum.

    Args:
        candidate: The candidate IBAN, with or without spaces.

    Returns:
        ``True`` when the checksum is valid.
    """
    cleaned = "".join(candidate.split()).upper()
    if not 15 <= len(cleaned) <= 34 or not cleaned[:2].isalpha():
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    try:
        numeric = "".join(
            str(int(ch, 36)) if ch.isalnum() else _raise_invalid(ch) for ch in rearranged
        )
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def _raise_invalid(char: str) -> str:
    """Reject a non-alphanumeric character during IBAN normalisation.

    Args:
        char: The offending character.

    Returns:
        Never returns.

    Raises:
        ValueError: Always.
    """
    msg = f"invalid IBAN character: {char!r}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PIIPattern:
    """A regex detector for one PII or secret category.

    Attributes:
        entity_type: The category this pattern detects.
        pattern: The compiled regular expression.
        confidence: Confidence assigned to matches, before validation.
        validator: Optional predicate that must accept the matched text.
        group: Capture group holding the entity, when the pattern needs
            surrounding context to anchor the match.
    """

    entity_type: PIIEntityType
    pattern: re.Pattern[str]
    confidence: float = 0.9
    validator: Any = None
    group: int = 0

    def finditer(self, text: str) -> Iterator[PIIEntity]:
        """Yield validated entities found in ``text``.

        Args:
            text: The text to scan.

        Yields:
            One :class:`~llm_eval_guardrails.models.PIIEntity` per accepted match.
        """
        for match in self.pattern.finditer(text):
            value = match.group(self.group)
            if not value:
                continue
            if self.validator is not None and not self.validator(value):
                continue
            yield PIIEntity(
                entity_type=self.entity_type,
                start=match.start(self.group),
                end=match.end(self.group),
                confidence=self.confidence,
                detector="regex",
            )


#: Ordered detectors. Higher-confidence, more specific patterns come first so
#: that overlap resolution keeps the most precise interpretation of a span.
DEFAULT_PATTERNS: Final[tuple[PIIPattern, ...]] = (
    PIIPattern(
        entity_type=PIIEntityType.PRIVATE_KEY,
        pattern=re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY(?: BLOCK)?-----"
            r".*?-----END (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY(?: BLOCK)?-----",
            re.DOTALL,
        ),
        confidence=1.0,
    ),
    PIIPattern(
        entity_type=PIIEntityType.AWS_ACCESS_KEY,
        pattern=re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
        confidence=1.0,
    ),
    PIIPattern(
        entity_type=PIIEntityType.AWS_SECRET_KEY,
        # Anchored on an explicit secret-key label: the value alone is
        # indistinguishable from any other 40-character base64 string.
        pattern=re.compile(
            r"(?i:aws[_\- ]?secret[_\- ]?(?:access[_\- ]?)?key)\s*[:=]\s*"
            r"[\"']?([A-Za-z0-9/+=]{40})[\"']?"
        ),
        confidence=0.95,
        group=1,
    ),
    PIIPattern(
        entity_type=PIIEntityType.JWT,
        pattern=re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        confidence=0.95,
    ),
    PIIPattern(
        entity_type=PIIEntityType.EMAIL,
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        confidence=0.95,
    ),
    PIIPattern(
        entity_type=PIIEntityType.CREDIT_CARD,
        pattern=re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        confidence=0.95,
        validator=luhn_valid,
    ),
    PIIPattern(
        entity_type=PIIEntityType.SSN,
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        confidence=0.95,
        validator=valid_ssn,
    ),
    PIIPattern(
        entity_type=PIIEntityType.IBAN,
        pattern=re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}[ ]?[A-Z0-9]{1,4}\b"),
        confidence=0.9,
        validator=valid_iban,
    ),
    PIIPattern(
        entity_type=PIIEntityType.PHONE,
        # E.164 and common North American / European formats. Requires a
        # separator or country prefix so bare 10-digit ids are not swept up.
        pattern=re.compile(
            r"(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\b\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}\b"
        ),
        confidence=0.75,
    ),
    PIIPattern(
        entity_type=PIIEntityType.IP_ADDRESS,
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
        ),
        confidence=0.7,
    ),
    PIIPattern(
        entity_type=PIIEntityType.PASSPORT,
        pattern=re.compile(r"(?i:passport(?:\s+(?:number|no\.?|#))?)\s*[:#]?\s*([A-Z0-9]{6,9})\b"),
        confidence=0.8,
        group=1,
    ),
    PIIPattern(
        entity_type=PIIEntityType.DATE_OF_BIRTH,
        pattern=re.compile(
            r"(?i:(?:date of birth|d\.?o\.?b\.?|born on))\s*[:\-]?\s*"
            r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})"
        ),
        confidence=0.85,
        group=1,
    ),
)


class PIIDetector:
    """Detects PII and secrets in text, optionally augmented by Presidio.

    The detector is stateless and safe to share across threads and tasks.

    Example:
        >>> detector = PIIDetector()
        >>> entities = detector.detect("contact me at ada@example.com")
        >>> entities[0].entity_type
        <PIIEntityType.EMAIL: 'email'>
    """

    def __init__(
        self,
        *,
        patterns: Sequence[PIIPattern] = DEFAULT_PATTERNS,
        allowed_types: Iterable[str] | None = None,
        use_presidio: bool = False,
        presidio_language: str = "en",
    ) -> None:
        """Initialise the detector.

        Args:
            patterns: Regex detectors to apply.
            allowed_types: Restrict detection to these entity-type values;
                ``None`` or empty enables every type. Unknown names are ignored
                with a warning rather than failing start-up.
            use_presidio: Augment regex detection with Presidio NER when the
                optional extra is installed.
            presidio_language: Language code passed to the Presidio analyzer.
        """
        selected = {t.strip().lower() for t in (allowed_types or []) if t.strip()}
        if selected:
            known = {t.value for t in PIIEntityType}
            unknown = selected - known
            if unknown:
                _log.warning("pii.unknown_entity_types", entity_types=sorted(unknown))
            self._patterns = tuple(p for p in patterns if p.entity_type.value in selected)
        else:
            self._patterns = tuple(patterns)

        self._allowed = selected
        self._use_presidio = use_presidio
        self._presidio_language = presidio_language

    @property
    def presidio_available(self) -> bool:
        """Whether Presidio NER is enabled and importable.

        Returns:
            ``True`` when Presidio will actually contribute detections.
        """
        return self._use_presidio and _load_presidio(self._presidio_language) is not None

    def detect(self, text: str) -> list[PIIEntity]:
        """Find all PII and secret spans in ``text``.

        Overlapping matches are resolved in favour of the longest span, then
        the highest confidence, so a credit-card match is never fragmented by
        an overlapping phone-number match.

        Args:
            text: The text to scan.

        Returns:
            Non-overlapping entities sorted by start offset.
        """
        if not text:
            return []

        found: list[PIIEntity] = []
        for pattern in self._patterns:
            found.extend(pattern.finditer(text))

        if self._use_presidio:
            found.extend(self._detect_with_presidio(text))

        return _resolve_overlaps(found)

    def _detect_with_presidio(self, text: str) -> list[PIIEntity]:
        """Run Presidio NER, degrading to no detections when unavailable.

        Args:
            text: The text to scan.

        Returns:
            Entities detected by Presidio, or an empty list when the optional
            dependency is missing or the analyzer fails.
        """
        analyzer = _load_presidio(self._presidio_language)
        if analyzer is None:
            return []
        try:
            raw_results = analyzer.analyze(text=text, language=self._presidio_language)
        except Exception:  # noqa: BLE001 - NER must never break the request path
            _log.warning("pii.presidio_analyze_failed", exc_info=True)
            return []

        entities: list[PIIEntity] = []
        for result in raw_results:
            entity_type = _PRESIDIO_TYPE_MAP.get(result.entity_type, PIIEntityType.OTHER)
            if self._allowed and entity_type.value not in self._allowed:
                continue
            try:
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        start=result.start,
                        end=result.end,
                        confidence=min(1.0, max(0.0, float(result.score))),
                        detector="presidio",
                    )
                )
            except ValueError:  # pragma: no cover - defensive against odd spans
                continue
        return entities


#: Maps Presidio's entity vocabulary onto this framework's categories.
_PRESIDIO_TYPE_MAP: Final[dict[str, PIIEntityType]] = {
    "EMAIL_ADDRESS": PIIEntityType.EMAIL,
    "PHONE_NUMBER": PIIEntityType.PHONE,
    "US_SSN": PIIEntityType.SSN,
    "CREDIT_CARD": PIIEntityType.CREDIT_CARD,
    "IBAN_CODE": PIIEntityType.IBAN,
    "IP_ADDRESS": PIIEntityType.IP_ADDRESS,
    "US_PASSPORT": PIIEntityType.PASSPORT,
    "DATE_TIME": PIIEntityType.DATE_OF_BIRTH,
    "PERSON": PIIEntityType.PERSON,
    "LOCATION": PIIEntityType.LOCATION,
    "NRP": PIIEntityType.OTHER,
}


@lru_cache(maxsize=4)
def _load_presidio(language: str) -> Any | None:
    """Load and cache a Presidio analyzer engine.

    Loading spaCy models is expensive, so the engine is cached per language for
    the process lifetime.

    Args:
        language: The analyzer language code.

    Returns:
        The analyzer, or ``None`` when the optional extra is not installed or
        its model could not be loaded.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        _log.info("pii.presidio_unavailable", reason="presidio-analyzer not installed")
        return None
    try:
        return AnalyzerEngine(supported_languages=[language])
    except Exception:  # noqa: BLE001 - missing spaCy model, bad config, etc.
        _log.warning("pii.presidio_init_failed", language=language, exc_info=True)
        return None


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Drop entities whose spans overlap a previously accepted, better match.

    Args:
        entities: Candidate entities, in arbitrary order.

    Returns:
        Non-overlapping entities sorted by start offset.
    """
    if not entities:
        return []

    # Prefer longer spans, then higher confidence, then earlier position.
    ranked = sorted(entities, key=lambda e: (-(e.end - e.start), -e.confidence, e.start))
    accepted: list[PIIEntity] = []
    for candidate in ranked:
        if any(candidate.start < kept.end and kept.start < candidate.end for kept in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda e: e.start)


def redact(
    text: str, entities: Sequence[PIIEntity], template: str = "[REDACTED:{entity_type}]"
) -> str:
    """Replace detected entity spans with a redaction token.

    Args:
        text: The original text.
        entities: Entities to redact. Overlapping spans are handled safely;
            they are replaced right-to-left so earlier offsets stay valid.
        template: Redaction template; ``{entity_type}`` is substituted.

    Returns:
        The redacted text. Returns the input unchanged when no entities are
        supplied.
    """
    if not entities:
        return text

    result = text
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        token = template.format(entity_type=entity.entity_type.value)
        result = result[: entity.start] + token + result[entity.end :]
    return result
