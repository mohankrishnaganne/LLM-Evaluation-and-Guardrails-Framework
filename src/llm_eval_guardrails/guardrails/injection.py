"""Heuristic prompt-injection and jailbreak detection.

This is a weighted-signal detector, not a classifier. Each signal is a compiled
regex over a normalised copy of the prompt; matches contribute weights that are
combined into a score in [0, 1]. No model is loaded, so a scan costs
microseconds and adds no cold-start penalty to a container.

Two design decisions are worth calling out for review:

* **Normalisation before matching.** Injection payloads routinely use zero-width
  characters, full-width Unicode homoglyphs and runs of whitespace to defeat
  naive pattern matching. Text is normalised (NFKC, zero-width stripped,
  whitespace collapsed) before scanning, and offsets are mapped back to the
  original so evidence quotes stay meaningful.

* **Saturating combination, not a sum.** Scores combine as
  ``1 - prod(1 - w_i)``, so several weak signals raise suspicion without any
  single one dominating, and the result stays inside [0, 1] no matter how many
  patterns fire.

The detector is intentionally tuned to favour recall on well-known attack
phrasings and to stay quiet on ordinary questions. It is a first line of
defence: pair it with output-side grounding checks rather than treating it as
sufficient on its own.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

__all__ = [
    "INJECTION_SIGNALS",
    "InjectionScan",
    "InjectionSignal",
    "normalize_for_matching",
    "scan_for_injection",
]

#: Characters used to visually hide or break up injection payloads.
_ZERO_WIDTH: Final[str] = "\u200b‌‍⁠﻿­"

_ZERO_WIDTH_RE: Final[re.Pattern[str]] = re.compile("[" + _ZERO_WIDTH + "]")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class InjectionSignal:
    """One weighted injection heuristic.

    Attributes:
        signal_id: Stable identifier reported in findings and metrics.
        pattern: Compiled regex applied to the normalised prompt.
        weight: Contribution to the combined score, in (0, 1].
        description: Human-readable explanation of what the signal detects.
    """

    signal_id: str
    pattern: re.Pattern[str]
    weight: float
    description: str


@dataclass(frozen=True, slots=True)
class InjectionScan:
    """The outcome of scanning one prompt.

    Attributes:
        score: Combined suspicion score in [0, 1].
        signals: Identifiers of every signal that matched.
        evidence: Short quoted excerpts, one per matched signal.
    """

    score: float
    signals: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether no signal matched.

        Returns:
            ``True`` when the prompt raised no suspicion at all.
        """
        return not self.signals


def normalize_for_matching(text: str) -> str:
    """Normalise text so evasion tricks do not defeat pattern matching.

    Applies NFKC compatibility folding (collapsing full-width and other
    homoglyph forms), strips zero-width characters, collapses whitespace runs
    and lower-cases the result.

    Args:
        text: The raw prompt.

    Returns:
        The normalised text.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = _ZERO_WIDTH_RE.sub("", folded)
    folded = _WHITESPACE_RE.sub(" ", folded)
    return folded.lower()


def _signal(signal_id: str, regex: str, weight: float, description: str) -> InjectionSignal:
    """Compile one signal definition.

    Args:
        signal_id: Stable identifier.
        regex: The pattern source, matched case-insensitively.
        weight: Contribution to the combined score.
        description: What the signal detects.

    Returns:
        The compiled signal.
    """
    return InjectionSignal(
        signal_id=signal_id,
        pattern=re.compile(regex, re.IGNORECASE),
        weight=weight,
        description=description,
    )


#: Weighted heuristics, ordered roughly by decreasing severity.
INJECTION_SIGNALS: Final[tuple[InjectionSignal, ...]] = (
    _signal(
        "instruction_override",
        r"\b(?:ignore|disregard|forget|override|discard)\b[^.?!]{0,40}?"
        r"\b(?:previous|prior|above|earlier|all|any|the)\b[^.?!]{0,20}?"
        r"\b(?:instruction|prompt|rule|direction|guideline|command)s?\b",
        0.85,
        "Asks the model to discard its prior instructions.",
    ),
    _signal(
        "system_prompt_exfiltration",
        r"\b(?:reveal|show|print|repeat|output|display|disclose|tell me)\b[^.?!]{0,40}?"
        r"\b(?:system prompt|initial instruction|your instruction|the prompt above|"
        r"your rules|hidden prompt|developer message)\b",
        0.85,
        "Attempts to exfiltrate the system prompt or hidden instructions.",
    ),
    _signal(
        "role_reassignment",
        r"\b(?:you are now|from now on,? you|act as if you|pretend (?:to be|you are)|"
        r"roleplay as|simulate being)\b[^.?!]{0,60}?"
        r"\b(?:unrestricted|unfiltered|no (?:longer|restrictions)|without (?:any )?"
        r"(?:restriction|filter|rule|limit)|dan|jailbroken|developer mode|god mode)\b",
        0.8,
        "Reassigns the assistant to an unrestricted persona.",
    ),
    _signal(
        "guardrail_disable",
        r"\b(?:disable|turn off|bypass|circumvent|remove|switch off|ignore)\b[^.?!]{0,30}?"
        r"\b(?:safety|guardrail|filter|moderation|restriction|content polic|"
        r"safeguard|protection)\w*\b",
        0.85,
        "Asks the model to disable its safety controls.",
    ),
    _signal(
        "fake_system_turn",
        # Anchored on start-or-whitespace rather than a newline: normalisation
        # collapses newlines to spaces before matching, so a newline anchor
        # could only ever fire at offset 0 and would miss a forged turn that
        # appears later in the prompt.
        r"(?:^|\s)(?:\[?(?:system|assistant|developer)\]?\s*[:>]|"
        r"<\|?(?:im_start|system|endoftext)\|?>|#{2,}\s*(?:system|instruction))",
        0.75,
        "Forges a system or assistant turn inside user input.",
    ),
    _signal(
        "encoded_payload",
        r"\b(?:base64|rot13|hex|url)[- ]?(?:encoded?|decode)\b[^.?!]{0,40}?"
        r"\b(?:then|and)\b[^.?!]{0,20}?\b(?:execute|run|follow|obey|do)\b",
        0.7,
        "Hides instructions behind an encoding the model is told to decode and obey.",
    ),
    _signal(
        "developer_impersonation",
        r"\b(?:i am|this is|as)\b[^.?!]{0,20}?"
        r"\b(?:your (?:developer|creator|admin|administrator|owner)|"
        r"openai|anthropic|the system administrator)\b[^.?!]{0,40}?"
        r"\b(?:grant|authorize|permit|allow|override|instruct)\w*\b",
        0.7,
        "Claims privileged authority to authorise otherwise-blocked behaviour.",
    ),
    _signal(
        "exfiltration_channel",
        r"\b(?:send|post|upload|transmit|exfiltrate|forward)\b[^.?!]{0,40}?"
        r"\b(?:to )?(?:https?://|webhook|my server|this url|this endpoint)\b",
        0.65,
        "Asks the model to transmit data to an external endpoint.",
    ),
    _signal(
        "prompt_delimiter_injection",
        r"(?:-{3,}\s*end of (?:context|document|prompt)|"
        r"\bend of (?:context|instructions)\b\s*[.:]?\s*\bnew instructions?\b)",
        0.7,
        "Fakes a delimiter to make user input look like a new instruction block.",
    ),
    _signal(
        "hypothetical_bypass",
        r"\b(?:hypothetically|in a fictional (?:story|world|scenario)|"
        r"for (?:educational|research) purposes only|"
        r"this is (?:just )?a (?:test|simulation))\b[^.?!]{0,60}?"
        r"\b(?:how (?:to|do i)|steps to|instructions for)\b",
        0.4,
        "Uses a fictional or educational framing to solicit restricted content.",
    ),
    _signal(
        "output_format_hijack",
        r"\b(?:respond|answer|reply|output)\b[^.?!]{0,30}?"
        r"\bwith(?:out)?\b[^.?!]{0,30}?"
        r"\b(?:any (?:warning|caveat|disclaimer)|refusing|refusal|apolog)\w*\b",
        0.45,
        "Instructs the model to suppress refusals, warnings or disclaimers.",
    ),
)


def scan_for_injection(
    text: str, signals: tuple[InjectionSignal, ...] = INJECTION_SIGNALS
) -> InjectionScan:
    """Score a prompt for injection and jailbreak indicators.

    Args:
        text: The raw prompt.
        signals: Heuristics to apply; defaults to :data:`INJECTION_SIGNALS`.

    Returns:
        The scan result, with a saturating combined score in [0, 1].
    """
    if not text.strip():
        return InjectionScan(score=0.0, signals=(), evidence=())

    normalised = normalize_for_matching(text)
    matched_ids: list[str] = []
    evidence: list[str] = []
    residual = 1.0

    for signal in signals:
        match = signal.pattern.search(normalised)
        if match is None:
            continue
        matched_ids.append(signal.signal_id)
        evidence.append(_excerpt(normalised, match.start(), match.end()))
        residual *= 1.0 - signal.weight

    score = 1.0 - residual
    return InjectionScan(
        score=min(1.0, max(0.0, score)),
        signals=tuple(matched_ids),
        evidence=tuple(evidence),
    )


def _excerpt(text: str, start: int, end: int, *, window: int = 24, limit: int = 160) -> str:
    """Quote a bounded excerpt around a matched span.

    Args:
        text: The normalised text that was scanned.
        start: Inclusive start offset of the match.
        end: Exclusive end offset of the match.
        window: Extra characters of context on each side.
        limit: Maximum length of the returned excerpt.

    Returns:
        The excerpt, with ellipses marking elided content.
    """
    left = max(0, start - window)
    right = min(len(text), end + window)
    body = text[left:right].strip()
    if len(body) > limit:
        body = body[:limit] + "..."
    return ("..." if left > 0 else "") + body + ("..." if right < len(text) else "")
