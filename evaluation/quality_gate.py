"""
Quality Gate for Fast CPU Screening Pass.

Computes a composite quality score on page-level OCR output to decide whether
to accept the fast CPU result (PASS) or route the page to the heavy specialist (FAIL).

Evaluation Criteria:
  1. Mean OCR confidence threshold (>= 65.0%)
  2. Minimum word count for text-bearing pages
  3. Arabic sanity ratio (valid Arabic/Latin letters vs garbage noise)
  4. Repetition & gibberish anomaly check (e.g. 'ااااا')
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_CHAR_RE = re.compile(r"[\w\u0600-\u06FF\s.,:;!?()\-\"\']+")
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")  # 5+ consecutive identical characters


@dataclass
class QualityGateResult:
    passed: bool
    confidence: float
    word_count: int
    sanity_ratio: float
    reason: str


def evaluate_page_quality(
    text: str,
    confidence: float,
    min_confidence: float = 65.0,
    min_words: int = 15,
) -> QualityGateResult:
    """Evaluate whether a single page OCR result is high enough quality to accept."""
    clean_text = text.strip()
    words = clean_text.split()
    word_count = len(words)

    # If completely empty
    if not clean_text or word_count == 0:
        return QualityGateResult(
            passed=False,
            confidence=confidence,
            word_count=0,
            sanity_ratio=0.0,
            reason="Empty page or no text detected",
        )

    # 1. Word count check: if very few words and low confidence, likely noise
    if word_count < min_words and confidence < 75.0:
        return QualityGateResult(
            passed=False,
            confidence=confidence,
            word_count=word_count,
            sanity_ratio=0.0,
            reason=f"Insufficient word count ({word_count} < {min_words}) with low confidence ({confidence:.1f}%)",
        )

    # 2. Confidence threshold
    if confidence < min_confidence:
        return QualityGateResult(
            passed=False,
            confidence=confidence,
            word_count=word_count,
            sanity_ratio=0.0,
            reason=f"OCR confidence too low ({confidence:.1f}% < {min_confidence:.1f}%)",
        )

    # 3. Sanity ratio: valid characters vs total characters
    valid_chars = len("".join(_VALID_CHAR_RE.findall(clean_text)))
    total_chars = max(len(clean_text), 1)
    sanity_ratio = valid_chars / total_chars

    if sanity_ratio < 0.80:
        return QualityGateResult(
            passed=False,
            confidence=confidence,
            word_count=word_count,
            sanity_ratio=sanity_ratio,
            reason=f"High garbage character ratio (sanity {sanity_ratio*100:.1f}% < 80%)",
        )

    # 4. Repeated character anomaly check
    repeats = _REPEATED_CHAR_RE.findall(clean_text)
    if len(repeats) >= 3:
        return QualityGateResult(
            passed=False,
            confidence=confidence,
            word_count=word_count,
            sanity_ratio=sanity_ratio,
            reason="Repetition anomaly detected (gibberish/tiling artifact)",
        )

    return QualityGateResult(
        passed=True,
        confidence=confidence,
        word_count=word_count,
        sanity_ratio=sanity_ratio,
        reason="Page passed all quality gate checks",
    )
