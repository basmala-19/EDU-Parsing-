"""
Unit tests for evaluation/quality_gate.py.

Covers all four failure conditions (empty, low word count + low confidence,
low confidence, low sanity ratio, repetition anomaly) plus the success path.
"""
import pytest

from evaluation.quality_gate import evaluate_page_quality, QualityGateResult


# ─── helpers ────────────────────────────────────────────────────────────────

def _arabic_sentence(n_words: int = 20) -> str:
    """Return a clean Arabic sentence repeated to reach n_words."""
    word = "الكلمة"
    return " ".join([word] * n_words)


def _english_sentence(n_words: int = 20) -> str:
    word = "word"
    return " ".join([word] * n_words)


# ─── empty / no text ────────────────────────────────────────────────────────

def test_empty_text_fails():
    result = evaluate_page_quality(text="", confidence=90.0)
    assert not result.passed
    assert result.word_count == 0
    assert "Empty" in result.reason


def test_whitespace_only_fails():
    result = evaluate_page_quality(text="   \n\t  ", confidence=90.0)
    assert not result.passed


# ─── low word count + low confidence ────────────────────────────────────────

def test_few_words_and_low_confidence_fails():
    # 5 words, confidence 50% → both below thresholds → fail
    result = evaluate_page_quality(text="word " * 5, confidence=50.0, min_words=15)
    assert not result.passed
    assert result.word_count == 5
    assert "Insufficient word count" in result.reason


def test_few_words_but_high_confidence_passes():
    # 5 words but very high confidence (≥75%) → word count alone doesn't fail
    result = evaluate_page_quality(text="word " * 5, confidence=80.0, min_words=15)
    # confidence check will then run; should pass since 80% > 65%
    assert result.passed


# ─── confidence threshold ────────────────────────────────────────────────────

def test_low_confidence_fails():
    text = _english_sentence(20)
    result = evaluate_page_quality(text=text, confidence=40.0, min_confidence=65.0)
    assert not result.passed
    assert "confidence too low" in result.reason


def test_exactly_at_confidence_threshold_passes():
    text = _english_sentence(20)
    result = evaluate_page_quality(text=text, confidence=65.0, min_confidence=65.0)
    assert result.passed


def test_above_confidence_threshold_passes():
    text = _english_sentence(20)
    result = evaluate_page_quality(text=text, confidence=82.0)
    assert result.passed


# ─── sanity ratio ───────────────────────────────────────────────────────────

def test_high_garbage_ratio_fails():
    # Replace most of the text with garbage bytes / symbols
    garbage = "☃✓⚠⛔" * 100  # non-word, non-Arabic symbols
    result = evaluate_page_quality(text=garbage, confidence=70.0)
    assert not result.passed
    # Either garbage check or word count fires first
    assert result.passed is False


# ─── repetition anomaly ──────────────────────────────────────────────────────

def test_repetition_anomaly_fails():
    # Three instances of 5+ consecutive identical chars
    text = _english_sentence(20) + " اااااا بببببب جججججج"
    result = evaluate_page_quality(text=text, confidence=80.0)
    assert not result.passed
    assert "Repetition" in result.reason


def test_two_repetitions_still_passes():
    # Only 2 instances — below the threshold of 3
    text = _english_sentence(20) + " اااااا بببببب"
    result = evaluate_page_quality(text=text, confidence=80.0)
    assert result.passed


# ─── Arabic text ─────────────────────────────────────────────────────────────

def test_clean_arabic_text_passes():
    text = _arabic_sentence(25)
    result = evaluate_page_quality(text=text, confidence=78.0)
    assert result.passed
    assert result.word_count == 25


def test_mixed_arabic_english_passes():
    # Build a sentence with ≥15 words and confidence ≥75% so all checks pass
    text = ("التركيب الكيميائي chemical composition مادة " * 4).strip()
    result = evaluate_page_quality(text=text, confidence=75.0)
    assert result.passed


# ─── return type ─────────────────────────────────────────────────────────────

def test_returns_quality_gate_result_dataclass():
    result = evaluate_page_quality(text=_english_sentence(20), confidence=75.0)
    assert isinstance(result, QualityGateResult)
    assert hasattr(result, "passed")
    assert hasattr(result, "confidence")
    assert hasattr(result, "word_count")
    assert hasattr(result, "sanity_ratio")
    assert hasattr(result, "reason")


# ─── custom thresholds ───────────────────────────────────────────────────────

def test_custom_min_confidence_stricter():
    text = _english_sentence(20)
    result = evaluate_page_quality(text=text, confidence=70.0, min_confidence=80.0)
    assert not result.passed


def test_custom_min_words_stricter():
    text = _english_sentence(10)
    result = evaluate_page_quality(text=text, confidence=85.0, min_words=20)
    # 10 words < 20, confidence 85 ≥ 75 → word count check passes,
    # but confidence check passes too; only word_count<min_words AND conf<75 triggers that path
    # So this should actually PASS because confidence >= 75.0
    assert result.passed
