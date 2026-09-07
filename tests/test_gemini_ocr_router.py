"""Guardrails for GeminiOCRParser and its promotion in the router chain.

Covers the two production decisions made from the Biology/Physics A/B test:
  - Gemini becomes the primary OCR engine for scanned docs (any language)
    when OPENROUTER_API_KEY is set, with Tesseract kept as the fallback.
  - Gemini also becomes primary for born-digital Arabic (skipping LiteParse,
    whose bidi bug was confirmed separately in test_liteparse_parser.py /
    test_arabic.py), while born-digital English is left on LiteParse.
  - None of this activates without an API key: the chain must be byte-for-
    byte identical to the pre-existing behaviour in that case.
"""
import pytest
from PIL import Image

from parsers.gemini_ocr_parser import GeminiOCRParser, openrouter_key_available
from parsers.ocr_parser import OCR_NO_TEXT_MARKER
from routing.probe import ProbeResult
from routing.router import ParserRouter


def _probe(is_born_digital: bool) -> ProbeResult:
    return ProbeResult(
        file="dummy.pdf",
        num_pages=5,
        avg_chars_per_page=100.0 if is_born_digital else 0.0,
        is_born_digital=is_born_digital,
        image_count=2,
        distinct_font_sizes=3,
        likely_has_headings=True,
        text_block_count=10,
        likely_has_complex_layout=False,
        route="docling" if is_born_digital else "ocr_parser",
        probe_time_seconds=0.1,
        per_page=[],
    )


class TestRouterPromotion:
    def test_no_key_scanned_chain_unchanged(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        router = ParserRouter()
        chain = router.build_chain(_probe(is_born_digital=False), "ar")
        assert [p.name for p in chain] == ["tesseract", "llamaparse"]

    def test_no_key_born_digital_arabic_chain_unchanged(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        router = ParserRouter()
        chain = router.build_chain(_probe(is_born_digital=True), "ar")
        assert [p.name for p in chain] == ["liteparse", "docling", "llamaparse", "tesseract"]

    def test_with_key_scanned_promotes_gemini_first(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        router = ParserRouter()
        chain = router.build_chain(_probe(is_born_digital=False), "ar")
        assert [p.name for p in chain] == ["gemini_ocr", "tesseract", "llamaparse"]

    def test_with_key_scanned_english_also_promotes_gemini(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        router = ParserRouter()
        chain = router.build_chain(_probe(is_born_digital=False), "en")
        assert chain[0].name == "gemini_ocr"

    def test_with_key_born_digital_arabic_skips_liteparse(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        router = ParserRouter()
        chain = router.build_chain(_probe(is_born_digital=True), "ar")
        assert [p.name for p in chain] == ["gemini_ocr", "docling", "llamaparse", "tesseract"]
        assert "liteparse" not in [p.name for p in chain]

    def test_with_key_born_digital_english_keeps_liteparse_first(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        router = ParserRouter()
        chain = router.build_chain(_probe(is_born_digital=True), "en")
        assert chain[0].name == "liteparse"


class TestOpenrouterKeyAvailable:
    def test_false_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert openrouter_key_available() is False

    def test_true_when_set(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert openrouter_key_available() is True


class TestGeminiOCRParser:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        parser = GeminiOCRParser()
        with pytest.raises(RuntimeError):
            parser.parse("test_docs/scanned.pdf")

    def test_parses_pages_with_mocked_client(self, monkeypatch):
        calls = []

        def fake_client(image, prompt):
            calls.append((image.size, prompt))
            return "نص مستخرج من الصفحة"

        parser = GeminiOCRParser(client=_FakeClient(fake_client))
        markdown = parser.parse("test_docs/scanned.pdf")

        assert "<!-- page: 1 -->" in markdown
        assert "<!-- ocr_confidence: 95.00 -->" in markdown
        assert "نص مستخرج من الصفحة" in markdown
        assert len(calls) >= 1

    def test_empty_response_marks_no_text_detected(self, monkeypatch):
        parser = GeminiOCRParser(client=_FakeClient(lambda image, prompt: ""))
        markdown = parser.parse("test_docs/scanned.pdf")
        assert OCR_NO_TEXT_MARKER in markdown
        assert "<!-- ocr_confidence: 0.00 -->" in markdown

    def test_per_page_exception_is_isolated_and_tagged(self):
        parser = GeminiOCRParser(
            client=_FakeClient(lambda image, prompt: (_ for _ in ()).throw(RuntimeError("simulated API error")))
        )
        markdown = parser.parse("test_docs/scanned.pdf")
        # One-page fixture: a per-page exception must be caught and tagged,
        # not raised out of parse() and not aborting the whole document.
        assert OCR_NO_TEXT_MARKER in markdown
        assert "<!-- page: 1 -->" in markdown
        assert parser.last_run_profile["error_pages"] == 1


class _FakeClient:
    """Stand-in for OpenRouterVLMClient — same callable signature, no network."""

    model = "google/gemini-3-flash-preview"

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, image: Image.Image, prompt: str) -> str:
        return self._fn(image, prompt)
