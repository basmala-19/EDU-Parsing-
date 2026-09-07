"""
ParserRouter — content-based language detection and document routing.

Features:
  - Content-based language detection via character frequency and langdetect.
  - Primary Local Engine: DoclingParser (free, offline, layout-aware).
  - Scanned Document Engine: TesseractParser (free, offline OCR).
  - Cloud OCR Engine: OpenRouter vision models (Gemini, Qwen, Gemma).
  - Emergency Fallback Engine: LlamaParser (LlamaCloud API, credit-conserving fallback).
"""
from __future__ import annotations

import re
import pymupdf as fitz
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from parsers.base import BaseParser
from parsers.docling_parser import DoclingParser
from parsers.gemini_ocr_parser import GeminiOCRParser, openrouter_key_available
from parsers.llama_parser import LlamaParser
from parsers.liteparse_parser import LiteParseParser
from parsers.ocr_parser import TesseractParser
from routing.probe import probe_document, ProbeResult

DetectorFactory.seed = 0


_FILENAME_LANG_PATTERNS = [
    (re.compile(r"(?:^|[_\- ])(?:ar|ara|arabic)(?:$|[_\- ])", re.I), "ar"),
    (re.compile(r"(?:^|[_\- ])(?:en|eng|english)(?:$|[_\- ])", re.I), "en"),
]


def detect_language_from_filename(file_path: str) -> str | None:
    """Check filename for explicit language tokens (e.g. Biology_ARABIC_Sec3.pdf -> 'ar')."""
    from pathlib import Path
    stem = Path(file_path).stem
    for pattern, lang in _FILENAME_LANG_PATTERNS:
        if pattern.search(stem):
            return lang
    return None


def detect_language_from_text(text: str) -> str:
    """Infer Arabic/English/Mixed from text using EDU-RAG-new-version majority rule."""
    arabic_count = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))

    if arabic_count and latin_count:
        # Treat small number of Latin tokens (formulas, terminology) inside Arabic as Arabic
        if arabic_count >= 3 * max(latin_count, 1):
            return "ar"
        if latin_count >= 3 * max(arabic_count, 1):
            return "en"
        return "mixed"
    if arabic_count:
        return "ar"
    if latin_count:
        return "en"
    return "unknown"


def detect_language_from_content(file_path: str, sample_pages: int = 5) -> str:
    """Detect primary document language from actual text content or filename/OSD.

    Args:
        file_path: Path to the input document.
        sample_pages: Number of initial pages to sample.

    Returns:
        ``ar``, ``en``, ``mixed``, or ``unknown``.
    """
    # 1. Quick check: explicit language in filename
    filename_lang = detect_language_from_filename(file_path)

    try:
        doc = fitz.open(file_path)
        sampled_text = []

        for i in range(min(sample_pages, len(doc))):
            text = doc[i].get_text().strip()
            if text:
                sampled_text.append(text)
        doc.close()

        if not sampled_text:
            # Scanned document — if filename had explicit marker, trust it over OSD
            if filename_lang:
                return filename_lang
            return _detect_language_with_osd(file_path)

        full_text = " ".join(sampled_text)
        detected = detect_language_from_text(full_text)
        if detected != "unknown":
            return detected

        if filename_lang:
            return filename_lang

        # Statistical detection
        clean_text = re.sub(r"[0-9\s\.\,\;\:\-\(\)\[\]\{\}]+", " ", full_text).strip()
        if len(clean_text) > 50:
            stat_lang = detect(clean_text[:1000])
            if stat_lang.startswith("ar"):
                return "ar"

        return "en" if len(re.findall(r"[A-Za-z]", full_text)) else "unknown"
    except (LangDetectException, Exception):
        if filename_lang:
            return filename_lang
        return _detect_language_with_osd(file_path)


def _detect_language_with_osd(file_path: str, sample_pages: int = 10) -> str:
    """Detect script for scanned PDFs using multi-page OSD voting + pixel fallback.

    Strategy:
      1. Try Tesseract OSD on up to ``sample_pages`` pages with confidence >= 10.
         Collect per-page votes (arabic / latin / other) and return the majority.
      2. If OSD yields no confident votes, fall back to pixel-level Arabic glyph
         detection: compare the mean pixel brightness of Arabic-range character
         crops versus background — Arabic script tends to produce darker, more
         connected strokes than Latin on binarised images.  We use a simpler
         proxy: count dark-pixel density in the upper-right quadrant (where Arabic
         text starts) vs upper-left quadrant (where Latin text starts).
      3. If even pixel analysis is inconclusive, return ``"ar"`` as a conservative
         default for scanned PDFs (Tesseract ara is more permissive than eng on
         ambiguous glyphs, and GeminiOCR will be tried first anyway when the key
         is available).
    """
    try:
        import io
        from PIL import Image
        import pytesseract

        document = fitz.open(file_path)
        arabic_votes = 0
        latin_votes = 0
        total_votes = 0

        try:
            pages_to_check = list(document[:sample_pages])
            for page in pages_to_check:
                try:
                    pixmap = page.get_pixmap(dpi=150, alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
                    confidence = float(osd.get("script_confidence", 0))
                    script = str(osd.get("script", "")).lower()
                    if confidence >= 10:
                        total_votes += 1
                        if script == "arabic":
                            arabic_votes += 1
                        elif script in {"latin", "cyrillic"}:
                            latin_votes += 1
                except Exception:
                    # OSD failed on this page — try next
                    continue
        finally:
            document.close()

        # Majority vote from OSD
        if total_votes > 0:
            if arabic_votes > latin_votes:
                return "ar"
            if latin_votes > arabic_votes:
                return "en"
            # Tie → arabic-biased default for scanned
            return "ar"

        # ── Pixel fallback: dark-pixel density heuristic ──────────────────────
        return _detect_language_pixel_fallback(file_path)

    except Exception:
        return "ar"  # conservative default for scanned PDFs


def _detect_language_pixel_fallback(file_path: str, sample_pages: int = 5) -> str:
    """Pixel-level Arabic vs Latin script heuristic for completely unreadable scans.

    Arabic text is written right-to-left and tends to cluster dense ink strokes
    in the right portion of a text line.  We rasterise a few pages, binarise them,
    and compare the dark-pixel density in the right 40 % of the page versus the
    left 40 %.  A consistently higher right-side density across multiple pages is
    a reliable signal for Arabic script.

    Returns ``"ar"``, ``"en"``, or ``"unknown"``.
    """
    try:
        import io
        import numpy as np
        from PIL import Image, ImageFilter

        doc = fitz.open(file_path)
        right_heavy_count = 0
        total_pages_checked = 0

        for page in list(doc[:sample_pages]):
            try:
                pixmap = page.get_pixmap(dpi=100, alpha=False)
                img = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")
                # Mild blur to reduce noise before thresholding
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
                arr = np.array(img)

                # Binarise: dark pixels = text (value < 128)
                dark = (arr < 128).astype(float)

                w = dark.shape[1]
                # Skip top/bottom 10 % (headers/footers/page numbers)
                h = dark.shape[0]
                content = dark[int(h * 0.1) : int(h * 0.9), :]

                left_density = content[:, : int(w * 0.4)].mean()
                right_density = content[:, int(w * 0.6) :].mean()

                if right_density > left_density * 1.05:  # right is at least 5 % denser
                    right_heavy_count += 1
                total_pages_checked += 1
            except Exception:
                continue

        doc.close()

        if total_pages_checked == 0:
            return "ar"  # conservative default

        arabic_ratio = right_heavy_count / total_pages_checked
        if arabic_ratio >= 0.5:
            return "ar"
        return "en"

    except Exception:
        return "ar"  # conservative default for scanned


class ParserRouter:
    """Document router that selects local parsers first and reserves cloud APIs for fallbacks."""

    def __init__(self, ocr_engine: str = "auto") -> None:
        """
        Args:
            ocr_engine: Which cloud OCR engine to promote for scanned documents.
                        'auto'     — use Gemini if OPENROUTER_API_KEY is set
                        'gemini'/'openrouter' — force the configured OpenRouter vision model
                        'tesseract'— force local Tesseract (offline, free)
        """
        self._ocr_engine_pref = ocr_engine.lower().strip() if ocr_engine else "auto"

        self._registry: dict[str, BaseParser] = {
            "docling": DoclingParser(),
            "liteparse": LiteParseParser(),
            "tesseract": TesseractParser(lang="ara+eng"),
            "ocr_parser": TesseractParser(lang="ara+eng"),
            "gemini_ocr": GeminiOCRParser(),
            "llamaparse": LlamaParser(language="auto"),
        }
        self._llama_cache: dict[str, LlamaParser] = {}

    def get_llama_parser(self, language: str) -> LlamaParser:
        """Retrieve or instantiate a LlamaParser configured for the specified language."""
        if language not in self._llama_cache:
            self._llama_cache[language] = LlamaParser(language=language)
        return self._llama_cache[language]

    def _preferred_cloud_ocr(self, language: str) -> BaseParser | None:
        """Return the preferred OCR parser based on user preference and available keys.

        Returns None when no cloud OCR engine is configured so the caller can
        fall back to the local Tesseract engine.
        """
        pref = self._ocr_engine_pref

        # Explicit user selection
        if pref in {"gemini", "openrouter"}:
            return self._registry["gemini_ocr"]
        if pref == "tesseract":
            return None  # caller uses local Tesseract directly

        # Auto mode: OpenRouter Gemini is primary VLM for scans and Arabic
        if openrouter_key_available():
            return self._registry["gemini_ocr"]
        return None  # No cloud key → local Tesseract only

    def select(self, probe: ProbeResult, file_path: str | None = None) -> BaseParser:
        """Select the optimal parser for the given document."""
        if not probe.is_born_digital:
            return self._registry.get("ocr_parser", self._registry.get("tesseract"))
        return self._registry["docling"]

    def build_chain(self, probe: ProbeResult, language: str) -> list[BaseParser]:
        """Build an ordered, content-driven parser fallback chain.

        Born-digital PDFs retain their text layer, so Docling is the preferred
        layout-aware parser.  Scanned PDFs have no usable text layer; running
        Docling first only consumes RAM, therefore local Tesseract starts the
        scan path and LlamaParse remains the cloud recovery option.

        The configured OpenRouter vision model is promoted ahead of
        the free local engines in two cases:

          - Scanned documents, any language: Tesseract silently returns 0
            characters on pages with real content, especially mixed
            text+diagram layouts.
          - Born-digital Arabic: LiteParse's reading-order reconstruction
            scrambles Arabic bidi text; cloud OCR on the rasterised page
            sidesteps that entirely.

        The preferred cloud engine is resolved by _preferred_cloud_ocr() which
        respects the ocr_engine constructor parameter and falls back through
        OpenRouter in auto mode when an OpenRouter key is configured.
        """
        ocr_language = {"ar": "ara", "en": "eng"}.get(language, "ara+eng")
        llama_language = language if language in {"ar", "en"} else "auto"

        cloud_ocr = self._preferred_cloud_ocr(language)

        # A named cloud engine is an explicit user choice, not a preference.
        # Propagate provider errors instead of silently switching engines.
        if self._ocr_engine_pref in {"gemini", "openrouter"}:
            return [cloud_ocr] if cloud_ocr is not None else [TesseractParser(lang=ocr_language)]

        if probe.is_born_digital:
            if cloud_ocr and language == "ar":
                # Skip LiteParse: known RTL/bidi corruption on Arabic text.
                return [
                    cloud_ocr,
                    self._registry["docling"],
                    self.get_llama_parser(llama_language),
                    TesseractParser(lang=ocr_language),
                ]
            return [
                self._registry["liteparse"],
                self._registry["docling"],
                self.get_llama_parser(llama_language),
                TesseractParser(lang=ocr_language),
            ]

        if cloud_ocr:
            return [
                cloud_ocr,
                TesseractParser(lang=ocr_language),
                self.get_llama_parser(llama_language),
            ]
        return [
            TesseractParser(lang=ocr_language),
            self.get_llama_parser(llama_language),
        ]

    def route(self, file_path: str) -> tuple[BaseParser, ProbeResult, str]:
        """Probe document, detect content language, and return (selected_parser, probe, detected_language)."""
        probe = probe_document(file_path)
        detected_lang = detect_language_from_content(file_path)
        parser = self.select(probe, file_path)
        return parser, probe, detected_lang
