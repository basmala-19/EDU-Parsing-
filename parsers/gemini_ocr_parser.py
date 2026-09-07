"""
GeminiOCRParser — full-page OCR using a hosted vision-language model via
OpenRouter, positioned as the *primary* engine (not just a refinement pass)
for two proven-weak spots in the free local chain:

  1. Scanned documents (any language). Confirmed via a controlled A/B test
     on Biology_ARABIC_Sec3.pdf: Tesseract returned 0 characters on 2/3
     sampled real-content pages (one text-only, one mixed text+diagram —
     Tesseract's layout analysis broke on the mixed layout), and even on
     the one page it did read, character-level errors were frequent
     (e.g. "الفدة النطامية" instead of "الغدة النخامية"). Gemini 3 Flash
     read all three pages correctly, including diagram labels.

  2. Born-digital Arabic documents. Confirmed on Physics-Ar-EB-part1.pdf:
     LiteParse's spatial/PDFium-based reading-order reconstruction does not
     handle Arabic bidi text correctly and produces scrambled output, even
     though raw PyMuPDF text extraction of the same page is readable. Since
     this parser rasterises the page and reads it as an image, it sidesteps
     the PDF text layer (and its bidi bug) entirely.

Why this isn't just "run QwenVLRefiner on every element" (the old design):
that only fires on *already-flagged* low-confidence/empty elements, so it
can't help when the upstream parser silently drops or scrambles content
with a misleadingly high confidence score. This parser instead treats the
VLM as the first reader of the page, with Tesseract retained as the
zero-cost fallback for when no OPENROUTER_API_KEY is configured.

Cost: ~$0.003-0.004/page with the default google/gemini-3-flash-preview
(~$0.55 for a 167-page book) — see parsers/openrouter_vlm_client.py for the
full pricing comparison across model options.
"""
from __future__ import annotations

import io
import logging
import os

import pymupdf as fitz
from PIL import Image

from parsers.base import BaseParser
from parsers.ocr_parser import OCR_NO_TEXT_MARKER
from parsers.openrouter_vlm_client import OpenRouterVLMClient

logger = logging.getLogger(__name__)

_OCR_PROMPT_AR_EN = (
    "استخرج كل النص من هذه الصفحة بالترتيب الصحيح وبدون أي تعليق إضافي. "
    "حافظ على فواصل الأسطر والفقرات كما هي في الصورة، واقرأ تسميات الأشكال "
    "والرسوم التخطيطية إن وجدت. أرجع النص فقط بدون شرح.\n\n"
    "Extract all text from this page in correct reading order, no extra "
    "commentary. Preserve paragraph breaks and read any diagram/figure "
    "labels present. Return only the extracted text."
)

DEFAULT_DPI = 140


def openrouter_key_available() -> bool:
    """Cheap, side-effect-free check used by the router to decide whether
    GeminiOCRParser can be placed ahead of Tesseract in the parser chain."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


class GeminiOCRParser(BaseParser):
    """Full-page OCR via a hosted VLM (default: Gemini 3 Flash on OpenRouter).

    Produces the same ``<!-- page: N -->`` / ``<!-- ocr_confidence: XX.XX -->``
    marker format as TesseractParser (parsers/ocr_parser.py) so it is a
    drop-in replacement anywhere in the parser chain — educational/
    rule_based_parser.py already knows how to consume this format.
    """

    name = "gemini_ocr"

    def __init__(
        self,
        client: OpenRouterVLMClient | None = None,
        dpi: int = DEFAULT_DPI,
        prompt: str = _OCR_PROMPT_AR_EN,
    ) -> None:
        self.client = client
        self.dpi = dpi
        self.prompt = prompt
        self.last_run_profile: dict[str, int | str] = {}

    def _get_client(self) -> OpenRouterVLMClient:
        if self.client is None:
            self.client = OpenRouterVLMClient()
        return self.client

    def parse(self, file_path: str) -> str:
        client = self._get_client()  # raises RuntimeError early if no API key
        doc = fitz.open(file_path)
        page_count = doc.page_count
        full_text: list[str] = []
        empty_pages = 0
        error_pages = 0
        consecutive_errors = 0
        try:
            for page_num, page in enumerate(doc, 1):
                pix = page.get_pixmap(dpi=self.dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                try:
                    text = client(img, self.prompt).strip()
                    consecutive_errors = 0
                except Exception as exc:
                    error_str = str(exc)
                    logger.warning("OpenRouter VLM (%s) failed on page %s: %s", client.model, page_num, exc)
                    error_pages += 1
                    consecutive_errors += 1
                    full_text.append(
                        f"<!-- page: {page_num} -->\n"
                        f"<!-- ocr_confidence: 0.00 -->\n"
                        f"{OCR_NO_TEXT_MARKER}"
                    )
                    # If OpenRouter key is unauthorized/expired/quota exceeded, stop spinning on remaining pages
                    if "401" in error_str or "402" in error_str or "403" in error_str or consecutive_errors >= 5:
                        logger.error(
                            "OpenRouter VLM (%s) aborted early at page %s due to persistent/auth errors: %s",
                            client.model, page_num, exc
                        )
                        # Fill remaining pages with no_text markers so page indexing remains aligned
                        for remaining_p in range(page_num + 1, page_count + 1):
                            full_text.append(
                                f"<!-- page: {remaining_p} -->\n"
                                f"<!-- ocr_confidence: 0.00 -->\n"
                                f"{OCR_NO_TEXT_MARKER}"
                            )
                            empty_pages += 1
                        break
                    continue

                if not text or len(text) < 3:
                    empty_pages += 1
                    full_text.append(
                        f"<!-- page: {page_num} -->\n"
                        f"<!-- ocr_confidence: 0.00 -->\n"
                        f"{OCR_NO_TEXT_MARKER}"
                    )
                    continue

                # No native per-page confidence from a VLM response; a fixed
                # high value reflects the observed A/B accuracy (see module
                # docstring) without implying false precision.
                full_text.append(
                    f"<!-- page: {page_num} -->\n"
                    f"<!-- ocr_confidence: 95.00 -->\n"
                    f"{text}"
                )
        finally:
            doc.close()

        self.last_run_profile = {
            "pages": page_count,
            "empty_pages": empty_pages,
            "error_pages": error_pages,
            "model": client.model,
        }
        return "\n\n".join(full_text)
