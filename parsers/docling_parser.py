"""
DoclingParser — born-digital PDFs, free, self-hosted.

First run downloads the Granite-Docling-258M model (~1 GB from HuggingFace).
Subsequent runs use the cached model.

Install: pip install docling
"""
import os
import logging

import pymupdf as fitz

from .base import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Parse born-digital PDFs using IBM Docling (Granite model)."""

    name = "docling"
    # A one-page conversion preserves an unambiguous source-page marker in
    # exported Markdown.  Larger ranges lose page-level provenance in
    # Docling's current Markdown exporter and made zero-page detection
    # impossible.  This can be increased only after provenance is retained.
    _MAX_PAGES_PER_CONVERSION = 1

    def __init__(self) -> None:
        self.last_page_markers: list[int] = []

    @staticmethod
    def _has_usable_text_layer(file_path: str, sample_pages: int = 3) -> bool:
        """Detect digital PDFs before starting Docling's expensive OCR stage."""
        pdf = fitz.open(file_path)
        try:
            sample = "".join(page.get_text() for page in pdf[:sample_pages])
            return len(sample.strip()) >= 200
        finally:
            pdf.close()

    def parse(self, file_path: str) -> str:
        # Native Windows often lacks the MSVC ``cl.exe`` compiler required by
        # PyTorch Inductor.  Compilation is an optimisation, not a parsing
        # requirement, so keep Docling portable and let it run eagerly.
        os.environ.setdefault("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "false")
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        has_text_layer = self._has_usable_text_layer(file_path)
        if not has_text_layer:
            # Let the router's fallback chain select the cloud VLM / Tesseract
            # for scans.  Running RapidOCR over a large raster textbook here
            # has proved both memory-heavy and less reliable for Arabic.
            raise RuntimeError("No usable PDF text layer; defer scanned OCR to LlamaParse/Tesseract")

        options = PdfPipelineOptions()
        # Do not invoke RapidOCR on a born-digital textbook: its own PDF text
        # is more accurate and the OCR stage wastes significant memory.
        options.do_ocr = False
        # Verified against installed Docling 2.118.0.  It already defaults to
        # ACCURATE, but setting it explicitly prevents version drift.
        from docling.datamodel.pipeline_options import TableFormerMode
        options.do_table_structure = True
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        # Exporting picture assets is enabled only for born-digital documents;
        # image_extractor.py remains the portable fallback for PDFs whose
        # Markdown exporter still emits a bare <!-- image --> placeholder.
        options.generate_picture_images = True
        # Process large books conservatively; default batches of four pages can
        # exhaust Windows/ONNXRuntime memory on illustrated textbooks.
        options.ocr_batch_size = 1
        options.layout_batch_size = 1
        options.table_batch_size = 1
        options.accelerator_options.num_threads = 1

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )
        pdf = fitz.open(file_path)
        try:
            page_count = len(pdf)
            source_chars = sum(len(page.get_text().strip()) for page in pdf)
        finally:
            pdf.close()

        # Docling's Windows layout backend may retain large rendered pages over
        # a full-book conversion.  Bound each invocation to ten pages so the
        # backend releases page memory between ranges while preserving Docling
        # as the primary parser for born-digital books.
        markdown_parts: list[str] = []
        self.last_page_markers = []
        for first_page in range(1, page_count + 1, self._MAX_PAGES_PER_CONVERSION):
            last_page = min(first_page + self._MAX_PAGES_PER_CONVERSION - 1, page_count)
            result = converter.convert(file_path, page_range=(first_page, last_page))
            if result.status != ConversionStatus.SUCCESS:
                raise RuntimeError(
                    f"Docling conversion status on pages {first_page}-{last_page}: {result.status.value}"
                )
            markdown_parts.append(
                f"<!-- page: {first_page} -->\n{result.document.export_to_markdown()}"
            )
            self.last_page_markers.append(first_page)

        logger.info(
            "Docling page markers for %s: %s",
            file_path,
            self.last_page_markers,
        )

        markdown = "\n\n".join(markdown_parts)
        # A non-empty partial conversion is still a failure for RAG.  This
        # guards against Docling returning only a few successful pages after
        # hidden stage errors.
        if source_chars and len(markdown.strip()) / source_chars < 0.20:
            raise RuntimeError(
                f"Docling output is incomplete ({len(markdown.strip())}/{source_chars} text characters)"
            )
        return markdown
