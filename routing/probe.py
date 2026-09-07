"""
Document probe — extracts structural fingerprints from a PDF without
invoking any heavyweight parser.

Uses PyMuPDF (fitz) page.get_text("dict") to measure:
  - Character density per page  → born-digital vs. scanned
  - Image count                 → actual count, not estimated
  - Distinct font sizes         → proxy for heading presence
  - Text block density          → proxy for complex layout / multi-column
"""
import time
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF


@dataclass
class ProbeResult:
    file: str
    num_pages: int
    avg_chars_per_page: float
    is_born_digital: bool
    image_count: int
    distinct_font_sizes: int
    likely_has_headings: bool
    text_block_count: int
    likely_has_complex_layout: bool
    route: str
    probe_time_seconds: float
    per_page: list = field(default_factory=list)


def probe_document(
    path: str,
    min_chars_per_page: int = 200,
    block_density_threshold: int = 12,
) -> ProbeResult:
    """Fingerprint a document to determine the appropriate parser.

    Args:
        path:                   Path to the PDF.
        min_chars_per_page:     Minimum avg characters/page to qualify as born-digital.
        block_density_threshold: Avg text blocks/page above which layout is considered complex.

    Returns:
        ProbeResult with all fingerprinting fields populated.
    """
    t0 = time.time()
    doc = fitz.open(path)

    total_chars = 0
    total_images = 0
    total_blocks = 0
    font_sizes: set[float] = set()
    per_page: list[dict] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_chars = 0
        page_blocks = 0

        for block in page_dict.get("blocks", []):
            if block.get("type") == 1:  # image block
                total_images += 1
                continue
            page_blocks += 1
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    page_chars += len(span.get("text", "").strip())
                    font_sizes.add(round(span.get("size", 0), 1))

        total_chars += page_chars
        total_blocks += page_blocks
        per_page.append({"page": page_num + 1, "chars": page_chars, "blocks": page_blocks})

    num_pages = len(doc)
    avg_chars = total_chars / max(num_pages, 1)
    avg_blocks = total_blocks / max(num_pages, 1)

    is_born_digital = avg_chars >= min_chars_per_page
    likely_has_headings = len(font_sizes) >= 2
    likely_has_complex_layout = avg_blocks >= block_density_threshold

    if not is_born_digital:
        route = "ocr_parser"
    elif likely_has_complex_layout:
        route = "llamaparse"
    else:
        route = "docling"

    return ProbeResult(
        file=path,
        num_pages=num_pages,
        avg_chars_per_page=round(avg_chars, 1),
        is_born_digital=is_born_digital,
        image_count=total_images,
        distinct_font_sizes=len(font_sizes),
        likely_has_headings=likely_has_headings,
        text_block_count=total_blocks,
        likely_has_complex_layout=likely_has_complex_layout,
        route=route,
        probe_time_seconds=round(time.time() - t0, 4),
        per_page=per_page,
    )
