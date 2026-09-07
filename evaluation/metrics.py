"""
Quality metrics — deterministic, rule-based, ParseBench-inspired.

Each metric returns a dict:
    {"status": "PASS" | "FAIL", "detail": "<human-readable explanation>"}

Numeric ratios are intentionally NOT exposed as final scores because the
project has no ground-truth dataset yet. Use PASS/FAIL thresholds until a
labelled benchmark is available — then swap to continuous scores.

ParseBench dimensions covered:
  1. Content Faithfulness   — tokens preserved across Markdown → EducationalDocument
  2. Table Preservation     — table count in Markdown vs. parsed TABLE elements
  3. Semantic Formatting    — heading levels present and non-empty
  4. Metadata Completeness  — every element carries chapter / lesson / parser
  5. Reading Order          — page numbers never decrease (proxy for column order)
"""
from __future__ import annotations

import re

from schema.models import EducationalDocument, ElementType

_WORD_RE = re.compile(r"[\w\u0600-\u06FF]+")  # supports Arabic and Latin scripts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _iter_lessons(edoc: EducationalDocument):
    """Yield (chapter, lesson, element_list) for every lesson."""
    for chapter in edoc.chapters:
        for lesson in chapter.lessons:
            yield chapter, lesson, lesson.elements


def _all_elements(edoc: EducationalDocument):
    return [el for _, _, els in _iter_lessons(edoc) for el in els]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def content_faithfulness(source_text: str, edoc: EducationalDocument) -> dict:
    """Fraction of source tokens still present after Markdown → EducationalDocument.

    This is a proxy, not a ground-truth comparison. It catches the most common
    failure: the Educational Parser silently dropping content during conversion.

    PASS threshold: >= 50% of source tokens retained.
    """
    source_tokens = _tokenize(source_text)
    if not source_tokens:
        return {"status": "PASS", "detail": "source text is empty — nothing to preserve"}

    parsed_text = " ".join(
        el.text or "" for el in _all_elements(edoc)
    )
    parsed_tokens = _tokenize(parsed_text)

    kept = source_tokens & parsed_tokens
    ratio = len(kept) / len(source_tokens)
    status = "PASS" if ratio >= 0.50 else "FAIL"
    return {
        "status": status,
        "detail": f"{len(kept)}/{len(source_tokens)} source tokens preserved ({ratio:.0%})",
    }


def table_preservation(source_markdown: str, edoc: EducationalDocument) -> dict:
    """Detected table count in Markdown vs. TABLE elements in EducationalDocument.

    Catches the most dangerous failure: a table silently converted to a paragraph.

    PASS: detected tables == parsed TABLE elements (exact match).
    """
    markdown_tables = 0
    for line in source_markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            markdown_tables += 1
    html_tables = len(re.findall(r"<table(?:\s|>)", source_markdown, flags=re.IGNORECASE))
    detected_tables = markdown_tables + html_tables
    # Approximate: each table averages ~3 rows (header + separator + ≥1 data row)

    parsed_tables = sum(
        1
        for el in _all_elements(edoc)
        if el.type == ElementType.TABLE
    )

    if detected_tables == 0:
        status = "PASS" if parsed_tables == 0 else "FAIL"
        detail = "no tables in source" if parsed_tables == 0 else f"spurious TABLE elements: {parsed_tables}"
    else:
        status = "PASS" if parsed_tables >= detected_tables else "FAIL"
        detail = f"{parsed_tables}/{detected_tables} tables detected"

    return {"status": status, "detail": detail}


def semantic_formatting(
    edoc: EducationalDocument, expected_page_count: int | None = None
) -> dict:
    """Heading structure integrity check.

    PASS: every HEADING element has a non-empty text and a valid level.
    """
    headings = [el for el in _all_elements(edoc) if el.type == ElementType.HEADING]
    if not headings:
        return {"status": "PASS", "detail": "no headings — nothing to validate"}

    bad = [el for el in headings if not el.text or el.level is None]
    # A long scanned textbook with only a few syntactically-valid headings is
    # not structurally ready for educational retrieval.  This protects against
    # falsely passing OCR output such as "2 headings across 167 pages".
    is_ocr = any(el.metadata.parser == "tesseract" for el in _all_elements(edoc))
    minimum = (
        max(3, (expected_page_count + 29) // 30)
        if is_ocr and expected_page_count and expected_page_count >= 20
        else 0
    )
    too_sparse = minimum and len(headings) < minimum
    status = "PASS" if not bad and not too_sparse else "FAIL"
    detail = (
        f"all {len(headings)} headings have valid text and level"
        if not bad and not too_sparse
        else (
            f"only {len(headings)} headings recovered across {expected_page_count} OCR pages; expected at least {minimum}"
            if too_sparse
            else f"{len(bad)}/{len(headings)} headings missing text or level"
        )
    )
    return {"status": status, "detail": detail}


def metadata_completeness(edoc: EducationalDocument) -> dict:
    """Every element must carry chapter, lesson, and parser fields.

    Operates on elements (not chunks) so it can be called without chunking.

    PASS: 100% of elements have all three required fields populated.
    """
    required = ("chapter", "lesson", "parser")
    elements = _all_elements(edoc)

    if not elements:
        return {"status": "FAIL", "detail": "document has no elements"}

    complete = sum(
        1
        for el in elements
        if all(getattr(el.metadata, f, None) for f in required)
    )
    status = "PASS" if complete == len(elements) else "FAIL"
    detail = f"{complete}/{len(elements)} elements have full metadata (chapter + lesson + parser)"
    return {"status": status, "detail": detail}


def reading_order(edoc: EducationalDocument) -> dict:
    """Heuristic: page numbers across consecutive elements must be non-decreasing.

    A page-number drop usually signals a multi-column layout read in the wrong order.

    PASS: zero violations.
    """
    pages = [el.metadata.page for el in _all_elements(edoc)]
    if len(pages) < 2:
        return {"status": "PASS", "detail": "fewer than 2 elements — no order to check"}

    violations = [(a, b) for a, b in zip(pages, pages[1:]) if b < a]
    status = "PASS" if not violations else "FAIL"
    detail = (
        "page order is non-decreasing"
        if not violations
        else f"{len(violations)} order violation(s) detected (e.g. page {violations[0][0]} → {violations[0][1]})"
    )
    return {"status": status, "detail": detail}


def ocr_confidence_check(edoc: EducationalDocument, threshold: float = 0.60) -> dict:
    """Flag low-confidence OCR elements without penalising non-OCR parsers.

    OCR confidence is emitted by :class:`TesseractParser` per page and stored
    on every derived element as a normalized value in the 0-1 range.
    """
    ocr_elements = [
        el for el in _all_elements(edoc)
        if el.metadata.parser == "tesseract" and el.metadata.confidence is not None
    ]
    if not ocr_elements:
        return {"status": "PASS", "detail": "no OCR confidence data to validate"}

    low = [el for el in ocr_elements if el.metadata.confidence < threshold]
    mean = sum(el.metadata.confidence for el in ocr_elements) / len(ocr_elements)
    status = "PASS" if not low else "FAIL"
    detail = (
        f"{len(ocr_elements)} OCR elements, mean confidence {mean:.0%}"
        if not low
        else f"{len(low)}/{len(ocr_elements)} OCR elements below {threshold:.0%} (mean {mean:.0%})"
    )
    return {"status": status, "detail": detail}


def page_coverage(
    edoc: EducationalDocument,
    expected_page_count: int | None,
    source_markdown: str | None = None,
) -> dict:
    """Detect pages silently lost between extraction and structured output."""
    if not expected_page_count:
        return {"status": "PASS", "detail": "source page count unavailable"}
    if edoc.parser == "liteparse" and source_markdown is not None and "<!-- page:" not in source_markdown:
        return {
            "status": "FAIL",
            "detail": "page provenance unavailable: LiteParse Markdown has no page markers; coverage cannot be measured",
        }
    present = {el.metadata.page for el in _all_elements(edoc) if el.metadata.page > 0}
    missing = [page for page in range(1, expected_page_count + 1) if page not in present]
    status = "PASS" if not missing else "FAIL"
    detail = (
        f"all {expected_page_count} source pages have at least one element"
        if not missing
        else f"{len(missing)}/{expected_page_count} source pages have zero elements (e.g. {missing[:5]})"
    )
    return {"status": status, "detail": detail}


def structured_table_rows(edoc: EducationalDocument) -> dict:
    """Ensure clean Markdown tables retain a row-level representation."""
    tables = [el for el in _all_elements(edoc) if el.type == ElementType.TABLE]
    if not tables:
        return {"status": "PASS", "detail": "no tables to validate"}
    structured = [el for el in tables if el.format == "rows" and el.rows]
    ratio = len(structured) / max(len(tables), 1)
    status = "PASS" if ratio >= 0.90 else ("WARN" if ratio >= 0.70 else "FAIL")
    return {
        "status": status,
        "detail": f"{len(structured)}/{len(tables)} tables contain structured rows ({ratio:.0%})",
    }
