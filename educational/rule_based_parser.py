"""
Educational Parser — Version 1 (rule-based, language-agnostic with OCR heuristics).

Input : clean Markdown from Docling / LlamaParse / OCR (# ## ### + paragraphs + tables).
Output: EducationalDocument (Pydantic) — Chapter -> Lesson -> Element.

Mapping:
    # Title      →  new Chapter
    ## Title     →  new Lesson inside current Chapter
    | … |        →  TABLE element
    Heuristic    →  HEADING element (for un-marked OCR text)
    other text   →  PARAGRAPH element

Built on heading-level layout and heuristic pattern matching.
The same code path handles Arabic, English, and any other language.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

from educational.toc import TocIndex
from schema.models import (
    Chapter,
    EducationalDocument,
    Element,
    ElementMetadata,
    ElementType,
    Lesson,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_RE = re.compile(r"^!\[(?P<alt>.*?)\]\((?P<path>[^)]+)\)\s*$")
IMAGE_PLACEHOLDER_RE = re.compile(r"^<!--\s*image\s*-->$", re.IGNORECASE)
HTML_TABLE_START_RE = re.compile(r"^<table(?:\s|>)", re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"^<!--\s*page\s*:\s*(?P<page>\d+)\s*-->$", re.IGNORECASE)
OCR_CONFIDENCE_RE = re.compile(
    r"^<!--\s*ocr_confidence\s*:\s*(?P<confidence>\d+(?:\.\d+)?)\s*-->$",
    re.IGNORECASE,
)
# Emitted by parsers/ocr_parser.py (OCR_NO_TEXT_MARKER) for pages classified
# as image-only, or where OCR ran but found no words at all. Must be
# recognised structurally rather than falling through to the plain-paragraph
# case below — otherwise the raw HTML comment leaks into the document as
# literal paragraph text, and pages with no usable text produce no element
# at all (silently vanishing from the structured output and from every
# needs_refinement() candidate check).
OCR_NO_TEXT_RE = re.compile(r"^<!--\s*ocr_status\s*:\s*no_text_detected\s*-->$", re.IGNORECASE)
ARABIC_CHAPTER_RE = re.compile(r"^(?:الفصل|الوحدة)\s+[\d\u0660-\u0669\u0600-\u06FF]+")
ARABIC_LESSON_RE = re.compile(r"^(?:الدرس|درس)\s+[\d\u0660-\u0669\u0600-\u06FF]+")

HEADING_PATTERNS = [
    re.compile(r"^(?:الفصل|درس|الدرس|الوحدة|Chapter|Lesson|Unit)\s+[\d\u0600-\u06FF]+", re.IGNORECASE),
    re.compile(r"^(\d+\.\d+)\s+"),  # 1.1, 2.3
    re.compile(r"^(\d+)\s+"),       # 1, 2
    re.compile(r"^[أ-ي]\."),        # أ. ب.
]

# Parsers whose Markdown has no native heading markup (# / ##) and therefore
# need the same OCR-oriented heuristics below: embedded Arabic
# chapter/lesson-banner recovery, spurious leading-number-as-heading
# suppression, and pipe-noise-vs-real-table disambiguation.
#
# This used to be a literal `parser == "tesseract"` check. Tesseract was never
# special here — it was simply the only registered parser that emits flat,
# unstructured OCR text. Every other flat-text OCR candidate
# (candidate_engines/*.py, wrapped by candidate_engines/engine_adapters.py)
# needs exactly the same treatment, or the comparison between engines is
# biased: engines with no native heading markup would silently lose all
# structure while Tesseract keeps it, even when the underlying text quality
# is comparable or better.
#
# Engines that already emit real Markdown headings on their own
# (paddleocr_vl, deepseek_ocr, ain) are intentionally excluded — they go
# through the ordinary HEADING_RE path above like Docling/LlamaParse, and
# forcing OCR heuristics onto already-structured Markdown would only risk
# double-matching or corrupting good headings.
FLAT_OCR_PARSERS = frozenset({
    "tesseract",
    "easyocr",
    "paddleocr",
    "paddleocr_classic",
    "donut",
    "qari_ocr",
    "hybrid_ocr",
    "hybrid_tesseract",
    "hybrid_paddle",
    "gemini_ocr",
})

# Tesseract often merges a coloured textbook banner with nearby body text into
# one long line. Recover the banner as structure, but retain the full OCR line
# as a paragraph so this heuristic can never discard educational content.
_ARABIC_ORDINAL = r"(?:\u0627\u0644\u0623\u0648\u0644|\u0627\u0644\u0627\u0648\u0644|\u0627\u0644\u062b\u0627\u0646\u064a|\u0627\u0644\u062b\u0627\u0646\u0649|\u0627\u0644\u062b\u0627\u0644\u062b|\u0627\u0644\u0631\u0627\u0628\u0639|\u0627\u0644\u062e\u0627\u0645\u0633|\u0627\u0644\u0633\u0627\u062f\u0633|\u0627\u0644\u0633\u0627\u0628\u0639|\u0627\u0644\u062b\u0627\u0645\u0646|\u0627\u0644\u062a\u0627\u0633\u0639|\u0627\u0644\u0639\u0627\u0634\u0631|[0-9\u0660-\u0669]+)"
_OCR_ARABIC_CHAPTER_EMBEDDED_RE = re.compile(
    rf"(?P<title>(?:\u0627\u0644\u0628\u0627\u0628|\u0627\u0644\u0641\u0635\u0644|\u0627\u0644\u0648\u062d\u062f\u0629)\s*{_ARABIC_ORDINAL}(?:\s*[0-9\u0660-\u0669]+)?(?:\s*[-:\u2013\u2014]?\s*[\u0621-\u064A]{2,}){{0,8}})"
)
_OCR_ARABIC_SECTION_EMBEDDED_RE = re.compile(
    rf"(?P<title>(?:\u0627\u0644\u0641\u0635\u0644|\u0627\u0644\u0648\u062d\u062f\u0629)\s*{_ARABIC_ORDINAL}(?:\s*[0-9\u0660-\u0669]+)?(?:\s*[-:\u2013\u2014]?\s*[\u0621-\u064A]{2,}){{0,8}})"
)
_OCR_ARABIC_LESSON_EMBEDDED_RE = re.compile(
    rf"(?P<title>(?:\u0627\u0644\u062f\u0631\u0633|\u062f\u0631\u0633)\s*{_ARABIC_ORDINAL}(?:\s*[0-9\u0660-\u0669]+)?(?:\s*[-:\u2013\u2014]?\s*[\u0621-\u064A]{2,}){{0,8}})"
)
_OCR_TITLE_STOP_RE = re.compile(
    r"\s+(?:\u0641\u064a\s+\u0646\u0647\u0627\u064a\u0629|\u0623\u0647\u062f\u0627\u0641|\u064a\u062a\u0639\u0631\u0641|\u064a\u062a\u0639\u0644\u0645|\u0646\u0634\u0627\u0637|\u062a\u062f\u0631\u064a\u0628|\u0627\u062e\u062a\u0631|\u0639\u0644\u0644|\u0645\u0627\u0630\u0627|\u0623\u0633\u0626\u0644\u0629)\b"
)


def detect_heading(text: str) -> tuple[bool, int]:
    """Detect heading status and level from un-marked text lines.

    Args:
        text: Input line string.

    Returns:
        Tuple of (is_heading, level).
    """
    clean_text = text.strip()
    if not clean_text:
        return False, 0

    if len(clean_text) < 80 and not clean_text.endswith("."):
        if ARABIC_CHAPTER_RE.match(clean_text):
            return True, 1
        if ARABIC_LESSON_RE.match(clean_text):
            return True, 2
        for pattern in HEADING_PATTERNS:
            if pattern.match(clean_text):
                if re.search(r"(فصل|الوحدة|Chapter|Unit)", clean_text, re.IGNORECASE):
                    return True, 1
                return True, 2

    return False, 0


def recover_ocr_arabic_heading(text: str, parser: str, language: str | None) -> tuple[str, int] | None:
    """Recover a chapter or lesson banner embedded in Arabic OCR text.

    Applies to any flat-text OCR parser (see FLAT_OCR_PARSERS), not only
    Tesseract — every one of them can bury a chapter/lesson banner inside a
    long merged OCR line the same way.
    """
    if parser not in FLAT_OCR_PARSERS or language != "ar" or len(text) < 8:
        return None

    # Prefer "chapter"/"unit" over an earlier "part" (باب), because a
    # banner commonly contains both and the chapter is the RAG hierarchy key.
    preferred = _OCR_ARABIC_SECTION_EMBEDDED_RE.search(text)
    if preferred and preferred.start() <= 160:
        candidates: list[tuple[int, re.Match[str]]] = [(1, preferred)]
    else:
        candidates = []
    if not candidates:
        for level, pattern in ((1, _OCR_ARABIC_CHAPTER_EMBEDDED_RE), (2, _OCR_ARABIC_LESSON_EMBEDDED_RE)):
            match = pattern.search(text)
            if match and match.start() <= 160:
                candidates.append((level, match))
    if not candidates:
        return None

    level, match = min(candidates, key=lambda item: item[1].start())
    title = _OCR_TITLE_STOP_RE.split(match.group("title"), maxsplit=1)[0]
    title = re.sub(r"\s+", " ", title).strip(" -:\u2013\u2014")
    return (title, level) if len(title) >= 7 else None


def _parse_markdown_table_rows(lines: list[str]) -> list[list[str]] | None:
    """Parse pipe-table rows while auto-padding slightly ragged OCR rows when a separator exists."""
    rows: list[list[str]] = []
    has_separator = False
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or (len(cells) == 1 and not cells[0]):
            continue
        # Markdown separator row (| --- | :---: |)
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            has_separator = True
            continue
        rows.append(cells)

    if len(rows) < 1:
        return None

    # If has a separator, auto-pad ragged rows to the maximum column count
    if has_separator:
        max_cols = max(len(r) for r in rows)
        if max_cols >= 2:
            return [r + [""] * (max_cols - len(r)) for r in rows]

    # If no separator, require exact uniform columns and at least 2 rows
    if len(rows) >= 2 and len({len(row) for row in rows}) == 1 and len(rows[0]) >= 2:
        return rows

    return None


def _has_markdown_table_separator(lines: list[str]) -> bool:
    """A separator row distinguishes real Markdown from OCR pipe noise."""
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            return True
    return False


class _HTMLTableReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.headers: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._header_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._header_cell = tag == "th"

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            value = " ".join("".join(self._cell).split())
            self._row.append(value)
            if self._header_cell:
                self.headers.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _parse_html_table(html: str) -> tuple[list[list[str]] | None, list[str]]:
    reader = _HTMLTableReader()
    reader.feed(html)
    rows = reader.rows or None
    return rows, reader.headers


def _extract_document_metadata(markdown_text: str, source_file: str) -> dict[str, str | None]:
    """Extract document-level educational metadata (title, subject, grade, term) from text and filename."""
    from pathlib import Path
    stem = Path(source_file).stem
    sample = markdown_text[:30000]

    # 1. Subject extraction
    subject = None
    m_subj = re.search(r"(?:^|\n)(?:Subject|المادة|مادة|Discipline)\s*[:=\-–—]\s*([^\n]{2,80})", sample, re.I)
    if m_subj:
        subject = m_subj.group(1).strip()
    else:
        m_intro = re.search(
            r"(?:Introduction to|An Introduction to|مقدمة في)\s+([A-Za-z\u0600-\u06FF\s&/\-]{3,80}?)(?=\s+(?:for|للصف|grade|secondary|term)|\n|$)",
            sample,
            re.I,
        )
        if m_intro:
            subject = m_intro.group(1).strip()
        else:
            parts = [p for p in re.split(r"[_\-\s]+", stem) if p]
            for p in parts:
                low = p.casefold()
                if low in {"ar", "ara", "arabic", "en", "eng", "english", "pdf"}:
                    continue
                if re.match(r"^(?:sec|secondary|grade|class|prim|prep|t|term|sem)\d*$", low):
                    continue
                if len(p) >= 2 and not p.isdigit():
                    subject = p
                    break

    # 2. Grade extraction
    grade = None
    m_grade = re.search(r"(?:^|\n)(?:Grade|Class|الصف|للصف|المرحلة)\s*[:=\-–—]?\s*([^\n]{1,40})", sample, re.I)
    if m_grade:
        grade = m_grade.group(1).strip()
    else:
        m_stage = re.search(
            r"(?:For|for)?\s*(?:the\s+)?(First|Second|Third|Fourth|Fifth|Sixth|1st|2nd|3rd|[1-9])\s+Year\s+(?:of\s+)?(Secondary|Primary|Preparatory|Prep)(?:\s+School)?",
            sample,
            re.I,
        )
        if m_stage:
            grade = f"{m_stage.group(1)} Year {m_stage.group(2)}"
        else:
            m_sec = re.search(r"(?:Sec|Secondary|Grade|Prep|Prim)(\d{1,2})", stem, re.I)
            if m_sec:
                grade = f"Secondary {m_sec.group(1)}" if "sec" in stem.lower() else f"Grade {m_sec.group(1)}"

    # 3. Term extraction
    term = None
    m_term = re.search(r"(?:Term|Semester|الترم|الفصل الدراسي)\s*[:=\-–—]?\s*(First|Second|One|Two|1|2|الأول|الثاني)", sample, re.I)
    if m_term:
        term = m_term.group(1).strip()
    else:
        m_t = re.search(r"[_\- ](?:T|Term|Sem)(\d{1,2})", stem, re.I)
        if m_t:
            term = f"Term {m_t.group(1)}"

    # 4. Title
    if subject and grade:
        title = f"{subject} — {grade}"
    elif subject:
        title = subject
    else:
        title = stem.replace("_", " ")

    return {
        "document_title": title,
        "subject": subject,
        "grade": grade,
        "term": term,
    }


def parse_markdown_to_education(
    markdown_text: str,
    source_file: str,
    parser: str,
    language: str | None = None,
) -> EducationalDocument:
    """Convert a Markdown string into a structured EducationalDocument.

    Args:
        markdown_text: Markdown output from any BaseParser implementation.
        source_file:   Original file path (stored as provenance).
        parser:        Name of the parser that produced the Markdown.
        language:      ISO 639-1 language code when known (e.g. 'en', 'ar').
                       Pass None to remain language-agnostic.

    Returns:
        EducationalDocument ready for chunking or direct RAG ingestion.
    """
    meta_info = _extract_document_metadata(markdown_text, source_file)
    doc = EducationalDocument(
        source_file=source_file,
        language=language,
        parser=parser,
        document_title=meta_info["document_title"],
        subject=meta_info["subject"],
        grade=meta_info["grade"],
        term=meta_info["term"],
    )

    # ── TOC Index ─────────────────────────────────────────────────────────────
    # For flat-text OCR parsers (OCR.space, Tesseract, and similar engines) the body
    # text contains no native Markdown heading markers.  The textbook's own
    # Table of Contents (الفهرس) is the authoritative source for the chapter /
    # lesson hierarchy.  Build an index once up-front; on every page boundary
    # we look up which chapter that page belongs to and update current_chapter
    # without waiting for a heading keyword to appear in the body text.
    toc_index: TocIndex | None = None
    if parser in FLAT_OCR_PARSERS:
        toc_index = TocIndex.from_markdown(markdown_text)

    current_chapter: Chapter | None = None
    current_lesson: Lesson | None = None
    page = 1
    current_confidence: float | None = None

    def ensure_default_chapter() -> None:
        nonlocal current_chapter
        if current_chapter is None:
            current_chapter = Chapter(title="Untitled")
            doc.chapters.append(current_chapter)

    def ensure_default_lesson() -> None:
        nonlocal current_lesson
        ensure_default_chapter()
        if current_lesson is None:
            current_lesson = Lesson(title=current_chapter.title)
            current_chapter.lessons.append(current_lesson)

    def make_meta(chapter_title: str, lesson_title: str) -> ElementMetadata:
        extra = {}
        if current_confidence is not None:
            extra["needs_review"] = current_confidence < 0.60
        return ElementMetadata(
            page=page,
            chapter=chapter_title,
            lesson=lesson_title,
            parser=parser,
            confidence=current_confidence,
            extra=extra,
        )

    lines = markdown_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        page_match = PAGE_MARKER_RE.match(line)
        if page_match:
            page = int(page_match.group("page"))
            current_confidence = None
            # ── TOC-driven chapter advancement ────────────────────────────
            # When a TOC was found, authoritative chapter boundaries come from
            # the TOC page-range lookup.  Advance current_chapter whenever we
            # cross into a new chapter's page range.  This replaces the old
            # keyword-scanning approach for flat-text OCR parsers.
            if toc_index is not None:
                toc_ch = toc_index.get_chapter_for_page(page)
                if toc_ch and (
                    current_chapter is None
                    or current_chapter.title != toc_ch
                ):
                    current_chapter = Chapter(title=toc_ch)
                    doc.chapters.append(current_chapter)
                    current_lesson = None  # reset lesson within new chapter
            i += 1
            continue

        confidence_match = OCR_CONFIDENCE_RE.match(line)
        if confidence_match:
            raw = float(confidence_match.group("confidence"))
            current_confidence = max(0.0, min(raw / 100.0, 1.0))
            i += 1
            continue

        # The contents page lists every future chapter.  It is front matter,
        # not a sequence of real chapter boundaries.
        in_toc_front_matter = bool(
            toc_index is not None
            and toc_index.first_chapter_page is not None
            and page < toc_index.first_chapter_page
        )

        if OCR_NO_TEXT_RE.match(line):
            ensure_default_lesson()
            meta = make_meta(current_chapter.title, current_lesson.title)
            current_lesson.elements.append(
                Element(
                    type=ElementType.PARAGRAPH,
                    text="",
                    metadata=meta.model_copy(
                        update={"extra": {**meta.extra, "needs_review": True, "ocr_status": "no_text_detected"}}
                    ),
                )
            )
            i += 1
            continue

        image_match = IMAGE_RE.match(line)
        if image_match:
            ensure_default_lesson()
            alt_text = image_match.group("alt").strip() or "Image"
            current_lesson.elements.append(
                Element(
                    type=ElementType.IMAGE,
                    text=f"[Image: {alt_text}]",
                    metadata=make_meta(current_chapter.title, current_lesson.title).model_copy(
                        update={
                            "extra": {
                                **make_meta(current_chapter.title, current_lesson.title).extra,
                                "image_path": image_match.group("path"),
                                "alt_text": alt_text,
                            }
                        }
                    ),
                )
            )
            i += 1
            continue

        if IMAGE_PLACEHOLDER_RE.match(line):
            ensure_default_lesson()
            current_lesson.elements.append(
                Element(
                    type=ElementType.IMAGE,
                    text="[Image: Docling placeholder]",
                    metadata=make_meta(current_chapter.title, current_lesson.title).model_copy(
                        update={"extra": {"association": "docling_placeholder"}}
                    ),
                )
            )
            i += 1
            continue

        if HTML_TABLE_START_RE.match(line):
            table_lines = [line]
            j = i + 1
            while j < len(lines):
                table_lines.append(lines[j])
                if "</table>" in lines[j].lower():
                    j += 1
                    break
                j += 1
            raw_html = "\n".join(table_lines)
            rows, headers = _parse_html_table(raw_html)
            ensure_default_lesson()
            current_lesson.elements.append(
                Element(
                    type=ElementType.TABLE,
                    text=raw_html,
                    format="html",
                    rows=rows,
                    metadata=make_meta(current_chapter.title, current_lesson.title).model_copy(
                        update={"extra": {"headers": headers, "rows": rows or []}}
                    ),
                )
            )
            i = j
            continue

        # --- Markdown Heading (# ## ###) ---
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # ── TOC-aware level resolution ─────────────────────────────────
            # If the TOC index is available, consult it to confirm whether
            # this heading is a chapter (level 1) or a lesson (level 2).
            # This overrides the raw Markdown # count:
            # 1. If it matches a TOC entry, use the TOC's authoritative level and title.
            # 2. If it does NOT match any TOC chapter, demote level 1 to level 2 (lesson),
            #    because the TOC is the sole authority on what constitutes a chapter.
            if toc_index is not None:
                toc_entry = toc_index.match_heading(title, page)
                if toc_entry is not None:
                    level = toc_entry.level
                    title = toc_entry.title
                elif level == 1:
                    # Heading claims to be a chapter (#), but the TOC has no record of it.
                    # Treat it as a lesson inside the currently active TOC chapter.
                    level = 2

            if level == 1:
                # When a TOC is present, avoid duplicating a chapter that the
                # page-range advancement already created (body banner echoing
                # the chapter title).  Without a TOC (e.g. docling), every
                # explicit # heading must always create a new chapter node.
                if toc_index is None or current_chapter is None or current_chapter.title != title:
                    current_chapter = Chapter(title=title)
                    doc.chapters.append(current_chapter)
                    current_lesson = None
            else:
                ensure_default_chapter()
                current_lesson = Lesson(title=title)
                current_chapter.lessons.append(current_lesson)

            ensure_default_lesson()
            current_lesson.elements.append(
                Element(
                    type=ElementType.HEADING,
                    text=title,
                    level=level,
                    metadata=make_meta(current_chapter.title, current_lesson.title),
                )
            )
            i += 1
            continue

        # --- Arabic heading embedded in a long OCR line ---
        recovered_heading = None if in_toc_front_matter else recover_ocr_arabic_heading(line, parser, language)
        if recovered_heading:
            title, level = recovered_heading
            # ── TOC-aware level resolution ──────────────────────────────────
            if toc_index is not None:
                toc_entry = toc_index.match_heading(title, page)
                if toc_entry is not None:
                    level = toc_entry.level
                    title = toc_entry.title
                    # If TOC already advanced current_chapter to this title,
                    # don't create a duplicate chapter node.
                    if level == 1 and current_chapter and current_chapter.title == title:
                        # Just record a HEADING element; skip chapter creation.
                        ensure_default_lesson()
                        current_lesson.elements.append(
                            Element(
                                type=ElementType.HEADING,
                                text=title,
                                level=level,
                                metadata=make_meta(current_chapter.title, current_lesson.title),
                            )
                        )
                        # Continue into paragraph handling for the rest of the OCR line
                        recovered_heading = (title, level)
                    else:
                        recovered_heading = (title, level)
                        if level == 1:
                            if current_chapter is None or current_chapter.title != title:
                                current_chapter = Chapter(title=title)
                                doc.chapters.append(current_chapter)
                                current_lesson = None
                        else:
                            ensure_default_chapter()
                            current_lesson = Lesson(title=title)
                            current_chapter.lessons.append(current_lesson)
                else:
                    # No TOC match — when TOC exists, treat as a lesson
                    # (chapter boundaries are handled by page-range lookup).
                    if toc_index is not None and level == 1:
                        level = 2
                    if level == 1:
                        if current_chapter is None or current_chapter.title != title:
                            current_chapter = Chapter(title=title)
                            doc.chapters.append(current_chapter)
                            current_lesson = None
                    else:
                        ensure_default_chapter()
                        current_lesson = Lesson(title=title)
                        current_chapter.lessons.append(current_lesson)
            else:
                # No TOC — original behaviour
                if level == 1:
                    current_chapter = Chapter(title=title)
                    doc.chapters.append(current_chapter)
                    current_lesson = None
                else:
                    ensure_default_chapter()
                    current_lesson = Lesson(title=title)
                    current_chapter.lessons.append(current_lesson)

            ensure_default_lesson()
            current_lesson.elements.append(
                Element(
                    type=ElementType.HEADING,
                    text=title,
                    level=level,
                    metadata=make_meta(current_chapter.title, current_lesson.title),
                )
            )
            # Continue into paragraph handling below: objective/body text that
            # shares the OCR line must remain retrievable.

        # --- Heuristic Heading (for un-marked OCR / plain text) ---
        is_heuristic_heading, level = (False, 0) if recovered_heading else detect_heading(line)
        # OCR body text often begins with a stray page/question number.  Bare
        # numeric-prefix headings are useful for clean digital Markdown, but
        # create false lessons such as "2 9 ..." in scanned books.
        if parser in FLAT_OCR_PARSERS and re.match(r"^[0-9\u0660-\u0669]", line):
            is_heuristic_heading, level = False, 0
        # Author names, contributors, and academic titles on credit/cover pages
        # (e.g. Dr., Prof., Mr., Eng., أ.د., د., أ., م.) frequently match letter/abbreviation patterns
        # and must never become chapter or lesson headings across all languages (Arabic & English).
        if (
            re.match(r"^(?:[أاإآ]\s*\.\s*(?:د\s*\.)?|د\s*\.|م\s*\.)\s*[\u0600-\u06FF]", line)
            or re.match(r"^(?:Prof\.|Professor|Dr\.|Doctor|Mr\.|Mrs\.|Ms\.|Eng\.)\s+[A-Z\u0600-\u06FF]", line, re.IGNORECASE)
            or re.match(r"^(?:إعداد|مراجعة|تأليف|إشراف|تصميم|Prepared by|Edited by|Written by|Supervised by)\b", line, re.IGNORECASE)
        ):
            is_heuristic_heading, level = False, 0
        # When a TOC is present, chapter boundaries are already handled by the
        # page-range lookup.  Suppress heuristic chapter creation (level 1) to
        # prevent spurious chapters from stray "الفصل" mentions in body text.
        if toc_index is not None and is_heuristic_heading and level == 1:
            is_heuristic_heading, level = False, 0
        if is_heuristic_heading:
            if level == 1:
                current_chapter = Chapter(title=line)
                doc.chapters.append(current_chapter)
                current_lesson = None
            else:
                ensure_default_chapter()
                current_lesson = Lesson(title=line)
                current_chapter.lessons.append(current_lesson)

            ensure_default_lesson()
            current_lesson.elements.append(
                Element(
                    type=ElementType.HEADING,
                    text=line,
                    level=level,
                    metadata=make_meta(current_chapter.title, current_lesson.title),
                )
            )
            i += 1
            continue

        # --- Markdown table (lines starting with |) ---
        if line.startswith("|"):
            table_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            ensure_default_lesson()
            parsed_rows = _parse_markdown_table_rows(table_lines)
            # OCR noise can contain stray pipe characters.  Unlike Docling
            # Markdown, it is not evidence of a table unless rows are valid.
            # Applies to any flat-text OCR parser, not only Tesseract literally.
            if parser in FLAT_OCR_PARSERS and (parsed_rows is None or not _has_markdown_table_separator(table_lines)):
                current_lesson.elements.append(
                    Element(
                        type=ElementType.PARAGRAPH,
                        text="\n".join(table_lines),
                        metadata=make_meta(current_chapter.title, current_lesson.title),
                    )
                )
                i = j
                continue
            current_lesson.elements.append(
                Element(
                    type=ElementType.TABLE,
                    text="\n".join(table_lines),
                    format="rows" if parsed_rows else "markdown",
                    rows=parsed_rows,
                    metadata=make_meta(current_chapter.title, current_lesson.title),
                )
            )
            i = j
            continue

        # --- Plain paragraph ---
        ensure_default_lesson()
        current_lesson.elements.append(
            Element(
                type=ElementType.PARAGRAPH,
                text=line,
                metadata=make_meta(current_chapter.title, current_lesson.title),
            )
        )
        i += 1

    return doc
