"""
edoc.json (EducationalDocument) -> Markdown, shaped for EDU-RAG's document loader.

Why this exists
----------------
EDU-RAG's ingestion pipeline is:

    document_loader.load_file(path)   -> list[LoadedDocument(text, metadata)]
    chunker.split(loaded.text, base)  -> parent/child TextChunks

For a .json file, load_file() calls `_render_structured_json()`, a *generic*
key/value walker (looks for "content"/"text"/"body"/"description" keys, and a
fixed set of flat metadata keys). It has no idea edu22's schema even exists,
so pointing it at edu2's own `Element`-per-node edoc.json (type/text/metadata
nested per element) makes it emit one LoadedDocument *per element* (one
sentence each), with a useless "chapters\\nItem 1\\nlessons\\nItem 1..." path
prefix glued onto the text, and none of subject/grade/chapter/lesson surviving
(they live one level too deep, inside `element.metadata`, which the generic
walker never descends into for metadata purposes).

For a .md/.txt file, by contrast, load_file() just reads the whole file as
one block of text and calls extract_hierarchy() on it, which is the *same*
heading-detection code (`detect_heading` / `update_structure` in
application/metadata.py) the chunker itself uses while splitting. A markdown
heading (`#`, `##`, ...) is a 100%-confidence, "positional eligible" match
there, mapped directly to chapter/lesson/section/topic by heading depth
(1->chapter, 2->lesson, 3->section, 4->topic).

So: rather than reshaping edu22's JSON to fit the generic JSON walker (which
structurally can never avoid at least one noise line per element — see
document_loader._render_structured_json), this renders the *already correct*
Chapter -> Lesson -> Element hierarchy edu22 produced straight to Markdown.
That is a format EDU-RAG's loader (and its chunker, using the exact same
heading grammar) was actually built to consume, with zero changes required
on the EDU-RAG side.

This intentionally does NOT try to fix upstream chapter/lesson mis-titling
("Untitled", a repeated running-header being mistaken for a new lesson on
every page, etc.) -- that is a separate parsing-quality problem. Whatever
title is in the edoc, right or wrong, is emitted as the heading verbatim.
"""
from __future__ import annotations

import re
from typing import Any

# EDU-RAG's ChunkMetadata (src/features/rag/domain/schemas.py) caps
# heading text length; application/metadata.py's own detect_heading() default
# max_length is 180. Keep headings comfortably under that so a long running
# page-header line doesn't silently fail heading detection and fall back to
# being ordinary paragraph text.
_MAX_HEADING_LEN = 150


def _clean_line(text: str | None) -> str:
    """Collapse internal whitespace/newlines; a heading or paragraph line must
    be single-line or update_structure() will only ever see its first line."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _heading_line(level: int, title: str) -> str | None:
    title = _clean_line(title)
    if not title:
        return None
    if len(title) > _MAX_HEADING_LEN:
        # Still emit it, but as a paragraph rather than a heading: a
        # heading this long is almost always mis-OCR'd running text, not a
        # real title, and detect_heading() would reject it anyway (silently
        # falling through to paragraph handling one call later). Doing it
        # here keeps the mapping predictable instead of relying on that
        # fallback.
        return title
    level = max(1, min(level, 6))
    return f"{'#' * level} {title}"


def _rows_to_markdown_table(rows: list[list[Any]]) -> str | None:
    rows = [r for r in (rows or []) if r]
    if not rows:
        return None
    width = max(len(r) for r in rows)

    def pad(row: list[Any]) -> list[str]:
        cells = [str(c).strip().replace("|", "\\|").replace("\n", " ") for c in row]
        cells += [""] * (width - len(cells))
        return cells

    header = pad(rows[0])
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(pad(row)) + " |")
    return "\n".join(lines)


def _element_to_markdown(element: dict[str, Any]) -> str | None:
    """One edoc Element -> zero or more Markdown lines (still one "paragraph"
    block: callers join blocks with a *blank* line, never inside this)."""
    etype = element.get("type")
    text = _clean_line(element.get("text"))

    if etype == "heading":
        level = element.get("level") or 3
        # Chapter/lesson titles are rendered separately from the Chapter/Lesson
        # objects themselves (see educational_document_to_markdown). A heading
        # *element* found inside a lesson's element list is therefore always a
        # sub-heading of that lesson, hence level 1/2 -> level 3 floor so it
        # never collides with (and gets merged into) the surrounding chapter
        # or lesson.
        return _heading_line(max(level, 1) + 2, text)

    if etype == "table":
        rows = element.get("rows")
        if rows:
            table_md = _rows_to_markdown_table(rows)
            if table_md:
                return table_md
        if element.get("format") == "markdown" and text:
            return text
        return text or None

    if etype == "image":
        # Preserved for traceability, deliberately styled so it can never be
        # mistaken for a heading or for real prose content (classify_content_type
        # in application/metadata.py has no image-specific bucket, so an
        # un-marked image caption would otherwise just read as a stray
        # paragraph).
        alt = text or (element.get("metadata", {}).get("extra", {}) or {}).get("image_path", "")
        alt = _clean_line(alt)
        return f"*[صورة: {alt}]*" if alt else "*[صورة]*"

    if etype == "list":
        if element.get("rows"):
            items = [_clean_line(" ".join(str(c) for c in row)) for row in element["rows"] if row]
            items = [f"- {item}" for item in items if item]
            return "\n".join(items) if items else None
        return text or None

    # paragraph / example / exercise / definition / equation / anything else:
    # emitted as plain text. Deliberately no synthetic "مثال:"/"تمرين:" prefix
    # -- classify_content_type() downstream keys off the *actual* wording
    # (e.g. "مثال", "تمرين", "تعريف") already present in real curriculum text;
    # inventing a label the source text doesn't contain would just duplicate it.
    return text or None


def _lesson_to_markdown(lesson: dict[str, Any], lesson_level: int) -> str:
    blocks: list[str] = []
    heading = _heading_line(lesson_level, lesson.get("title"))
    if heading:
        blocks.append(heading)
    for element in lesson.get("elements", []):
        block = _element_to_markdown(element)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def _chapter_to_markdown(chapter: dict[str, Any]) -> str:
    blocks: list[str] = []
    heading = _heading_line(1, chapter.get("title"))
    if heading:
        blocks.append(heading)
    lessons = chapter.get("lessons", [])
    for lesson in lessons:
        lesson_md = _lesson_to_markdown(lesson, lesson_level=2)
        if lesson_md:
            blocks.append(lesson_md)
    return "\n\n".join(blocks)


def educational_document_to_markdown(edoc: dict[str, Any]) -> str:
    """Render a full edoc.json dict (as produced by EducationalDocument.model_dump())
    into Markdown suitable to hand straight to EDU-RAG as a .md file -- i.e. the
    text EDU-RAG's chunker (`HierarchyAwareParentChildChunker.split`) will see,
    matching its input contract exactly: blank-line-separated paragraphs, and
    "#"-style headings it can detect on its own.

    Document-level fields that live outside the chapter/lesson tree
    (document_title/subject/grade/term/language) are intentionally NOT printed
    as `#`-level headings here -- a level-1 heading is reserved for a real
    Chapter, and a cover-page title would otherwise get mistaken for chapter 1
    (see the `_BOILERPLATE_LINE`-style problems this exact thing causes,
    documented in EDU-RAG's application/document_metadata.py). Pass them
    instead as `extra_metadata={"subject": ..., "grade": ...}` /
    `curriculum_id=...` to IngestionService.ingest() for this file -- edu22
    already extracted them once with an LLM; there's no reason to make
    EDU-RAG's regex-based cover-page sniffer re-derive them from scratch.
    """
    chapters = edoc.get("chapters", [])
    blocks = [_chapter_to_markdown(ch) for ch in chapters]
    return "\n\n".join(b for b in blocks if b).strip() + "\n"


def rag_extra_metadata(edoc: dict[str, Any]) -> dict[str, Any]:
    """Document-level fields to pass as `extra_metadata=` / `curriculum_id=`
    to IngestionService.ingest() alongside the .md file this module produces,
    instead of re-derived by EDU-RAG's own cover-page heuristics."""
    return {
        "subject": edoc.get("subject"),
        "grade": edoc.get("grade"),
        "language": edoc.get("language"),
        "document_title": edoc.get("document_title"),
    }
