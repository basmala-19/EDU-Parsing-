"""
Chunking over structured EducationalDocument — matching EDU-RAG-integration-version schema.

Each chunk carries:
  - raw_text: exact chunk text content (required for RAG grounding)
  - normalized_text: NFKC normalized text for semantic vector matching
  - chunk_id: deterministic 16-char sha1 identifier
  - type / content_type: classified content type (paragraph, table, heading, image, definition, exercise)
  - chapter, lesson, page, parser: standard direct access metadata
  - metadata: full nested ChunkMetadata dict conforming to EDU-RAG-integration-version
"""
from __future__ import annotations

import re
import unicodedata
from hashlib import sha1
from typing import Any
from schema.models import EducationalDocument, ElementType


def normalize_text(text: str) -> str:
    """Normalize text with NFKC and strip non-printable bidi/control chars."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[\u200e\u200f\u202a\u202b\u202c\u2066\u2067\u2069\ufeff]", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def classify_content_type(text: str, element_type: str | None = None) -> str:
    """Classify chunk content type matching EDU-RAG standard."""
    if element_type in {"table", "heading", "image"}:
        return element_type
    t = (text or "").strip()
    if re.search(r"(?:اختر|أجب|أكمل|املأ|تمرين|تحدى معلوماتك|جرب بنفسك|exercise|question|fill in|choose|answer)", t, flags=re.I):
        return "exercise"
    if re.search(r"(?:مثال|example)", t, flags=re.I):
        return "example"
    if re.search(r"(?:تعريف|definition|يعرف|تعني|يسمى)", t, flags=re.I):
        return "definition"
    if "|" in t and t.count("|") >= 2:
        return "table"
    return "paragraph"


def generate_chunk_id(source: str, page: int | None, heading_path: list[str], raw_text: str) -> str:
    """Generate deterministic 16-char sha1 chunk identifier."""
    basis = f"{source}|{page or 1}|{'>'.join(heading_path)}|{raw_text.strip()}"
    return sha1(basis.encode("utf-8")).hexdigest()[:16]


def chunk_educational_document(
    doc: EducationalDocument, max_chars: int = 1200
) -> list[dict[str, Any]]:
    """Split an EducationalDocument into retrieval-ready chunks matching EDU-RAG schema.

    Args:
        doc:       Structured document produced by Educational Parser.
        max_chars: Soft character limit per paragraph chunk.

    Returns:
        List of dicts, each with raw_text, text, normalized_text, chunk_id, type,
        chapter, lesson, page, parser, and full nested metadata dict.
    """
    chunks: list[dict[str, Any]] = []
    source_file = doc.source_file or "document.pdf"
    doc_lang = doc.language or "ar"
    curriculum_id = f"cur_{sha1(source_file.encode('utf-8')).hexdigest()[:12]}"

    for chapter_title, lesson_title, elements in _iter_lessons(doc):
        buffer: list[str] = []
        buffer_len = 0
        buffer_page: int | None = None
        heading_path = [chapter_title, lesson_title]

        def flush() -> None:
            if buffer:
                raw = "\n".join(buffer).strip()
                if not raw:
                    return
                norm = normalize_text(raw)
                page_num = buffer_page or 1
                parser_name = elements[0].metadata.parser if elements else doc.parser
                ctype = classify_content_type(raw, "paragraph")
                cid = generate_chunk_id(source_file, page_num, heading_path, raw)

                meta: dict[str, Any] = {
                    "curriculum_id": curriculum_id,
                    "version": "v1",
                    "subject": doc.subject,
                    "grade": doc.grade,
                    "term": doc.term,
                    "language": doc_lang,
                    "source": source_file,
                    "source_type": "pdf",
                    "page": page_num,
                    "parser": parser_name,
                    "heading": lesson_title,
                    "heading_level": 2,
                    "heading_path": heading_path,
                    "chapter": chapter_title,
                    "lesson": lesson_title,
                    "section": None,
                    "topic": None,
                    "content_type": ctype,
                    "parent_chunk_id": None,
                    "chunk_role": "child",
                }

                chunks.append(
                    {
                        "chunk_id": cid,
                        "raw_text": raw,
                        "text": raw,  # backward compatibility
                        "normalized_text": norm,
                        "type": "paragraph",
                        "content_type": ctype,
                        "chapter": chapter_title,
                        "lesson": lesson_title,
                        "page": page_num,
                        "parser": parser_name,
                        "metadata": meta,
                    }
                )

        for el in elements:
            piece = (el.text or "").strip()
            if not piece:
                continue

            # Tables, headings, and figures are always standalone chunks.
            if el.type in (ElementType.TABLE, ElementType.HEADING, ElementType.IMAGE):
                flush()
                buffer, buffer_len, buffer_page = [], 0, None

                raw = piece
                norm = normalize_text(raw)
                page_num = el.metadata.page or 1
                parser_name = el.metadata.parser or doc.parser
                ctype = el.type.value
                cid = generate_chunk_id(source_file, page_num, heading_path, raw)

                meta: dict[str, Any] = {
                    "curriculum_id": curriculum_id,
                    "version": "v1",
                    "subject": doc.subject,
                    "grade": doc.grade,
                    "term": doc.term,
                    "language": doc_lang,
                    "source": source_file,
                    "source_type": "pdf",
                    "page": page_num,
                    "parser": parser_name,
                    "heading": piece if el.type == ElementType.HEADING else lesson_title,
                    "heading_level": el.level or (1 if el.type == ElementType.HEADING else 2),
                    "heading_path": heading_path + ([piece] if el.type == ElementType.HEADING else []),
                    "chapter": chapter_title,
                    "lesson": lesson_title,
                    "section": None,
                    "topic": None,
                    "content_type": ctype,
                    "parent_chunk_id": None,
                    "chunk_role": "child",
                }

                chunk_dict: dict[str, Any] = {
                    "chunk_id": cid,
                    "raw_text": raw,
                    "text": raw,  # backward compatibility
                    "normalized_text": norm,
                    "type": ctype,
                    "content_type": ctype,
                    "chapter": chapter_title,
                    "lesson": lesson_title,
                    "page": page_num,
                    "parser": parser_name,
                    "metadata": meta,
                }
                if el.type == ElementType.IMAGE:
                    chunk_dict["image_path"] = el.metadata.extra.get("image_path")
                    chunk_dict["alt_text"] = el.metadata.extra.get("alt_text")
                    meta["image_path"] = el.metadata.extra.get("image_path")

                chunks.append(chunk_dict)
                continue

            if buffer_len + len(piece) > max_chars:
                flush()
                buffer, buffer_len, buffer_page = [], 0, None

            if buffer_page is None:
                buffer_page = el.metadata.page
            buffer.append(piece)
            buffer_len += len(piece)

        flush()

    return chunks


def _iter_lessons(doc: EducationalDocument):
    """Yield (chapter_title, lesson_title, element_list) for every lesson."""
    for chapter in doc.chapters:
        for lesson in chapter.lessons:
            yield chapter.title, lesson.title, lesson.elements

