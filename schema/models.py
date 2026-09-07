"""
Educational document schema — Pydantic models replacing ad-hoc dictionaries.

Guarantees that every parser (Docling / LlamaParse / OCR) emits exactly the
same structure so downstream RAG / Tutor layers depend on a stable contract.

Upgrade notes:
  - 'source_engine' renamed to 'parser' (shorter, clearer)
  - Added 'confidence' (OCR / VLM confidence score, 0-1)
  - Added 'bbox' ([x0, y0, x1, y1] when the parser provides coordinates)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    DEFINITION = "definition"
    LIST = "list"
    EQUATION = "equation"


class ElementMetadata(BaseModel):
    page: int
    chapter: Optional[str] = None
    lesson: Optional[str] = None
    parser: str = Field(
        description="Which parser produced this element: docling / llamaparse / ocr"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Parser confidence score (0.0-1.0). Populated by OCR / VLM backends.",
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [x0, y0, x1, y1] in page-coordinate space, when available.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser-specific metadata such as image_path or association quality.",
    )


class Element(BaseModel):
    type: ElementType
    text: Optional[str] = None
    level: Optional[int] = Field(
        default=None,
        description="Heading depth when type=heading (1=chapter, 2=lesson, 3=sub-lesson …)",
    )
    format: Optional[str] = Field(
        default=None,
        description="Structured representation format, e.g. markdown, html, or rows for tables.",
    )
    rows: Optional[list[list[str]]] = Field(
        default=None,
        description="Parsed table cells when the source table can be read without loss.",
    )
    metadata: ElementMetadata


class Lesson(BaseModel):
    title: str
    elements: list[Element] = Field(default_factory=list)


class Chapter(BaseModel):
    title: str
    lessons: list[Lesson] = Field(default_factory=list)


class EducationalDocument(BaseModel):
    """Final output of the Educational Parser — consumed by Chunking or RAG directly."""

    source_file: str
    language: Optional[str] = None
    parser: str
    document_title: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None
    term: Optional[str | int] = None
    chapters: list[Chapter] = Field(default_factory=list)

    def all_elements(self) -> list[tuple[Element, str, str]]:
        """Flat iterator: (element, chapter_title, lesson_title)."""
        out = []
        for ch in self.chapters:
            for lesson in ch.lessons:
                for el in lesson.elements:
                    out.append((el, ch.title, lesson.title))
        return out
