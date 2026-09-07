"""
Unit tests — Educational Parser / rule_based_parser (Test 3).

Verifies the Markdown → EducationalDocument mapping:

  # 1 Introduction        →  Chapter(title="1 Introduction")
  ## 1.1 Motivation       →  Lesson(title="1.1 Motivation") inside "1 Introduction"
  # 2 Benchmark           →  new Chapter(title="2 Benchmark")
  ## 2.1 Dataset          →  Lesson(title="2.1 Dataset") inside "2 Benchmark"

Also verifies:
  - Table rows are captured as TABLE elements
  - Paragraphs are captured as PARAGRAPH elements
  - Element metadata carries parser, chapter, lesson, page
  - Nested headings (##, ###) produce separate Lesson objects
"""
import pytest

from educational.rule_based_parser import parse_markdown_to_education
from schema.models import ElementType, EducationalDocument

# ---------------------------------------------------------------------------
# Shared fixture markdown
# ---------------------------------------------------------------------------

_ACADEMIC_MD = """\
# 1 Introduction

Machine learning is a subset of artificial intelligence focused on building
systems that learn from data.

## 1.1 Motivation

Traditional rule-based systems do not scale to high-dimensional problems.

# 2 Benchmark

## 2.1 Dataset

The dataset consists of 2000 human-verified pages.

| Dimension | Metric | Pages |
| --- | --- | --- |
| Tables | GTRM | 503 |
| Charts | ChartDataPointMatch | 568 |
"""

_PARSER_NAME = "docling"


@pytest.fixture()
def edoc() -> EducationalDocument:
    return parse_markdown_to_education(
        markdown_text=_ACADEMIC_MD,
        source_file="parsebench.pdf",
        parser=_PARSER_NAME,
    )


# ---------------------------------------------------------------------------
# Chapter-level tests
# ---------------------------------------------------------------------------


class TestChapterMapping:
    def test_correct_chapter_count(self, edoc):
        assert len(edoc.chapters) == 2, (
            f"Expected 2 chapters, got {len(edoc.chapters)}: "
            f"{[c.title for c in edoc.chapters]}"
        )

    def test_first_chapter_title(self, edoc):
        assert edoc.chapters[0].title == "1 Introduction"

    def test_second_chapter_title(self, edoc):
        assert edoc.chapters[1].title == "2 Benchmark"


# ---------------------------------------------------------------------------
# Lesson-level tests
# ---------------------------------------------------------------------------


class TestLessonMapping:
    def test_introduction_has_lesson_motivation(self, edoc):
        ch = edoc.chapters[0]
        lesson_titles = [le.title for le in ch.lessons]
        assert "1.1 Motivation" in lesson_titles, (
            f"Expected lesson '1.1 Motivation' in chapter '{ch.title}', "
            f"got: {lesson_titles}"
        )

    def test_benchmark_has_lesson_dataset(self, edoc):
        ch = edoc.chapters[1]
        lesson_titles = [le.title for le in ch.lessons]
        assert "2.1 Dataset" in lesson_titles, (
            f"Expected lesson '2.1 Dataset' in chapter '{ch.title}', "
            f"got: {lesson_titles}"
        )


# ---------------------------------------------------------------------------
# Element-type tests
# ---------------------------------------------------------------------------


class TestElementTypes:
    def test_table_elements_exist(self, edoc):
        tables = [
            el
            for ch in edoc.chapters
            for le in ch.lessons
            for el in le.elements
            if el.type == ElementType.TABLE
        ]
        assert len(tables) >= 1, "Expected at least one TABLE element"

    def test_paragraph_elements_exist(self, edoc):
        paragraphs = [
            el
            for ch in edoc.chapters
            for le in ch.lessons
            for el in le.elements
            if el.type == ElementType.PARAGRAPH
        ]
        assert len(paragraphs) >= 1, "Expected at least one PARAGRAPH element"

    def test_heading_elements_have_level(self, edoc):
        for ch in edoc.chapters:
            for le in ch.lessons:
                for el in le.elements:
                    if el.type == ElementType.HEADING:
                        assert el.level is not None, (
                            f"Heading '{el.text}' is missing a level"
                        )
                        assert el.level >= 1


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestElementMetadata:
    def test_all_elements_have_parser(self, edoc):
        for el, ch_title, le_title in edoc.all_elements():
            assert el.metadata.parser == _PARSER_NAME, (
                f"Element in '{ch_title}/{le_title}' has wrong parser: "
                f"'{el.metadata.parser}'"
            )

    def test_all_elements_have_chapter(self, edoc):
        for el, ch_title, _ in edoc.all_elements():
            assert el.metadata.chapter, f"Element missing chapter metadata"

    def test_all_elements_have_lesson(self, edoc):
        for el, _, le_title in edoc.all_elements():
            assert el.metadata.lesson, f"Element missing lesson metadata"

    def test_all_elements_have_positive_page(self, edoc):
        for el, _, _ in edoc.all_elements():
            assert el.metadata.page > 0, (
                f"Element page must be > 0, got {el.metadata.page}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_markdown_produces_no_chapters(self):
        doc = parse_markdown_to_education("", "empty.pdf", "docling")
        assert doc.chapters == []

    def test_only_paragraph_creates_untitled_chapter(self):
        doc = parse_markdown_to_education(
            "Just a paragraph with no headings.", "plain.pdf", "docling"
        )
        assert len(doc.chapters) == 1
        assert doc.chapters[0].title == "Untitled"

    def test_source_file_stored(self):
        doc = parse_markdown_to_education("# Ch\n\nText.", "myfile.pdf", "docling")
        assert doc.source_file == "myfile.pdf"

    def test_parser_stored_on_document(self):
        doc = parse_markdown_to_education("# Ch\n\nText.", "f.pdf", "llamaparse")
        assert doc.parser == "llamaparse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
