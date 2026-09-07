"""
Unit tests — Language-Agnostic behaviour (Test 5).

Verifies that the Educational Parser processes Arabic text correctly
using the exact same code path as English — no language-specific branches.

Key principle: the pipeline accepts language=None (language-agnostic mode)
unless a language detector explicitly identifies the language. Hardcoding
language="ar" is NOT done here.
"""
import pytest

from educational.rule_based_parser import parse_markdown_to_education
from chunking.chunker import chunk_educational_document
from schema.models import ElementType

# ---------------------------------------------------------------------------
# Arabic fixture markdown
# ---------------------------------------------------------------------------

_ARABIC_MD = """\
# الفصل الأول: مقدمة

التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يهتم ببناء أنظمة تتعلم من البيانات.

## الدافع

الأنظمة القائمة على القواعد التقليدية لا تصلح للمشكلات المعقدة عالية الأبعاد.

# الفصل الثاني: المعيار

## مجموعة البيانات

تتكون مجموعة البيانات من 2000 صفحة تم التحقق منها بشرياً.

| البُعد       | المقياس | الصفحات |
| ------------ | ------- | ------- |
| الجداول      | GTRM    | 503     |
| المحتوى      | درجة    | 506     |
"""


@pytest.fixture()
def arabic_edoc():
    # language=None: language-agnostic mode
    return parse_markdown_to_education(
        markdown_text=_ARABIC_MD,
        source_file="arabic_book.pdf",
        parser="docling",
        language=None,
    )


@pytest.fixture()
def arabic_chunks(arabic_edoc):
    return chunk_educational_document(arabic_edoc)


# ---------------------------------------------------------------------------
# Structure tests — same assertions as English
# ---------------------------------------------------------------------------


class TestArabicStructure:
    def test_correct_chapter_count(self, arabic_edoc):
        assert len(arabic_edoc.chapters) == 2, (
            f"Expected 2 chapters, got {len(arabic_edoc.chapters)}"
        )

    def test_first_chapter_title_arabic(self, arabic_edoc):
        assert arabic_edoc.chapters[0].title == "الفصل الأول: مقدمة"

    def test_second_chapter_title_arabic(self, arabic_edoc):
        assert arabic_edoc.chapters[1].title == "الفصل الثاني: المعيار"

    def test_lessons_exist(self, arabic_edoc):
        for chapter in arabic_edoc.chapters:
            assert len(chapter.lessons) >= 1, (
                f"Chapter '{chapter.title}' has no lessons"
            )

    def test_table_element_detected(self, arabic_edoc):
        tables = [
            el
            for ch in arabic_edoc.chapters
            for le in ch.lessons
            for el in le.elements
            if el.type == ElementType.TABLE
        ]
        assert len(tables) >= 1, "Arabic document must produce at least one TABLE element"

    def test_paragraph_elements_exist(self, arabic_edoc):
        paragraphs = [
            el
            for ch in arabic_edoc.chapters
            for le in ch.lessons
            for el in le.elements
            if el.type == ElementType.PARAGRAPH
        ]
        assert len(paragraphs) >= 1


class TestArabicLanguageAgnostic:
    """Language-agnostic: no hardcoded language code assumed."""

    def test_language_is_none(self, arabic_edoc):
        assert arabic_edoc.language is None, (
            "language must be None when no detector confirmed it — "
            "do not hardcode 'ar'"
        )

    def test_parser_field_set(self, arabic_edoc):
        assert arabic_edoc.parser == "docling"


class TestArabicMetadata:
    """Metadata completeness for Arabic elements — same contract as English."""

    def test_all_elements_have_parser(self, arabic_edoc):
        for el, _, _ in arabic_edoc.all_elements():
            assert el.metadata.parser, "Arabic element missing parser metadata"

    def test_all_elements_have_chapter(self, arabic_edoc):
        for el, _, _ in arabic_edoc.all_elements():
            assert el.metadata.chapter, "Arabic element missing chapter metadata"

    def test_all_elements_have_lesson(self, arabic_edoc):
        for el, _, _ in arabic_edoc.all_elements():
            assert el.metadata.lesson, "Arabic element missing lesson metadata"


class TestArabicChunking:
    """Chunking works on Arabic EducationalDocument without modification."""

    def test_chunks_produced(self, arabic_chunks):
        assert len(arabic_chunks) >= 1

    def test_chunks_have_required_fields(self, arabic_chunks):
        required = ("text", "type", "chapter", "lesson", "page", "parser")
        for i, chunk in enumerate(arabic_chunks):
            missing = [f for f in required if not chunk.get(f)]
            assert not missing, f"Arabic chunk {i} missing: {missing}"

    def test_arabic_table_chunk_exists(self, arabic_chunks):
        table_chunks = [c for c in arabic_chunks if c["type"] == "table"]
        assert len(table_chunks) >= 1, "Arabic document must produce a table chunk"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
