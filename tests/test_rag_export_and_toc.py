from educational.rule_based_parser import parse_markdown_to_education
from schema.rag_export import educational_document_to_markdown


def test_gemini_ocr_toc_creates_one_real_chapter_not_toc_duplicates():
    markdown = """<!-- page: 1 -->
محتوى الكتاب
الفصل الأول الخلية ٣
الفصل الثاني الوراثة ٩
<!-- page: 3 -->
الفصل الأول
الخلية
نص الدرس
<!-- page: 4 -->
مزيد من النص
"""
    document = parse_markdown_to_education(markdown, "book.pdf", "gemini_ocr", "ar")
    titles = [chapter.title for chapter in document.chapters]
    assert titles.count("الفصل الأول الخلية") == 1
    assert "الفصل الثاني الوراثة" not in titles


def test_rag_export_is_single_markdown_document_with_hierarchy():
    edoc = {
        "subject": "Biology",
        "chapters": [{"title": "الفصل الأول", "lessons": [{"title": "الدرس الأول", "elements": [{"type": "paragraph", "text": "نص علمي"}]}]}],
    }
    assert educational_document_to_markdown(edoc) == "# الفصل الأول\n\n## الدرس الأول\n\nنص علمي\n"
