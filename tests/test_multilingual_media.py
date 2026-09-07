"""Regression coverage for Arabic, page markers, and RAG image chunks."""
from educational.rule_based_parser import parse_markdown_to_education
from chunking.chunker import chunk_educational_document
from evaluation.metrics import semantic_formatting
from schema.models import ElementType


ARABIC_CHAPTER = "\u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u0623\u0648\u0644: \u0627\u0644\u0643\u0633\u0648\u0631"
ARABIC_LESSON = "\u0627\u0644\u062f\u0631\u0633 \u0627\u0644\u0623\u0648\u0644: \u0645\u0642\u0627\u0631\u0646\u0629 \u0627\u0644\u0643\u0633\u0648\u0631"


def test_real_arabic_unmarked_headings_are_structured():
    markdown = f"{ARABIC_CHAPTER}\n\n{ARABIC_LESSON}\n\n\u0646\u0635 \u062a\u0639\u0644\u064a\u0645\u064a.\n"
    document = parse_markdown_to_education(markdown, "math-ar.pdf", "docling", "ar")

    assert document.language == "ar"
    assert document.chapters[0].title == ARABIC_CHAPTER
    assert ARABIC_LESSON in [lesson.title for lesson in document.chapters[0].lessons]


def test_embedded_arabic_ocr_chapter_heading_is_recovered_without_dropping_text():
    markdown = (
        "<!-- page: 5 -->\n<!-- ocr_confidence: 75 -->\n"
        "123 \u0627\u0644\u0628\u0627\u0628\u0627\u0644\u0623\u0648\u0644 - \u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u0623\u0648\u0644 \u0627\u0644\u062f\u0639\u0627\u0645\u0629 \u0648\u0627\u0644\u062d\u0631\u0643\u0629 \u0641\u064a \u0646\u0647\u0627\u064a\u0629 \u0647\u0630\u0627 \u0627\u0644\u0641\u0635\u0644 \u064a\u062a\u0639\u0631\u0641 \u0627\u0644\u0637\u0627\u0644\u0628.\n"
    )
    document = parse_markdown_to_education(markdown, "biology.pdf", "tesseract", "ar")
    headings = [element for element, _, _ in document.all_elements() if element.type == ElementType.HEADING]
    paragraphs = [element for element, _, _ in document.all_elements() if element.type == ElementType.PARAGRAPH]

    assert any("\u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u0623\u0648\u0644" in heading.text for heading in headings)
    assert any("\u064a\u062a\u0639\u0631\u0641 \u0627\u0644\u0637\u0627\u0644\u0628" in paragraph.text for paragraph in paragraphs)


def test_long_ocr_document_with_sparse_headings_fails_structure_quality():
    document = parse_markdown_to_education(
        "\u0627\u0644\u0641\u0635\u0644 \u0627\u0644\u0623\u0648\u0644\n\n" + "\u0646\u0635 \u062a\u0639\u0644\u064a\u0645\u064a" * 30,
        "biology.pdf",
        "tesseract",
        "ar",
    )
    report = semantic_formatting(document, expected_page_count=167)
    assert report["status"] == "FAIL"
    assert "expected at least" in report["detail"]


def test_markdown_image_keeps_page_and_retrieval_context():
    markdown = """# Biology

## Cell

<!-- page: 16 -->
![Cell diagram](assets/cell.png)
"""
    document = parse_markdown_to_education(markdown, "biology.pdf", "docling", "en")
    image = next(element for element, _, _ in document.all_elements() if element.type == ElementType.IMAGE)
    chunks = chunk_educational_document(document)
    image_chunk = next(chunk for chunk in chunks if chunk["type"] == "image")

    assert image.metadata.page == 16
    assert image.metadata.extra["image_path"] == "assets/cell.png"
    assert image_chunk["chapter"] == "Biology"
    assert image_chunk["lesson"] == "Cell"
    assert image_chunk["image_path"] == "assets/cell.png"


def test_docling_image_placeholder_is_an_image_not_a_paragraph():
    document = parse_markdown_to_education(
        "# Biology\n\n## Cell\n\n<!-- image -->", "biology.pdf", "docling", "en"
    )
    element = document.all_elements()[-1][0]
    assert element.type == ElementType.IMAGE
    assert element.metadata.extra["association"] == "docling_placeholder"


def test_clean_pipe_table_has_structured_rows():
    document = parse_markdown_to_education(
        "# Math\n\n## Data\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
        "math.pdf",
        "docling",
        "en",
    )
    table = next(el for el, _, _ in document.all_elements() if el.type == ElementType.TABLE)
    assert table.format == "rows"
    assert table.rows == [["A", "B"], ["1", "2"]]


def test_noisy_tesseract_pipe_text_is_not_promoted_to_a_table():
    document = parse_markdown_to_education(
        "| OCR noise |\n| broken", "scan.pdf", "tesseract", "en"
    )
    assert all(element.type != ElementType.TABLE for element, _, _ in document.all_elements())


def test_html_table_is_preserved_with_rows_and_headers():
    document = parse_markdown_to_education(
        "# Data\n\n<table><tr><th>Name</th><th>Score</th></tr><tr><td>Ada</td><td>10</td></tr></table>",
        "table.pdf",
        "docling",
        "en",
    )
    table = next(element for element, _, _ in document.all_elements() if element.type == ElementType.TABLE)
    assert table.format == "html"
    assert table.rows == [["Name", "Score"], ["Ada", "10"]]
    assert table.metadata.extra["headers"] == ["Name", "Score"]
