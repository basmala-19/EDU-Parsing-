"""Regression coverage for visual assets changing document reading order."""

from educational.rule_based_parser import parse_markdown_to_education
from evaluation.metrics import reading_order
from parsers.image_extractor import attach_extracted_images


def test_extracted_images_are_inserted_at_their_page_position():
    document = parse_markdown_to_education(
        "<!-- page: 1 -->\n# Unit\n\n## Lesson\n\nFirst\n\n"
        "<!-- page: 3 -->\nThird",
        "book.pdf",
        "docling",
        "en",
    )
    attach_extracted_images(document, {2: ["assets/page-002.png"]}, parser="docling")
    pages = [element.metadata.page for element, _, _ in document.all_elements()]

    assert pages == sorted(pages)
    assert reading_order(document)["status"] == "PASS"


def test_image_attachment_handles_duplicate_chapter_and_lesson_titles():
    document = parse_markdown_to_education(
        "<!-- page: 1 -->\n# Unit\n\n## Introduction\n\nFirst\n\n"
        "<!-- page: 2 -->\n# Unit\n\n## Introduction\n\nSecond",
        "book.pdf",
        "docling",
        "en",
    )
    attach_extracted_images(document, {2: ["assets/page-002.png"]}, parser="docling")

    second_lesson = next(
        lesson for lesson in document.chapters[1].lessons if lesson.title == "Introduction"
    )
    assert any(element.type.value == "image" for element in second_lesson.elements)
