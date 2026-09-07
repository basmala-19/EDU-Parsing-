"""High-fidelity mode produces ordered, page-addressable Markdown."""

from educational.llm_parser import QwenPageParser


def test_qwen_page_parser_marks_every_page(monkeypatch, tmp_path):
    import pymupdf as fitz

    pdf_path = tmp_path / "two-pages.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    parser = QwenPageParser()
    monkeypatch.setattr(parser.refiner, "_page_image", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(parser.refiner, "_generate", lambda *_: "# Parsed page")
    markdown = parser.parse(str(pdf_path))

    assert parser.last_page_markers == [1, 2]
    assert "<!-- page: 1 -->" in markdown
    assert "<!-- page: 2 -->" in markdown
