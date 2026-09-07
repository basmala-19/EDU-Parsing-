"""Self-contained PDF fixtures for tests that exercise the probe stage."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


def _create_born_digital_pdf(path: Path, title: str) -> None:
    """Create a small, selectable-text PDF when the shipped fixture is absent."""
    document = fitz.open()
    page = document.new_page()
    body = " ".join(
        [
            "This educational sample contains selectable text for reliable probe testing."
        ]
        * 12
    )
    page.insert_textbox(fitz.Rect(54, 54, 540, 760), f"{title}\n\n{body}", fontsize=12)
    document.save(path)
    document.close()


def _create_scanned_pdf(path: Path) -> None:
    """Create an image-only PDF fixture without selectable text when absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=300, height=300)
    pix = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 100, 100), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(0, 0, 300, 300), pixmap=pix)
    document.save(path)
    document.close()


def pytest_sessionstart(session) -> None:
    """Keep the test suite runnable after zip extraction without binary assets."""
    root = Path(session.config.rootpath)
    fixtures_dir = root / "tests" / ".generated"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    for filename, title in (
        ("academic.pdf", "Academic sample"),
        ("arabic_book.pdf", "Arabic textbook sample"),
    ):
        path = fixtures_dir / filename
        if not path.exists():
            _create_born_digital_pdf(path, title)

    scanned_path = root / "test_docs" / "scanned.pdf"
    if not scanned_path.exists():
        _create_scanned_pdf(scanned_path)
