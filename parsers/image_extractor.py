"""PDF image extraction and attachment to an EducationalDocument."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from schema.models import EducationalDocument, Element, ElementMetadata, ElementType

MIN_IMAGE_EDGE_PX = 64
MAX_EMBEDDED_IMAGES_PER_PAGE = 24


def _save_page_snapshot(page: fitz.Page, destination: Path, page_number: int) -> str:
    """Save one page image for scans/composite pages with many image fragments."""
    path = destination / f"page-{page_number:03d}-render.png"
    pixmap = page.get_pixmap(dpi=144, alpha=False)
    pixmap.save(str(path))
    return path.as_posix()


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str = "artifacts/images",
    max_pages: int | None = None,
) -> dict[int, list[str]]:
    """Save embedded raster images and return ``page -> relative paths``.

    This deliberately preserves the original image bytes/format where possible;
    converting every image to PNG needlessly inflates a textbook corpus.
    """
    pdf = fitz.open(pdf_path)
    destination = Path(output_dir) / Path(pdf_path).stem
    destination.mkdir(parents=True, exist_ok=True)
    by_page: dict[int, list[str]] = {}

    try:
        for page_number, page in enumerate(pdf, start=1):
            if max_pages is not None and page_number > max_pages:
                break
            page_images = page.get_images(full=True)

            # Scanned textbooks often encode one visual page as hundreds of
            # image tiles/masks.  Extracting every tile bloats storage and can
            # crash on alpha JPEG conversion; a rendered page is the correct
            # visual RAG asset in that case.
            if len(page_images) > MAX_EMBEDDED_IMAGES_PER_PAGE:
                by_page[page_number] = [_save_page_snapshot(page, destination, page_number)]
                continue

            seen_xrefs: set[int] = set()
            for index, image in enumerate(page_images, start=1):
                xref = image[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                if image[2] < MIN_IMAGE_EDGE_PX or image[3] < MIN_IMAGE_EDGE_PX:
                    continue

                try:
                    extracted = pdf.extract_image(xref)
                    extension = extracted.get("ext", "png")
                    path = destination / f"page-{page_number:03d}-image-{index:02d}.{extension}"
                    path.write_bytes(extracted["image"])
                except Exception:
                    # Some PDFs contain alpha/CMYK objects PyMuPDF cannot
                    # encode as JPEG.  PNG pixmap output preserves them.
                    try:
                        path = destination / f"page-{page_number:03d}-image-{index:02d}.png"
                        fitz.Pixmap(pdf, xref).save(str(path))
                    except Exception:
                        continue

                by_page.setdefault(page_number, []).append(path.as_posix())
    finally:
        pdf.close()
    return by_page


def attach_extracted_images(
    document: EducationalDocument,
    images_by_page: dict[int, list[str]],
    parser: str,
) -> None:
    """Attach extracted images to the nearest preceding lesson in reading order.

    Markdown exports frequently omit page anchors.  The association is exact
    when page markers are present and explicitly marked ``approximate`` when
    they are not, so retrieval never pretends to know more than it does.
    """
    # Keep object references, not only display titles. Textbooks frequently
    # repeat titles such as "Introduction" across chapters; resolving a
    # chapter/lesson again by title can then select a different lesson and
    # raise StopIteration.
    elements = [
        (element, chapter, lesson)
        for chapter in document.chapters
        for lesson in chapter.lessons
        for element in lesson.elements
    ]
    for page, image_paths in images_by_page.items():
        earlier = [entry for entry in elements if entry[0].metadata.page <= page]
        if earlier:
            _, chapter, lesson = earlier[-1]
            chapter_title, lesson_title = chapter.title, lesson.title
            association = "page_anchored" if earlier[-1][0].metadata.page == page else "nearest_preceding"
        elif document.chapters:
            chapter = document.chapters[0]
            lesson = chapter.lessons[0]
            chapter_title, lesson_title, association = chapter.title, lesson.title, "document_default"
        else:
            continue

        for image_path in image_paths:
            image = Element(
                type=ElementType.IMAGE,
                text=f"[Image: {Path(image_path).name}]",
                metadata=ElementMetadata(
                    page=page,
                    chapter=chapter_title,
                    lesson=lesson_title,
                    parser=parser,
                    extra={"image_path": image_path, "association": association},
                ),
            )
            # Appending all extracted figures after textual parsing makes a
            # multi-page lesson appear as ``page 29 -> page 1`` in the flat
            # document order.  Insert each visual at its page position.
            insert_at = len(lesson.elements)
            for index, existing in enumerate(lesson.elements):
                if existing.metadata.page > page:
                    insert_at = index
                    break
            lesson.elements.insert(insert_at, image)
