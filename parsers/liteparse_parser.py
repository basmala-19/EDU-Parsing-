import re
from pathlib import Path

from parsers.base import BaseParser

# Known reversed prefixes for typical textbook Arabic vocabulary
_REVERSED_ARABIC_WORDS = frozenset({
    "لصفلا", "ةدحولا", "سردلا", "بابلا", "روحملا", "مسقلا", "ءزجلا",
    "ةمدقم", "ةمتاخ", "تارابتخا", "ةلئسأ", "بيردت", "عوضوملا", "ةحفصلا",
    "ةكرحلا", "ةماعدلا", "ةيلمع", "تامولعملا", "ةيميلعتلا", "بولطملا",
})


def is_reversed_arabic(line: str) -> bool:
    """Check if an Arabic line was extracted in visually reversed LTR order."""
    words = re.findall(r"[\u0600-\u06FF]+", line)
    if not words:
        return False
    for w in words:
        if w in _REVERSED_ARABIC_WORDS:
            return True
        # In reversed Arabic, words frequently start with taa marbuta 'ة'
        if len(w) >= 3 and w.startswith("ة"):
            return True
    return False


def fix_arabic_bidi_markdown(text: str) -> str:
    """Reorder reversed Arabic lines while preserving markdown syntax and English terms."""
    try:
        from bidi.algorithm import get_display
    except ImportError:
        return text

    lines = text.splitlines()
    fixed_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Preserve markdown comments, page markers, and empty lines
        if stripped.startswith("<!--") or not stripped:
            fixed_lines.append(line)
            continue

        ar_chars = re.findall(r"[\u0600-\u06FF]", line)
        if len(ar_chars) < 2:
            fixed_lines.append(line)
            continue

        # Only apply BiDi reordering if reversed Arabic characteristics are detected
        if is_reversed_arabic(line):
            h_match = re.match(r"^(#{1,6}\s+)(.*)$", line)
            if h_match:
                prefix, content = h_match.group(1), h_match.group(2)
                fixed_lines.append(prefix + get_display(content))
            else:
                fixed_lines.append(get_display(line))
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


class LiteParseParser(BaseParser):
    """Spatial PDFium parsing with local Markdown and image references."""

    name = "liteparse"

    def __init__(self) -> None:
        # LiteParse exposes structured page results in addition to the combined
        # Markdown.  Keep a lightweight page-indexed text view for the UI and
        # provenance checks; Markdown remains the parser interchange format.
        self.last_page_text: dict[int, str] = {}

    def parse(self, file_path: str) -> str:
        try:
            from liteparse import LiteParse
        except ImportError as exc:
            raise RuntimeError("LiteParse is not installed. Run: pip install liteparse") from exc

        image_dir = Path("artifacts") / "liteparse_images" / Path(file_path).stem
        image_dir.mkdir(parents=True, exist_ok=True)
        parser = LiteParse(
            output_format="markdown",
            ocr_enabled=False,
            image_mode="placeholder",
            extract_images=True,
            image_output_dir=str(image_dir),
            extract_links=True,
            quiet=True,
        )
        result = parser.parse(file_path)
        self.last_page_text = self._collect_page_text(result)

        # Assemble per-page Markdown with explicit provenance markers
        pages_md: list[str] = []
        if getattr(result, "pages", None):
            for fallback_num, page in enumerate(result.pages, 1):
                p_num = getattr(page, "page_num", getattr(page, "page_number", fallback_num))
                try:
                    p_num = int(p_num)
                except (TypeError, ValueError):
                    p_num = fallback_num
                p_md = getattr(page, "markdown", None) or getattr(page, "text", None) or ""
                if str(p_md).strip():
                    pages_md.append(f"<!-- page: {p_num} -->\n{str(p_md).strip()}")

        if pages_md:
            markdown = "\n\n".join(pages_md).strip()
        else:
            markdown = (getattr(result, "text", "") or "").strip()

        if not markdown:
            raise RuntimeError("LiteParse returned empty Markdown")

        # Apply BiDi reordering post-processor to fix reversed Arabic text
        markdown = fix_arabic_bidi_markdown(markdown)
        return markdown

    @staticmethod
    def _collect_page_text(result: object) -> dict[int, str]:
        """Read LiteParse's optional per-page items without coupling to a version.

        Different LiteParse releases expose item text as attributes or dict
        fields.  This deliberately degrades to an empty mapping: it must never
        make a valid Markdown parse fail merely because page metadata changed.
        """
        page_text: dict[int, str] = {}
        for fallback_number, page in enumerate(getattr(result, "pages", []) or [], 1):
            page_number = getattr(page, "page_num", getattr(page, "page_number", fallback_number))
            try:
                page_number = int(page_number)
            except (TypeError, ValueError):
                page_number = fallback_number

            items = getattr(page, "text_items", None) or getattr(page, "items", None) or []
            texts: list[str] = []
            for item in items:
                text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                if str(text).strip():
                    texts.append(str(text).strip())
            if texts:
                page_text[page_number] = "\n\n".join(texts)
        return page_text
