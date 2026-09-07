"""
Table of Contents (TOC / الفهرس) Extraction and Spine Index.

Ported and enhanced from EDU-RAG-integration-version/src/features/rag/application/toc.py.
Extracts the official chapters, units, lessons, and starting pages from the textbook's
own Table of Contents (pages 1-15), supporting both dot-leader lines and table formats.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional


_TOC_TITLE_RE = re.compile(
    r"^(?:table\s+of\s+contents|contents|index|course\s+contents|brief\s+contents|"
    r"(?:ال)?فهرس(?:\s+المحتويات)?|(?:ال)?محتويات|محتوى\s+الكتاب|فهرس\s+الموضوعات|قائمة\s+المحتويات)\s*$",
    re.I,
)

_UNIT_WORD_RE = re.compile(
    r"\b(?:unit|module|chapter|part|theme|book)\b|"
    r"الوحدة|الفصل(?!\s+الدراسي)|الباب|المحور|القسم|الجزء",
    re.I,
)
_LESSON_WORD_RE = re.compile(
    r"\b(?:lesson|lecture|topic|section|subtopic|session)\b|"
    r"الدرس|المحاضرة|الموضوع|المبحث|المطلب",
    re.I,
)

_LEADER_RUN_RE = re.compile(r"[.\u2026_\-·•\s]{3,}")
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_NON_WORD_RE = re.compile(r"[^\w\u0600-\u06FF ]+", re.U)


def convert_arabic_digits_to_ascii(text: str) -> str:
    """Convert Eastern Arabic/Persian digits to standard ASCII digits."""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    for i, d in enumerate(arabic_digits):
        text = text.replace(d, str(i))
    for i, d in enumerate(persian_digits):
        text = text.replace(d, str(i))
    # Handle OCR common misread of 7 (Arabic ٧) as 'V' or 'VV' (77)
    text = re.sub(r"\bVV\b", "77", text)
    text = re.sub(r"\bV\b", "7", text)
    return text


@dataclass
class TocEntry:
    """One verified entry from the textbook's Table of Contents."""
    title: str
    normalized: str
    level: int  # 1 = Chapter/Unit/الباب/الفصل, 2 = Lesson/الدرس/Topic
    page: Optional[int] = None
    end_page: Optional[int] = None


def normalize_for_match(text: str) -> str:
    """Normalize a heading/TOC title for tolerant comparison."""
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", text)
    value = _DIACRITICS_RE.sub("", value)
    for src, dst in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        value = value.replace(src, dst)
    value = _NON_WORD_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def similarity(a: str, b: str) -> float:
    """Fuzzy matching score between two titles in [0, 1]."""
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if len(na) >= 4 and len(nb) >= 4 and (na in nb or nb in na):
        return 0.92
    sa = " ".join(sorted(na.split()))
    sb = " ".join(sorted(nb.split()))
    return SequenceMatcher(None, sa, sb).ratio()


def is_toc_heading_line(line: str) -> bool:
    """True if line is a TOC header like 'Contents', 'الفهرس', 'محتوى الكتاب'."""
    stripped = (line or "").strip(" \t|-#:*_")
    return bool(_TOC_TITLE_RE.match(stripped))


def parse_toc_entry_line(line: str) -> tuple[str, int | None] | None:
    """Parse a TOC entry from either a pipe table row or a dot-leader line."""
    stripped = (line or "").strip()
    if not stripped:
        return None

    # Case A: Pipe table row e.g. "| ٥ | الفصل الأول الدعامة والحركة |" or "| Chapter 1 | 5 |"
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # Exclude separator lines (|---|---|)
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            return None
        if len(cells) >= 2:
            c0_digits = convert_arabic_digits_to_ascii(cells[0])
            c1_digits = convert_arabic_digits_to_ascii(cells[1])
            c_last_digits = convert_arabic_digits_to_ascii(cells[-1])
            
            # Check if first or last column is a page number
            if re.fullmatch(r"\d{1,4}", c0_digits):
                title = " — ".join(c for c in cells[1:] if c)
                return title, int(c0_digits)
            elif re.fullmatch(r"\d{1,4}", c_last_digits):
                title = " — ".join(c for c in cells[:-1] if c)
                return title, int(c_last_digits)
            elif re.fullmatch(r"\d{1,4}", c1_digits) and len(cells) == 2:
                return cells[0], int(c1_digits)
            else:
                # E.g. "| 1-1 | Information and Media |"
                title = " — ".join(c for c in cells if c and not re.fullmatch(r":?-{3,}:?", c))
                if title:
                    return title, None

    # Case B: Dot leader line or plain line with page number e.g. "Chapter 1 .......... 5"
    collapsed = _LEADER_RUN_RE.sub(" ", stripped)
    collapsed = re.sub(r"\s+", " ", collapsed).strip(" .")
    if not collapsed:
        return None

    # Convert digits
    ascii_line = convert_arabic_digits_to_ascii(collapsed)

    trailing = re.match(r"^(?P<title>.+?)\s+(?P<page>\d{1,4})$", ascii_line)
    if trailing:
        title = trailing.group("title").strip(" .:-")
        if title and len(title) >= 3:
            return title, int(trailing.group("page"))

    leading = re.match(r"^(?P<page>\d{1,4})\s+(?P<title>.+)$", ascii_line)
    if leading:
        title = leading.group("title").strip(" .:-")
        if title and len(title) >= 3:
            return title, int(leading.group("page"))

    # Case C: Bare chapter title with keyword e.g. "الفصل الأول: الدعامة والحركة" or "Chapter 1 Introduction"
    if _UNIT_WORD_RE.search(stripped) or _LESSON_WORD_RE.search(stripped):
        return stripped.strip(" *#-_"), None

    return None


def _keyword_level(title: str) -> int:
    """Determine if title is a Chapter (1) or Lesson (2)."""
    if _UNIT_WORD_RE.search(title):
        return 1
    if _LESSON_WORD_RE.search(title):
        return 2
    # Numbering patterns like 1.1 -> Lesson (2), 1 -> Chapter (1)
    if re.match(r"^\d+\.\d+", title):
        return 2
    return 1


class TocIndex:
    """Table of Contents index that anchors chapter and lesson boundaries by page and title."""

    def __init__(self, entries: list[TocEntry]) -> None:
        self.entries = entries
        self._calculate_page_ranges()

    def _calculate_page_ranges(self) -> None:
        """Assign chapter ranges from the next *chapter*, not next lesson.

        A contents page commonly interleaves chapter and lesson entries.  The
        old implementation ended a chapter at its first lesson's start page,
        which made a chapter disappear after a page or two and later caused a
        duplicate/default chapter to be created.  Only another level-1 entry
        is a chapter boundary.
        """
        chapters = [entry for entry in self.entries if entry.level == 1 and entry.page is not None]
        for index, chapter in enumerate(chapters):
            chapter.end_page = chapters[index + 1].page - 1 if index + 1 < len(chapters) else 99999

    @classmethod
    def from_markdown(cls, markdown_text: str, max_pages_to_scan: int = 15) -> Optional["TocIndex"]:
        """Scan early pages of document markdown to find and parse Table of Contents."""
        lines = markdown_text.splitlines()
        toc_lines: list[str] = []
        in_toc = False
        toc_start_page: int | None = None
        current_page = 1

        for line in lines:
            # Track page markers
            p_match = re.match(r"^<!--\s*page\s*:\s*(\d+)\s*-->", line, re.I)
            if p_match:
                current_page = int(p_match.group(1))
                if current_page > max_pages_to_scan:
                    break

            if not in_toc:
                if is_toc_heading_line(line) and current_page <= max_pages_to_scan:
                    in_toc = True
                    toc_start_page = current_page
                    continue
            else:
                # In TOC section: collect lines
                stripped = line.strip()
                if not stripped:
                    continue
                # If we encounter a new primary heading on a later page (e.g. # Chapter 1 / Unit 1 / الباب الأول)
                if toc_start_page is not None and current_page > toc_start_page and (
                    line.startswith("# ")
                    or re.match(r"^(?:الباب|الوحدة|الفصل|Chapter|Unit|Part)\s+(?:الأول|الاول|1|One)\b", stripped, re.I)
                ):
                    if not stripped.startswith("|") and not _LEADER_RUN_RE.search(stripped):
                        break
                toc_lines.append(line)

        if not toc_lines:
            # Fallback: scan for any table on pages 1-10 containing chapter/unit keywords in AR or EN
            candidate_table_lines: list[str] = []
            cur_p = 1
            for line in lines:
                p_m = re.match(r"^<!--\s*page\s*:\s*(\d+)\s*-->", line, re.I)
                if p_m:
                    cur_p = int(p_m.group(1))
                    if cur_p > 10:
                        break
                if cur_p <= 10 and line.strip().startswith("|"):
                    candidate_table_lines.append(line)
            if candidate_table_lines:
                # Check if candidate lines contain any chapter/unit words
                combined = " ".join(candidate_table_lines)
                if _UNIT_WORD_RE.search(combined) or _LESSON_WORD_RE.search(combined):
                    toc_lines = candidate_table_lines

        entries: list[TocEntry] = []
        for line in toc_lines:
            parsed = parse_toc_entry_line(line)
            if not parsed:
                continue
            title, page = parsed
            # Clean title
            title = re.sub(r"^(?:#+|\*+)\s*", "", title).strip(" *#-_")
            if not title or len(title) < 2:
                continue
            # Skip header rows with no page number that simply label columns (e.g. "Chapter — Page" or "الموضوع — الصفحة")
            if page is None and re.match(
                r"^(?:chapter|unit|lesson|topic|title|page|no|number|الموضوع|الصفحة|الفصل|الوحدة|الدرس|رقم|م)\s*[\u2014\-–|/]\s*(?:page|topic|title|chapter|lesson|الصفحة|الموضوع|الفصل|الوحدة|الدرس)",
                title,
                re.I,
            ):
                continue
            level = _keyword_level(title)
            entries.append(
                TocEntry(
                    title=title,
                    normalized=normalize_for_match(title),
                    level=level,
                    page=page,
                )
            )

        # A chapter start that moves backwards is an OCR artefact (for example
        # ``103`` read as ``10``).  It must not replace the active chapter in
        # the document spine.
        filtered: list[TocEntry] = []
        last_chapter_page: int | None = None
        for entry in entries:
            if entry.level == 1 and entry.page is not None:
                if last_chapter_page is not None and entry.page <= last_chapter_page:
                    continue
                last_chapter_page = entry.page
            filtered.append(entry)

        return cls(filtered) if filtered else None

    def get_chapter_for_page(self, page_num: int) -> Optional[str]:
        """Find the active Chapter title for a given page number."""
        chapters = [e for e in self.entries if e.level == 1 and e.page is not None]
        active = None
        for ch in chapters:
            if ch.page is not None and ch.page <= page_num:
                if ch.end_page is None or page_num <= ch.end_page:
                    active = ch.title
        return active

    @property
    def first_chapter_page(self) -> Optional[int]:
        """First credible chapter page, excluding cover and contents pages."""
        pages = [entry.page for entry in self.entries if entry.level == 1 and entry.page is not None]
        return min(pages) if pages else None

    def match_heading(self, heading_text: str, page_num: Optional[int] = None, min_ratio: float = 0.70) -> Optional[TocEntry]:
        """Find best matching TOC entry for a heading candidate."""
        if not heading_text or not self.entries:
            return None
        best: tuple[TocEntry, float] | None = None
        for entry in self.entries:
            score = similarity(heading_text, entry.title)
            if page_num and entry.page and abs(entry.page - page_num) <= 2:
                score += 0.15  # Bonus for matching nearby page number
            if score >= min_ratio and (best is None or score > best[1]):
                best = (entry, score)
        return best[0] if best else None
