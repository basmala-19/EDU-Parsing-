"""
Unit tests — Parser output structure (Test 2).

Verifies that any parser compatible with BaseParser emits Markdown that:
  - Contains at least one heading (#)
  - Preserves table structure (|…|)
  - Preserves list items (- or *)
  - Is non-empty

Uses a fake parser to avoid requiring Docling / LlamaParse installation.
The same assertions apply to any real parser plugged into the pipeline.
"""
import re
import pytest
from parsers.base import BaseParser


# ---------------------------------------------------------------------------
# Fake parsers simulating real parser output
# ---------------------------------------------------------------------------


class _AcademicFakeParser(BaseParser):
    """Simulates Docling output on an English academic document."""

    name = "docling"

    def parse(self, _: str) -> str:
        return """\
# 1 Introduction

Machine learning is a subset of artificial intelligence focused on building
systems that learn from data.

## 1.1 Motivation

Traditional rule-based systems do not scale to high-dimensional problems.

- Scale issues
- No automatic generalisation
- Brittle to distribution shift

# 2 Benchmark

## 2.1 Dataset

The dataset spans insurance, finance, and government documents.

| Dimension            | Metric               | Pages |
| -------------------- | -------------------- | ----- |
| Tables               | GTRM                 | 503   |
| Charts               | ChartDataPointMatch  | 568   |
| Content Faithfulness | Faithfulness Score   | 506   |

![Figure 1: System architecture](fig1.png)
Caption: Overview of the parsing pipeline.
"""


class _MinimalFakeParser(BaseParser):
    """Minimal parser — just a heading and a paragraph."""

    name = "minimal"

    def parse(self, _: str) -> str:
        return "# Only Heading\n\nSome text here.\n"


FAKE_PARSER = _AcademicFakeParser()
MINIMAL_PARSER = _MinimalFakeParser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
TABLE_RE = re.compile(r"^\|.+\|", re.MULTILINE)
LIST_RE = re.compile(r"^[-*]\s+\S", re.MULTILINE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParserOutputNotEmpty:
    def test_output_is_non_empty(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert md.strip(), "Parser must return non-empty Markdown"

    def test_output_is_string(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert isinstance(md, str)


class TestHeadingPreservation:
    """Headings (#, ##) must be present in parser output."""

    def test_has_at_least_one_heading(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert HEADING_RE.search(md), "Parser output must contain at least one Markdown heading"

    def test_has_level_1_heading(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert re.search(r"^#\s+\S", md, re.MULTILINE), "Expected at least one # (chapter-level) heading"

    def test_has_level_2_heading(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert re.search(r"^##\s+\S", md, re.MULTILINE), "Expected at least one ## (lesson-level) heading"


class TestTablePreservation:
    """Tables must survive parsing as Markdown pipe tables."""

    def test_tables_present(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert TABLE_RE.search(md), "Parser output must preserve tables as | … | rows"

    def test_table_has_separator_row(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert re.search(r"^\|[-| :]+\|", md, re.MULTILINE), (
            "Table must have a separator row (| --- | --- |)"
        )


class TestListPreservation:
    """Bullet lists must be preserved."""

    def test_lists_present(self):
        md = FAKE_PARSER.parse("any.pdf")
        assert LIST_RE.search(md), "Parser output must preserve bullet lists (- item or * item)"


class TestMinimalParser:
    """Assertions hold even for a minimal one-heading document."""

    def test_minimal_has_heading(self):
        md = MINIMAL_PARSER.parse("any.pdf")
        assert HEADING_RE.search(md)

    def test_minimal_has_text(self):
        md = MINIMAL_PARSER.parse("any.pdf")
        assert len(md.strip()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
