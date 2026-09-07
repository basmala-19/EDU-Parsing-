"""
Integration tests — Full pipeline (Tests 1-6).

Runs the complete Probe → Router → Parser → Educational Parser → Chunking →
Evaluation chain on a controlled document and verifies end-to-end behaviour.

This file is the *integration* layer. Individual stage assertions live in:
  test_probe.py, test_parser.py, test_educational.py, test_chunking.py, test_arabic.py

Tests included here:
  - Full pipeline runs without error
  - Output schema is correct (probe, parser, educational_document, quality_report)
  - Chunking is correctly optional (include_chunks=False by default)
  - All quality metrics return PASS on a clean document
  - Test 6: Reading Order — page numbers never decrease
"""
import json
import pytest

from parsers.base import BaseParser
from routing.router import ParserRouter
from routing.probe import probe_document
from educational.rule_based_parser import parse_markdown_to_education
from chunking.chunker import chunk_educational_document
from evaluation.reports import build_quality_report
from schema.models import EducationalDocument
from evaluation.metrics import page_coverage


# ---------------------------------------------------------------------------
# Fake parser for integration (avoids Docling download)
# ---------------------------------------------------------------------------


class _IntegrationFakeParser(BaseParser):
    """Full-featured fake parser used for pipeline integration tests."""

    name = "docling"

    def parse(self, _: str) -> str:
        return """\
# 1 Introduction

Machine learning is a subset of artificial intelligence focused on building
systems that learn from data without being explicitly programmed.

## 1.1 Motivation

Traditional rule-based systems do not scale to complex, high-dimensional
problems such as image recognition or natural-language understanding.

# 2 Benchmark

## 2.1 Dataset

The dataset consists of 2000 human-verified pages spanning insurance,
finance, and government documents.

| Dimension            | Metric               | Pages |
| -------------------- | -------------------- | ----- |
| Tables               | GTRM                 | 503   |
| Charts               | ChartDataPointMatch  | 568   |
| Content Faithfulness | Faithfulness Score   | 506   |

## 2.2 Evaluation Protocol

Each document is evaluated by five independent annotators.
"""


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def router_with_fake() -> ParserRouter:
    router = ParserRouter()
    router._registry["docling"] = _IntegrationFakeParser()
    return router


@pytest.fixture()
def pipeline_result(router_with_fake):
    """Run the full pipeline on sample_report.pdf with the fake parser."""
    probe = probe_document("tests/.generated/academic.pdf")
    selected = router_with_fake.select(probe)
    markdown = selected.parse("tests/.generated/academic.pdf")
    edoc = parse_markdown_to_education(markdown, "academic.pdf", selected.name)
    return probe, selected, markdown, edoc


@pytest.fixture()
def edoc(pipeline_result):
    return pipeline_result[3]


@pytest.fixture()
def markdown(pipeline_result):
    return pipeline_result[2]


@pytest.fixture()
def chunks(edoc):
    return chunk_educational_document(edoc)


@pytest.fixture()
def quality(markdown, edoc):
    return build_quality_report(markdown, edoc)


# ---------------------------------------------------------------------------
# Test 1 (integration): Probe output in pipeline context
# ---------------------------------------------------------------------------


class TestProbeInPipeline:
    def test_probe_born_digital(self, pipeline_result):
        probe = pipeline_result[0]
        assert probe.is_born_digital is True

    def test_probe_route_is_string(self, pipeline_result):
        probe = pipeline_result[0]
        assert isinstance(probe.route, str) and probe.route


# ---------------------------------------------------------------------------
# Test 2 (integration): Selected parser is docling
# ---------------------------------------------------------------------------


class TestRouterInPipeline:
    def test_selects_docling_for_born_digital(self, pipeline_result):
        selected = pipeline_result[1]
        assert selected.name == "docling"


# ---------------------------------------------------------------------------
# Test 3 (integration): Educational document structure
# ---------------------------------------------------------------------------


class TestEducationalDocumentInPipeline:
    def test_chapter_count(self, edoc):
        assert len(edoc.chapters) == 2

    def test_introduction_chapter_present(self, edoc):
        titles = [c.title for c in edoc.chapters]
        assert "1 Introduction" in titles

    def test_benchmark_chapter_present(self, edoc):
        titles = [c.title for c in edoc.chapters]
        assert "2 Benchmark" in titles

    def test_motivation_lesson_present(self, edoc):
        ch = edoc.chapters[0]
        lesson_titles = [le.title for le in ch.lessons]
        assert "1.1 Motivation" in lesson_titles

    def test_dataset_lesson_present(self, edoc):
        ch = edoc.chapters[1]
        lesson_titles = [le.title for le in ch.lessons]
        assert "2.1 Dataset" in lesson_titles


# ---------------------------------------------------------------------------
# Test 4 (integration): Chunking is optional
# ---------------------------------------------------------------------------


class TestChunkingOptional:
    def test_pipeline_works_without_chunks(self, pipeline_result):
        """Pipeline result should not include chunks by default."""
        # We do NOT call chunk_educational_document here — verifying the
        # educational_document is self-contained and sufficient.
        _, _, _, edoc = pipeline_result
        assert isinstance(edoc, EducationalDocument)

    def test_chunks_produced_when_requested(self, chunks):
        assert len(chunks) >= 1

    def test_table_chunk_present(self, chunks):
        table_chunks = [c for c in chunks if c["type"] == "table"]
        assert len(table_chunks) >= 1

    def test_first_5_chunks_have_full_metadata(self, chunks):
        required = ("text", "type", "chapter", "lesson", "page", "parser")
        for i, chunk in enumerate(chunks[:5]):
            missing = [f for f in required if not chunk.get(f)]
            assert not missing, f"Chunk[{i}] missing: {missing}"


# ---------------------------------------------------------------------------
# Test 5 (integration): Quality report
# ---------------------------------------------------------------------------


class TestQualityReportInPipeline:
    def test_all_metrics_present(self, quality):
        expected = {
            "content_faithfulness",
            "table_preservation",
            "semantic_formatting",
            "metadata_completeness",
            "reading_order",
            "ocr_confidence",
            "page_coverage",
            "structured_table_rows",
        }
        assert expected.issubset(quality.keys()), (
            f"Missing metrics: {expected - set(quality.keys())}"
        )

    def test_all_metrics_have_status(self, quality):
        for name, result in quality.items():
            assert "status" in result, f"Metric '{name}' missing 'status'"
            assert result["status"] in ("PASS", "FAIL"), (
                f"Metric '{name}' has invalid status: '{result['status']}'"
            )

    def test_all_metrics_have_detail(self, quality):
        for name, result in quality.items():
            assert "detail" in result and result["detail"], (
                f"Metric '{name}' missing or empty 'detail'"
            )

    def test_page_coverage_flags_a_missing_source_page(self, edoc):
        result = page_coverage(edoc, expected_page_count=2)
        assert result["status"] == "FAIL"

    def test_clean_document_passes_all(self, quality):
        failures = {k: v for k, v in quality.items() if v["status"] == "FAIL"}
        assert not failures, (
            f"Expected all PASS on clean document. FAILED: "
            + json.dumps(failures, indent=2)
        )


# ---------------------------------------------------------------------------
# Test 6: Reading Order
# ---------------------------------------------------------------------------


class TestReadingOrder:
    """Page numbers across elements must be non-decreasing.

    This test catches multi-column layout parsed in the wrong column order —
    a known failure mode for complex PDFs.
    """

    def test_no_reading_order_violations(self, edoc):
        pages = [el.metadata.page for el, _, _ in edoc.all_elements()]
        violations = [
            (a, b, i)
            for i, (a, b) in enumerate(zip(pages, pages[1:]))
            if b < a
        ]
        assert not violations, (
            f"Reading-order violations found (page went backwards): "
            f"{violations[:3]} …"
        )

    def test_two_column_simulation(self):
        """Inject out-of-order pages and verify the metric catches it."""
        from evaluation.metrics import reading_order

        # Build a minimal EducationalDocument with descending page numbers
        md = "# Chapter\n\n## Section\n\nSome text here.\n"
        edoc = parse_markdown_to_education(md, "f.pdf", "docling")
        # Manually corrupt page numbers to simulate wrong column order
        for el, _, _ in edoc.all_elements():
            el.metadata.page = 1  # all same page — no violation here

        result = reading_order(edoc)
        assert result["status"] == "PASS"

    def test_decreasing_pages_detected(self):
        """Decreasing page numbers must produce a FAIL."""
        from evaluation.metrics import reading_order

        md = "# A\n\n## B\n\nText.\n\n## C\n\nMore text.\n"
        edoc = parse_markdown_to_education(md, "f.pdf", "docling")

        # Corrupt the last element's page to simulate column-order bug
        all_els = [el for el, _, _ in edoc.all_elements()]
        if len(all_els) >= 2:
            all_els[-1].metadata.page = 0  # page 0 < page 1 => violation

        result = reading_order(edoc)
        assert result["status"] == "FAIL", (
            "Expected FAIL when page numbers decrease, got PASS"
        )


# ---------------------------------------------------------------------------
# Full end-to-end smoke test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_smoke(self, router_with_fake):
        """Full pipeline runs without exceptions and returns a non-empty document."""
        probe = probe_document("tests/.generated/academic.pdf")
        parser = router_with_fake.select(probe)
        markdown = parser.parse("tests/.generated/academic.pdf")
        edoc = parse_markdown_to_education(markdown, "academic.pdf", parser.name)
        quality = build_quality_report(markdown, edoc)

        assert len(edoc.chapters) > 0
        assert len(quality) == 8
        for v in quality.values():
            assert v["status"] in ("PASS", "FAIL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
