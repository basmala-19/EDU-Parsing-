"""
Unit tests — Probe stage (Test 1).

Verifies that probe_document() correctly fingerprints a PDF:
  - Reads num_pages
  - Detects born-digital vs. scanned
  - Counts images
  - Flags complex layout
  - Selects the expected route
"""
import pytest
from dataclasses import asdict
from routing.probe import probe_document


# ---------------------------------------------------------------------------
# Fixtures — paths to the test PDFs shipped with the project
# ---------------------------------------------------------------------------

ACADEMIC_PDF = "tests/.generated/academic.pdf"
ARABIC_PDF = "tests/.generated/arabic_book.pdf"
SCANNED_PDF = "test_docs/scanned.pdf"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProbeFields:
    """ProbeResult has all required fields and sensible types."""

    def test_returns_required_fields(self):
        result = probe_document(ACADEMIC_PDF)
        d = asdict(result)
        required = {
            "file", "num_pages", "avg_chars_per_page", "is_born_digital",
            "image_count", "distinct_font_sizes", "likely_has_headings",
            "text_block_count", "likely_has_complex_layout", "route",
            "probe_time_seconds", "per_page",
        }
        assert required.issubset(d.keys()), f"Missing fields: {required - set(d.keys())}"

    def test_num_pages_positive(self):
        result = probe_document(ACADEMIC_PDF)
        assert result.num_pages > 0

    def test_probe_time_non_negative(self):
        result = probe_document(ACADEMIC_PDF)
        assert result.probe_time_seconds >= 0.0

    def test_per_page_length_matches_num_pages(self):
        result = probe_document(ACADEMIC_PDF)
        assert len(result.per_page) == result.num_pages


class TestBornDigitalDetection:
    """Born-digital PDFs are identified correctly."""

    def test_academic_is_born_digital(self):
        result = probe_document(ACADEMIC_PDF)
        assert result.is_born_digital is True, (
            f"Expected born_digital=True for academic PDF, "
            f"got avg_chars_per_page={result.avg_chars_per_page}"
        )

    def test_arabic_is_born_digital(self):
        result = probe_document(ARABIC_PDF)
        assert result.is_born_digital is True

    def test_scanned_is_not_born_digital(self):
        result = probe_document(SCANNED_PDF)
        assert result.is_born_digital is False, (
            f"Expected born_digital=False for scanned PDF, "
            f"got avg_chars_per_page={result.avg_chars_per_page}"
        )


class TestRouteSelection:
    """Probe selects a non-empty route string."""

    def test_route_is_string(self):
        result = probe_document(ACADEMIC_PDF)
        assert isinstance(result.route, str) and result.route

    def test_scanned_routes_to_ocr(self):
        result = probe_document(SCANNED_PDF)
        assert "ocr" in result.route.lower(), (
            f"Scanned PDF should route to OCR, got: {result.route}"
        )

    def test_born_digital_does_not_route_to_ocr(self):
        result = probe_document(ACADEMIC_PDF)
        assert "ocr" not in result.route.lower(), (
            f"Born-digital PDF should not route to OCR, got: {result.route}"
        )


class TestImageCount:
    """Image counting is non-negative and type-correct."""

    def test_image_count_non_negative(self):
        result = probe_document(ACADEMIC_PDF)
        assert result.image_count >= 0

    def test_image_count_is_int(self):
        result = probe_document(ACADEMIC_PDF)
        assert isinstance(result.image_count, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
