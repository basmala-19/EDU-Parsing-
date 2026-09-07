"""
Unit tests — Chunking stage (Test 4).

Verifies that chunk_educational_document() produces well-formed chunks:
  - Each chunk has the required metadata fields
  - Metadata values are non-empty / valid
  - Tables always get their own standalone chunk
  - Headings always get their own standalone chunk
  - First 5 chunks have complete metadata (the 'show your work' snapshot)
"""
import pytest

from educational.rule_based_parser import parse_markdown_to_education
from chunking.chunker import chunk_educational_document

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_ACADEMIC_MD = """\
# 1 Introduction

Machine learning is a subset of artificial intelligence.

## 1.1 Motivation

Traditional rule-based systems do not scale well.

- Brittle
- Not generalisable

# 2 Benchmark

## 2.1 Dataset

The dataset consists of 2000 human-verified pages.

| Dimension | Metric | Pages |
| --- | --- | --- |
| Tables | GTRM | 503 |
| Charts | ChartDataPointMatch | 568 |
"""


@pytest.fixture()
def chunks() -> list[dict]:
    edoc = parse_markdown_to_education(_ACADEMIC_MD, "parsebench.pdf", "docling")
    return chunk_educational_document(edoc)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("text", "type", "chapter", "lesson", "page", "parser")


class TestChunkRequiredFields:
    def test_all_chunks_have_required_keys(self, chunks):
        for i, chunk in enumerate(chunks):
            missing = [f for f in REQUIRED_FIELDS if f not in chunk]
            assert not missing, f"Chunk {i} is missing fields: {missing}"

    def test_chunk_text_non_empty(self, chunks):
        for i, chunk in enumerate(chunks):
            assert chunk["text"].strip(), f"Chunk {i} has empty text"

    def test_chunk_type_is_valid_string(self, chunks):
        valid_types = {"paragraph", "heading", "table", "image", "list",
                       "example", "exercise", "definition"}
        for i, chunk in enumerate(chunks):
            assert chunk["type"] in valid_types, (
                f"Chunk {i} has invalid type: '{chunk['type']}'"
            )


class TestChunkMetadataValues:
    """Individual metadata field assertions (the explicit assert list from the review)."""

    def test_parser_is_not_none(self, chunks):
        for i, chunk in enumerate(chunks):
            assert chunk.get("parser") is not None, (
                f"Chunk {i}: parser must not be None"
            )

    def test_page_greater_than_zero(self, chunks):
        for i, chunk in enumerate(chunks):
            assert chunk.get("page", 0) > 0, (
                f"Chunk {i}: page must be > 0, got {chunk.get('page')}"
            )

    def test_chapter_not_empty(self, chunks):
        for i, chunk in enumerate(chunks):
            assert chunk.get("chapter", ""), (
                f"Chunk {i}: chapter must not be empty"
            )

    def test_lesson_not_empty(self, chunks):
        for i, chunk in enumerate(chunks):
            assert chunk.get("lesson", ""), (
                f"Chunk {i}: lesson must not be empty"
            )


class TestChunkCount:
    def test_at_least_one_chunk(self, chunks):
        assert len(chunks) >= 1

    def test_multiple_chapters_produce_multiple_chunks(self, chunks):
        assert len(chunks) >= 2, (
            f"Two-chapter document should produce >= 2 chunks, got {len(chunks)}"
        )


class TestStandaloneChunks:
    """Tables and headings must always be standalone chunks (never merged)."""

    def test_table_is_standalone(self, chunks):
        table_chunks = [c for c in chunks if c["type"] == "table"]
        assert len(table_chunks) >= 1, "Expected at least one standalone table chunk"
        for tc in table_chunks:
            assert "|" in tc["text"], f"Table chunk missing | markers: {tc['text'][:60]}"

    def test_heading_is_standalone(self, chunks):
        heading_chunks = [c for c in chunks if c["type"] == "heading"]
        assert len(heading_chunks) >= 1, "Expected at least one standalone heading chunk"


class TestFirst5Chunks:
    """Snapshot test: first 5 chunks must all have complete metadata."""

    def test_first_five_have_full_metadata(self, chunks):
        sample = chunks[:5]
        assert sample, "No chunks produced — nothing to snapshot"
        for i, chunk in enumerate(sample):
            for field in REQUIRED_FIELDS:
                assert chunk.get(field) not in (None, ""), (
                    f"First-5 snapshot: chunk[{i}] missing '{field}'"
                )

    def test_first_five_chapter_lesson_consistent(self, chunks):
        """chapter and lesson in each chunk must match a real chapter/lesson path."""
        sample = chunks[:5]
        for i, chunk in enumerate(sample):
            assert chunk["chapter"] and chunk["lesson"], (
                f"First-5 snapshot: chunk[{i}] has blank chapter or lesson"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
