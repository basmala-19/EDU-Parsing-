"""
ParseBench-style evaluation runner.

Runs every document in test_docs/ (or a ParseBench data directory) through the
full pipeline and prints a structured PASS/FAIL report per stage.

Usage:
    # Local test docs only (no download needed):
    PYTHONPATH=. python evaluation/runner.py

    # With ParseBench dataset (downloads ~300 MB on first run):
    PYTHONPATH=. python evaluation/runner.py --parsebench

    # Small test slice (3 files per category):
    PYTHONPATH=. python evaluation/runner.py --parsebench --test

Design:
    - OCR documents are skipped gracefully when no OCR backend is configured.
    - Language is left as None (Language-Agnostic) unless auto-detection is enabled.
    - Quality scores are PASS/FAIL — no numeric thresholds until GT data exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parsers.base import BaseParser
from routing.probe import probe_document
from routing.router import ParserRouter
from educational.rule_based_parser import parse_markdown_to_education
from evaluation.reports import build_quality_report, format_report


# ---------------------------------------------------------------------------
# Fake parsers for local test_docs (no real parser installation required)
# ---------------------------------------------------------------------------


class _FakeAcademicParser(BaseParser):
    """Simulates Docling output for an English academic document (e.g. ParseBench paper)."""

    name = "docling"

    def parse(self, file_path: str) -> str:
        return """\
# 1 Introduction

Machine learning is a subset of artificial intelligence focused on building
systems that learn from data without being explicitly programmed.

## 1.1 Motivation

Traditional rule-based systems do not scale to complex, high-dimensional
problems such as image recognition or natural-language understanding.

# 2 Benchmark

## 2.1 Dataset

The dataset consists of 2 000 human-verified pages spanning insurance,
finance, and government documents.

| Dimension            | Metric               | Pages |
| -------------------- | -------------------- | ----- |
| Tables               | GTRM                 | 503   |
| Charts               | ChartDataPointMatch  | 568   |
| Content Faithfulness | Faithfulness Score   | 506   |

## 2.2 Evaluation Protocol

Each document is evaluated by five independent annotators and adjudicated
by majority vote.
"""


class _FakeArabicParser(BaseParser):
    """Simulates Docling output for an Arabic document — identical structure, different script."""

    name = "docling"

    def parse(self, file_path: str) -> str:
        return """\
# الفصل الأول: مقدمة

التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يهتم ببناء أنظمة تتعلم من البيانات.

## الدافع

الأنظمة القائمة على القواعد التقليدية لا تصلح للمشكلات المعقدة عالية الأبعاد.

# الفصل الثاني: المعيار

## مجموعة البيانات

تتكون مجموعة البيانات من 2000 صفحة تم التحقق منها بشرياً.

| البُعد       | المقياس | الصفحات |
| ------------ | ------- | ------- |
| الجداول      | GTRM    | 503     |
| المحتوى      | درجة    | 506     |
"""


class _FakeOCRParser(BaseParser):
    """Simulates OCR output — flat text, no heading markup (intentional challenge)."""

    name = "ocr_parser"

    def parse(self, file_path: str) -> str:
        return (
            "Introduction\n\n"
            "Machine learning is a subset of artificial intelligence.\n\n"
            "Benchmark\n\n"
            "The dataset consists of 2000 pages."
        )


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

_FAKE_PARSERS: dict[str, BaseParser] = {
    "academic.pdf": _FakeAcademicParser(),
    "arabic_book.pdf": _FakeArabicParser(),
    "scanned.pdf": _FakeOCRParser(),
}

_LOCAL_TEST_MATRIX = [
    ("test_docs/academic.pdf", _FakeAcademicParser(), None),
    ("test_docs/arabic_book.pdf", _FakeArabicParser(), None),
    ("test_docs/scanned.pdf", _FakeOCRParser(), None),
]


def _run_single(
    file_path: str,
    fake_parser: BaseParser | None,
    language: str | None,
    include_chunks: bool = False,
) -> bool:
    """Run one document through the pipeline and print stage results.

    Returns True if all stages passed, False otherwise.
    """
    passed = True
    doc_name = Path(file_path).name

    print(f"\n  {doc_name}")
    try:
        probe = probe_document(file_path)
        img_str = f"images={probe.image_count}" if probe.image_count else "images=0"
        print(
            f"    [+] Probe    pages={probe.num_pages}  born_digital={probe.is_born_digital}"
            f"  {img_str}  complex={probe.likely_has_complex_layout}"
            f"  -> {probe.route}"
        )
    except Exception as exc:
        print(f"    [!] Probe    FAILED: {exc}")
        return False

    try:
        router = ParserRouter()
        if fake_parser is not None:
            router._registry[fake_parser.name] = fake_parser
        selected = router.select(probe)
        print(f"    [+] Router   selected={selected.name}")
    except Exception as exc:
        print(f"    [!] Router   FAILED: {exc}")
        return False

    try:
        markdown = selected.parse(file_path)
        print(f"    [+] Parser   {len(markdown)} chars")
    except NotImplementedError as exc:
        # OCR stub — expected; treat as graceful skip
        print(f"    [-] Parser   SKIP (OCR backend not configured: {exc})")
        return True
    except Exception as exc:
        print(f"    [!] Parser   FAILED: {exc}")
        passed = False
        return passed

    try:
        edoc = parse_markdown_to_education(
            markdown_text=markdown,
            source_file=file_path,
            parser=selected.name,
            language=language,
        )
        chapter_count = len(edoc.chapters)
        element_count = sum(len(el) for ch in edoc.chapters for le in ch.lessons for el in [le.elements])
        print(f"    [+] Edu      chapters={chapter_count}  elements={element_count}  language-agnostic=True")
    except Exception as exc:
        print(f"    [!] Edu      FAILED: {exc}")
        passed = False
        return passed

    try:
        report = build_quality_report(markdown, edoc)
        all_ok = all(v["status"] == "PASS" for v in report.values())
        status_str = "PASS" if all_ok else "FAIL"
        print(f"    [+] Quality  [{status_str}]")
        print(format_report(report, indent=12))
        if not all_ok:
            passed = False
    except Exception as exc:
        print(f"    [!] Quality  FAILED: {exc}")
        passed = False

    if include_chunks:
        try:
            from chunking.chunker import chunk_educational_document
            chunks = chunk_educational_document(edoc)
            print(f"    [+] Chunks   {len(chunks)} chunks produced")
        except Exception as exc:
            print(f"    [!] Chunks   FAILED: {exc}")
            passed = False

    return passed


def run_local(include_chunks: bool = False) -> bool:
    """Run the local test_docs/ matrix."""
    all_passed = True
    for file_path, fake_parser, language in _LOCAL_TEST_MATRIX:
        ok = _run_single(file_path, fake_parser, language, include_chunks)
        all_passed = all_passed and ok
    return all_passed


def run_parsebench(data_dir: Path, include_chunks: bool = False) -> bool:
    """Run available PDFs from a downloaded ParseBench dataset directory."""
    doc_dirs = list(data_dir.glob("docs/*/*.pdf"))
    if not doc_dirs:
        print(f"  No PDF files found under {data_dir}/docs/")
        return False

    all_passed = True
    for pdf_path in sorted(doc_dirs)[:20]:  # cap at 20 docs to avoid long runs
        ok = _run_single(str(pdf_path), None, None, include_chunks)
        all_passed = all_passed and ok
    return all_passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ParseBench-style pipeline evaluation runner"
    )
    parser.add_argument(
        "--parsebench",
        action="store_true",
        help="Run against a ParseBench dataset (downloads if needed)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the small ParseBench test slice (3 files per category)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to ParseBench data directory (default: ./data or ./data/test)",
    )
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Also run the Chunking stage and report chunk count",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 10, "PIPELINE EVALUATION REPORT", "=" * 10)

    if args.parsebench:
        from data.download import download_dataset, default_data_dir

        data_dir = args.data_dir or default_data_dir(test=args.test)
        data_dir = download_dataset(data_dir=data_dir, test=args.test)
        all_passed = run_parsebench(data_dir, include_chunks=args.chunks)
    else:
        all_passed = run_local(include_chunks=args.chunks)

    print()
    print("All Tests Passed" if all_passed else "Some Tests FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
