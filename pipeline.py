"""
Pipeline entry point.

Flow:
    PDF / Document
        |
        v
    Probe  (routing/probe.py)      — document fingerprinting
        |
        v
    Router (routing/router.py)     — content-based language detection & parser selection
        |
        v
    Parser (parsers/*.py)          — returns clean Markdown (Docling / Tesseract / LlamaParse fallback)
        |
        v
    Educational Parser             — Markdown → EducationalDocument (Pydantic)
    (educational/rule_based_parser.py)
        |
        v
Usage:
    PYTHONPATH=. python pipeline.py sample_report.pdf
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from routing.router import ParserRouter, detect_language_from_content, detect_language_from_text
from educational.rule_based_parser import parse_markdown_to_education
from evaluation.reports import build_quality_report
from parsers.image_extractor import attach_extracted_images, extract_images_from_pdf


def run_pipeline(
    file_path: str,
    language: str | None = None,
    include_markdown: bool = False,
    parsing_mode: str = "auto",
    ocr_engine: str = "openrouter",
) -> dict:
    """Run the full document pipeline.

    Args:
        file_path:      Path to the input document.
        language:       ISO language code when known ('en', 'ar'). Pass None to auto-detect
                        from actual document content.
        ocr_engine:     Cloud OCR engine selection for scanned documents.
                        'openrouter' (default), 'auto', or 'tesseract'.

    Returns:
        Dict with keys: probe, parser, language, educational_document, quality_report,
        and optionally parser_markdown.
    """
    if language is None:
        language = detect_language_from_content(file_path)

    if parsing_mode not in {"auto", "standard"}:
        raise ValueError("parsing_mode must be 'auto' or 'standard'")

    router = ParserRouter(ocr_engine=ocr_engine)
    _, probe, detected_language = router.route(file_path)
    language = language or detected_language

    # The router chooses a content-aware parser chain without loading any local
    # vision model. OpenRouter is the cloud VLM path; Tesseract is the CPU fallback.
    selected_mode = "standard"
    candidates = router.build_chain(probe, language)

    failures: list[str] = []
    parser_attempts: list[dict[str, str]] = []
    for selected_parser in candidates:
        try:
            markdown = selected_parser.parse(file_path)
            if markdown.strip():
                parser_attempts.append({"parser": selected_parser.name, "status": "success"})
                break
            detail = "returned empty output"
            failures.append(f"{selected_parser.name}: {detail}")
            parser_attempts.append({"parser": selected_parser.name, "status": "failed", "detail": detail})
        except Exception as exc:
            detail = str(exc)
            failures.append(f"{selected_parser.name}: {detail}")
            parser_attempts.append({"parser": selected_parser.name, "status": "failed", "detail": detail})
    else:
        raise RuntimeError("All parsers failed. " + " | ".join(failures))

    # A scanned PDF has no text layer, so OSD can legitimately return
    # ``unknown`` before OCR.  Infer it again from the actual OCR result so
    # the final JSON and downstream Arabic rules receive the correct language.
    if language == "unknown":
        recovered_language = detect_language_from_text(markdown)
        if recovered_language != "unknown":
            language = recovered_language

    # Keep the stable parser identity during structural parsing.  Gemini OCR
    # emits flat page text, and the EducationalDocument parser uses the
    # ``gemini_ocr`` identity to activate its TOC/Arabic OCR recovery rules.
    # Replacing it with a display string such as ``openrouter (model)`` made
    # those rules silently switch off, producing Untitled chapters.
    parser_label = selected_parser.name

    edoc = parse_markdown_to_education(
        markdown_text=markdown,
        source_file=file_path,
        parser=parser_label,
        language=language,
    )

    images_by_page = extract_images_from_pdf(file_path)
    attach_extracted_images(edoc, images_by_page, parser=parser_label)

    quality_report = build_quality_report(markdown, edoc, expected_page_count=probe.num_pages)

    result: dict = {
        "probe": asdict(probe),
        "parser": selected_parser.name,
        "ocr_engine": ocr_engine,
        "parser_attempts": parser_attempts,
        "language": language,
        "parsing_mode": selected_mode,
        "requested_mode": parsing_mode,
        "educational_document": json.loads(edoc.model_dump_json(exclude_none=True)),
        "quality_report": quality_report,
    }
    if hasattr(selected_parser, "last_page_markers"):
        result["page_markers"] = selected_parser.last_page_markers
    if hasattr(selected_parser, "last_page_text"):
        # LiteParse page text lets the review UI stay page-scoped even though
        # its combined Markdown currently has no <!-- page: N --> markers.
        result["page_text"] = selected_parser.last_page_text
    if hasattr(selected_parser, "last_ocr_profile"):
        result["ocr_profile"] = selected_parser.last_ocr_profile
    if include_markdown:
        result["parser_markdown"] = markdown
        # Clear public name; parser_markdown remains for compatibility.
        result["raw_markdown"] = markdown

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG document pipeline")
    parser.add_argument("file", nargs="?", default="test_docs/academic.pdf")
    parser.add_argument(
        "--language",
        default=None,
        help="ISO language code, e.g. 'en' or 'ar' (default: auto content-based detection)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    res = run_pipeline(args.file, language=args.language, include_markdown=True)

    print(f"Parser      : {res['parser']}")
    for attempt in res["parser_attempts"]:
        suffix = f" ({attempt['detail']})" if "detail" in attempt else ""
        print(f"  - {attempt['parser']}: {attempt['status']}{suffix}")
    print(f"Language    : {res['language']}")
    print(f"Chapters    : {len(res['educational_document']['chapters'])}")
    if "page_markers" in res:
        markers = res["page_markers"]
        preview = markers if len(markers) <= 20 else markers[:10] + ["..."] + markers[-3:]
        print(f"Page markers: {preview}")
    print("Quality     :")
    for metric, val in res["quality_report"].items():
        icon = "+" if val["status"] == "PASS" else "!"
        print(f"  [{icon}] {metric:<28} {val['detail']}")
