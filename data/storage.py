"""Data storage and registry manager for the EDU2 Educational RAG Parsing Pipeline.

Manages persistent storage of:
- Raw input PDFs in `data/raw/`
- Processed artifacts in `data/parsed/<book_id>/` (EducationalDocument, Markdown, Quality Report)
- Central catalog in `data/ingest_registry.json`
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from schema.rag_export import educational_document_to_markdown, rag_extra_metadata

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
REGISTRY_FILE = DATA_DIR / "ingest_registry.json"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(file_path: str | Path) -> str:
    """Calculate SHA256 hash of a file for unique identification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_registry() -> dict[str, Any]:
    """Load the central ingest registry catalog."""
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(registry: dict[str, Any]) -> None:
    """Save the central ingest registry catalog."""
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def save_parsed_document(
    file_path: str | Path,
    parse_result: dict[str, Any],
    custom_title: Optional[str] = None,
) -> dict[str, Any]:
    """Save raw PDF and all parsed artifacts into structured folders and update registry.

    Args:
        file_path: Path to the input PDF.
        parse_result: Dict returned by `pipeline.run_pipeline()`.
        custom_title: Optional title override.

    Returns:
        The registry record created/updated for this document.
    """
    file_path = Path(file_path)
    file_sha256 = compute_file_sha256(file_path)
    book_id = file_sha256[:16]

    # 1. Save raw PDF copy if not already in raw/
    raw_dest = RAW_DIR / f"{book_id}_{file_path.name}"
    if not raw_dest.exists():
        shutil.copy2(file_path, raw_dest)

    # 2. Prepare parsed artifacts folder
    book_parsed_dir = PARSED_DIR / book_id
    book_parsed_dir.mkdir(parents=True, exist_ok=True)

    edoc = parse_result.get("educational_document", {})
    markdown = parse_result.get("parser_markdown") or parse_result.get("raw_markdown", "")
    probe = parse_result.get("probe", {})
    quality_report = parse_result.get("quality_report", {})

    # Save individual artifact files
    with open(book_parsed_dir / "educational_document.json", "w", encoding="utf-8") as f:
        json.dump(edoc, f, ensure_ascii=False, indent=2)

    with open(book_parsed_dir / "raw_markdown.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    with open(book_parsed_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    # RAG-ready artifact: this is what should actually be handed to EDU-RAG's
    # ingestion (a .md file), not educational_document.json -- see
    # schema/rag_export.py for why the raw edoc.json trips up EDU-RAG's
    # generic JSON loader (one LoadedDocument per element, metadata dropped).
    # rag_extra_metadata.json holds the document-level fields (subject/grade/
    # language/document_title) to pass as extra_metadata=/curriculum_id= to
    # IngestionService.ingest() alongside this file, since they intentionally
    # aren't re-derived from the markdown body itself.
    if edoc:
        rag_markdown = educational_document_to_markdown(edoc)
        with open(book_parsed_dir / "document.rag.md", "w", encoding="utf-8") as f:
            f.write(rag_markdown)
        with open(book_parsed_dir / "rag_extra_metadata.json", "w", encoding="utf-8") as f:
            json.dump(rag_extra_metadata(edoc), f, ensure_ascii=False, indent=2)

    with open(book_parsed_dir / "probe.json", "w", encoding="utf-8") as f:
        json.dump(probe, f, ensure_ascii=False, indent=2)

    # 3. Build document metadata for registry
    title = custom_title or edoc.get("document_title") or edoc.get("title") or file_path.stem.replace("_", " ")
    chapters = edoc.get("chapters", [])
    total_elements = sum(
        len(lesson.get("elements", []))
        for ch in chapters
        for lesson in ch.get("lessons", [])
    )

    registry_record = {
        "book_id": book_id,
        "file_sha256": file_sha256,
        "file_name": file_path.name,
        "raw_file_path": str(raw_dest),
        "parsed_dir": str(book_parsed_dir),
        "document_title": title,
        "subject": edoc.get("subject") or "General",
        "grade": edoc.get("grade") or "Unspecified",
        "term": edoc.get("term"),
        "language": parse_result.get("language", "unknown"),
        "parser_used": parse_result.get("parser", "unknown"),
        "parsing_mode": parse_result.get("parsing_mode", "auto"),
        "num_pages": probe.get("num_pages", 1),
        "is_born_digital": probe.get("is_born_digital", True),
        "has_images": probe.get("has_images", False),
        "image_count": probe.get("image_count", 0),
        "chapter_count": len(chapters),
        "total_elements": total_elements,
        "quality_status": "PASS" if all(v.get("status") == "PASS" for v in quality_report.values()) else "WARN",
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 4. Update ingest registry catalog
    registry = load_registry()
    registry[book_id] = registry_record
    save_registry(registry)

    return registry_record


def get_book_record(book_id: str) -> Optional[dict[str, Any]]:
    """Retrieve metadata record for a specific book ID."""
    registry = load_registry()
    return registry.get(book_id)


def get_book_artifacts(book_id: str) -> Optional[dict[str, Any]]:
    """Load all stored artifacts for a specific book ID."""
    record = get_book_record(book_id)
    if not record:
        return None

    book_dir = Path(record.get("parsed_dir", PARSED_DIR / book_id))
    if not book_dir.exists():
        return None

    artifacts: dict[str, Any] = {"record": record}

    if (book_dir / "educational_document.json").exists():
        with open(book_dir / "educational_document.json", "r", encoding="utf-8") as f:
            artifacts["educational_document"] = json.load(f)

    if (book_dir / "raw_markdown.md").exists():
        with open(book_dir / "raw_markdown.md", "r", encoding="utf-8") as f:
            artifacts["raw_markdown"] = f.read()

    if (book_dir / "document.rag.md").exists():
        with open(book_dir / "document.rag.md", "r", encoding="utf-8") as f:
            artifacts["rag_markdown"] = f.read()

    if (book_dir / "rag_extra_metadata.json").exists():
        with open(book_dir / "rag_extra_metadata.json", "r", encoding="utf-8") as f:
            artifacts["rag_extra_metadata"] = json.load(f)

    if (book_dir / "quality_report.json").exists():
        with open(book_dir / "quality_report.json", "r", encoding="utf-8") as f:
            artifacts["quality_report"] = json.load(f)

    if (book_dir / "probe.json").exists():
        with open(book_dir / "probe.json", "r", encoding="utf-8") as f:
            artifacts["probe"] = json.load(f)

    return artifacts


def list_registered_books() -> list[dict[str, Any]]:
    """Return list of all registered books ordered by most recently processed."""
    registry = load_registry()
    return sorted(registry.values(), key=lambda r: r.get("processed_at", ""), reverse=True)


def delete_book_record(book_id: str) -> bool:
    """Delete book artifacts, raw file, and registry entry."""
    registry = load_registry()
    if book_id not in registry:
        return False

    record = registry.pop(book_id)
    save_registry(registry)

    # Clean parsed directory
    parsed_path = Path(record.get("parsed_dir", PARSED_DIR / book_id))
    if parsed_path.exists():
        shutil.rmtree(parsed_path, ignore_errors=True)

    # Clean raw file
    raw_path = Path(record.get("raw_file_path", ""))
    if raw_path.exists():
        raw_path.unlink(missing_ok=True)

    return True
