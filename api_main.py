"""FastAPI backend service for the EDU2 Educational RAG Parsing Pipeline.

Provides high-performance REST API endpoints to parse Arabic and English textbook PDFs,
fingerprint document structure, generate structured EducationalDocument JSON,
store persistent artifacts, and export the Markdown and metadata EDU-RAG ingests.

Run with:
    uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data.storage import (
    delete_book_record,
    get_book_artifacts,
    get_book_record,
    list_registered_books,
    load_registry,
    save_parsed_document,
)
from parsers.gemini_ocr_parser import openrouter_key_available
from pipeline import run_pipeline
from routing.probe import probe_document
from routing.router import ParserRouter, detect_language_from_content
from schema.models import EducationalDocument

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="EDU2 Educational RAG Parsing API",
    description=(
        "Production-grade pre-RAG document parsing service tailored for Arabic & English textbooks. "
        "Extracts chapter/lesson hierarchy, tables, images, provenance metadata, and RAG-ready Markdown."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend and microservice integration
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------
class ProbeResponse(BaseModel):
    num_pages: int
    is_born_digital: bool
    has_images: bool
    image_count: int
    detected_language: str
    recommended_parser: str
    filename: str


class ParseResponse(BaseModel):
    success: bool
    parser: str
    language: str
    parsing_mode: str
    probe: dict[str, Any]
    quality_report: dict[str, Any]
    educational_document: dict[str, Any]
    raw_markdown: Optional[str] = None
    refinement_report: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    tesseract_available: bool
    openrouter_key_configured: bool
    llamacloud_key_configured: bool
    supported_languages: list[str]
    default_vlm_model: str
    registered_books_count: int


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def _save_upload_temp(upload_file: UploadFile) -> Path:
    """Save an incoming UploadFile to a unique temporary directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="edu2_rag_api_"))
    safe_filename = Path(upload_file.filename or "uploaded.pdf").name
    temp_path = temp_dir / safe_filename
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_path


def _check_tesseract_installed() -> bool:
    """Check if tesseract binary is reachable and set TESSDATA_PREFIX."""
    import pytesseract
    for p in [
        os.getenv("TESSERACT_PATH", "").strip(),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]:
        if p and os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            tessdata = Path(p).parent / "tessdata"
            if tessdata.is_dir() and not os.getenv("TESSDATA_PREFIX"):
                os.environ["TESSDATA_PREFIX"] = str(tessdata)
            return True

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Core Endpoints
# ------------------------------------------------------------------------------
@app.get("/", summary="Service Overview & Console Redirect", tags=["System"])
def root():
    return {
        "service": "EDU2 Educational RAG Parsing API",
        "version": "2.0.0",
        "console": "/console",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "ready",
    }


@app.get("/console", response_class=HTMLResponse, summary="Interactive Web Console", tags=["System"])
def get_console():
    console_path = STATIC_DIR / "console.html"
    if console_path.exists():
        return FileResponse(str(console_path))
    return HTMLResponse("<h1>Console not found in static/</h1>", status_code=404)


@app.get("/api/v1/health", response_model=HealthResponse, summary="Health & Dependency Status", tags=["System"])
def health_check():
    tesseract_ok = _check_tesseract_installed()
    openrouter_ok = openrouter_key_available()
    llama_ok = bool(os.getenv("LLAMA_CLOUD_API_KEY"))
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    books = list_registered_books()

    return HealthResponse(
        status="healthy",
        tesseract_available=tesseract_ok,
        openrouter_key_configured=openrouter_ok,
        llamacloud_key_configured=llama_ok,
        supported_languages=["ar", "en", "mixed"],
        default_vlm_model=model,
        registered_books_count=len(books),
    )


@app.get("/api/v1/parsers", summary="Registered Parsers & Routing Policy", tags=["Routing"])
def list_parsers():
    router = ParserRouter()
    has_vlm = openrouter_key_available()
    return {
        "routing_matrix": {
            "english_born_digital": "LiteParse (Free, Fast, Layout-Preserving)",
            "arabic_born_digital": "OpenRouter Vision OCR (with OPENROUTER_API_KEY) or LiteParse/PyMuPDF (Local)",
            "scanned_any_language": "OpenRouter Vision OCR (with OPENROUTER_API_KEY) or Tesseract (Local)",
            "emergency_fallback": "LlamaParse (with LLAMA_CLOUD_API_KEY)",
        },
        "registered_parsers": list(router._registry.keys()),
        "vlm_promotion_active": has_vlm,
    }


@app.post("/api/v1/probe", response_model=ProbeResponse, summary="Fingerprint PDF Document", tags=["Inspection"])
async def probe_pdf(file: UploadFile = File(..., description="PDF textbook file to inspect")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = _save_upload_temp(file)
    try:
        probe = probe_document(str(temp_path))
        detected_lang = detect_language_from_content(str(temp_path))
        router = ParserRouter()
        selected_parser, _, _ = router.route(str(temp_path))

        return ProbeResponse(
            num_pages=probe.num_pages,
            is_born_digital=probe.is_born_digital,
            has_images=probe.has_images,
            image_count=probe.image_count,
            detected_language=detected_lang,
            recommended_parser=selected_parser.name,
            filename=file.filename,
        )
    finally:
        shutil.rmtree(temp_path.parent, ignore_errors=True)


@app.post("/api/v1/parse", response_model=ParseResponse, summary="Parse PDF into EducationalDocument", tags=["Parsing"])
async def parse_document(
    file: UploadFile = File(..., description="PDF textbook file"),
    language: Optional[str] = Form(None, description="ISO code: 'ar', 'en', or None to auto-detect"),
    include_markdown: bool = Form(True, description="When true, includes raw parser Markdown"),
    parsing_mode: str = Form("auto", description="'auto' or 'standard'"),
    ocr_engine: str = Form("openrouter", description="OCR engine: 'openrouter', 'auto', or 'tesseract'"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = _save_upload_temp(file)
    try:
        result = run_pipeline(
            file_path=str(temp_path),
            language=language,
            include_markdown=include_markdown,
            parsing_mode=parsing_mode,
            ocr_engine=ocr_engine,
        )

        return ParseResponse(
            success=True,
            parser=result["parser"],
            language=result["language"],
            parsing_mode=result["parsing_mode"],
            probe=result["probe"],
            quality_report=result["quality_report"],
            educational_document=result["educational_document"],
            raw_markdown=result.get("parser_markdown") or result.get("raw_markdown"),
            refinement_report=result.get("refinement_report"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline parsing failed: {exc}")
    finally:
        shutil.rmtree(temp_path.parent, ignore_errors=True)


# ------------------------------------------------------------------------------
# Persistent Storage & Registry Endpoints
# ------------------------------------------------------------------------------
@app.post("/api/v1/storage/ingest", summary="Ingest, Parse & Store Document in Registry", tags=["Storage & Library"])
async def ingest_and_store_document(
    file: UploadFile = File(..., description="PDF textbook file to ingest"),
    language: Optional[str] = Form(None, description="ISO code: 'ar', 'en', or None"),
    parsing_mode: str = Form("auto", description="'auto' or 'standard'"),
    ocr_engine: str = Form("openrouter", description="OCR engine: 'openrouter', 'auto', or 'tesseract'"),
    openrouter_api_key: Optional[str] = Form(None, description="Temporary OpenRouter API key"),
    openrouter_model: Optional[str] = Form(None, description="Optional OpenRouter model override"),
):
    """Parses a PDF, stores raw & structured artifacts in data/, and registers it in catalog."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = _save_upload_temp(file)

    # Temporarily inject user-supplied keys and model so cloud parsers are enabled
    env_backups: dict[str, str | None] = {}
    overrides = {
        "OPENROUTER_API_KEY": openrouter_api_key,
        "OPENROUTER_MODEL": openrouter_model,
    }
    for var, val in overrides.items():
        if val and val.strip():
            env_backups[var] = os.environ.get(var)
            os.environ[var] = val.strip()

    try:
        result = run_pipeline(
            file_path=str(temp_path),
            language=language,
            include_markdown=True,
            parsing_mode=parsing_mode,
            ocr_engine=ocr_engine,
        )

        record = save_parsed_document(temp_path, result)
        return {
            "success": True,
            "message": f"Document '{file.filename}' successfully parsed and stored.",
            "record": record,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        # Restore original environment
        for var, prev in env_backups.items():
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
        shutil.rmtree(temp_path.parent, ignore_errors=True)


@app.get("/api/v1/storage/registry", summary="List All Registered Books", tags=["Storage & Library"])
def get_registry():
    """Retrieve all cataloged textbook records in registry."""
    return load_registry()


@app.get("/api/v1/storage/book/{book_id}", summary="Get Book Record & Artifacts", tags=["Storage & Library"])
def get_book_details(book_id: str):
    """Retrieve the EducationalDocument, Markdown, and quality artifacts for a book."""
    artifacts = get_book_artifacts(book_id)
    if not artifacts:
        raise HTTPException(status_code=404, detail=f"Book ID '{book_id}' not found in registry.")
    return artifacts


@app.delete("/api/v1/storage/book/{book_id}", summary="Delete Book from Registry & Storage", tags=["Storage & Library"])
def delete_book(book_id: str):
    """Delete a textbook's stored raw file, parsed artifacts, and registry record."""
    success = delete_book_record(book_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Book ID '{book_id}' not found.")
    return {"success": True, "message": f"Book '{book_id}' and all associated artifacts deleted."}


# Standalone entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FASTAPI_PORT", 8000))
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    uvicorn.run("api_main:app", host=host, port=port, reload=True)
