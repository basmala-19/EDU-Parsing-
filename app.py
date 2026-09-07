"""Streamlit UI for reviewing the EDU2 parser output.

Run with: ``streamlit run app.py``.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pymupdf as fitz
import streamlit as st

from pipeline import run_pipeline


PAGE_MARKER_RE = re.compile(r"<!--\s*page\s*:\s*(\d+)\s*-->", re.I)
ENGINE_LABELS = {
    "gemini": "Gemini OCR",
    "auto": "Automatic routing",
    "ocrspace": "OCR.space",
    "tesseract": "Tesseract (local)",
}


def _save_upload(uploaded_file) -> Path:
    destination = Path(tempfile.gettempdir()) / "edu2_uploads"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / uploaded_file.name
    path.write_bytes(uploaded_file.getvalue())
    return path


def _markdown_for_page(markdown: str, page: int) -> str | None:
    markers = list(PAGE_MARKER_RE.finditer(markdown or ""))
    if not markers:
        return None
    for i, match in enumerate(markers):
        if int(match.group(1)) == page:
            end = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
            return markdown[match.end() : end].strip()
    return "_No Markdown segment was emitted for this page._"


def _elements_for_page(document: dict, page: int) -> list[dict]:
    elements = []
    for chapter in document.get("chapters", []):
        for lesson in chapter.get("lessons", []):
            for element in lesson.get("elements", []):
                if element.get("metadata", {}).get("page") == page:
                    elements.append({"chapter": chapter.get("title"), "lesson": lesson.get("title"), **element})
    return elements


def _elements_as_markdown(document: dict, page: int) -> str:
    parts = []
    for element in _elements_for_page(document, page):
        text = (element.get("text") or "").strip()
        if element.get("type") == "heading":
            parts.append(f"{'#' * min(element.get('level') or 2, 6)} {text}")
        elif text:
            parts.append(text)
    return "\n\n".join(parts)


def _page_markdown(result: dict, document: dict, markdown: str, page: int) -> str:
    extracted = _markdown_for_page(markdown, page)
    if extracted:
        return extracted
    page_text = result.get("page_text", {})
    fallback = page_text.get(page) or page_text.get(str(page))
    return str(fallback) if fallback else (_elements_as_markdown(document, page) or "_No page-scoped content._")


def _render_page(pdf_path: str, page_number: int) -> bytes:
    pdf = fitz.open(pdf_path)
    try:
        return pdf[page_number - 1].get_pixmap(dpi=150, alpha=False).tobytes("png")
    finally:
        pdf.close()


st.set_page_config(page_title="EDU2 Parse Studio", page_icon="📘", layout="wide")
st.title("📘 EDU2 Parse Studio")

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF textbook", type=["pdf"])
    engine = st.selectbox("OCR Engine", list(ENGINE_LABELS), index=0, format_func=lambda key: ENGINE_LABELS[key])
    parsing_mode = st.selectbox("Parsing mode", ["auto", "standard"], index=0)
    run = st.button("Run", type="primary", width="stretch", disabled=uploaded is None)

if run and uploaded is not None:
    file_path = _save_upload(uploaded)
    with st.spinner(f"Running {ENGINE_LABELS[engine]}…"):
        try:
            st.session_state["parse_result"] = run_pipeline(
                str(file_path),
                include_markdown=True,
                parsing_mode=parsing_mode,
                ocr_engine=engine,
            )
            st.session_state["pdf_path"] = str(file_path)
            st.session_state["pdf_name"] = uploaded.name
        except Exception as exc:
            st.exception(exc)

pdf_path = st.session_state.get("pdf_path")
result = st.session_state.get("parse_result")

if result and pdf_path:
    probe = result["probe"]
    document = result["educational_document"]
    markdown = result.get("raw_markdown", result.get("parser_markdown", ""))
    page_count = max(1, int(probe.get("num_pages", 1)))

    with st.sidebar:
        st.success(f"Parser: {result.get('parser', '—')}")
        st.write(f"Engine: `{result.get('ocr_engine', '—')}`")
        st.write(f"Device: `{result.get('device', 'local')}`")
        page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
        stem = Path(st.session_state["pdf_name"]).stem
        st.download_button("Download EducationalDocument JSON", json.dumps(document, ensure_ascii=False, indent=2), file_name=f"{stem}_edoc.json", mime="application/json", width="stretch")
        st.download_button("Download raw Markdown", markdown, file_name=f"{stem}.md", mime="text/markdown", width="stretch")

    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.subheader(f"PDF · page {page} of {page_count}")
        st.image(_render_page(pdf_path, int(page)), width="stretch")
    with right:
        tabs = st.tabs(["Markdown", "JSON", "Quality", "OCR Profile"])
        with tabs[0]:
            st.markdown(_page_markdown(result, document, markdown, int(page)))
        with tabs[1]:
            st.json({"page": int(page), "elements": _elements_for_page(document, int(page))}, expanded=False)
        with tabs[2]:
            for metric, value in result.get("quality_report", {}).items():
                st.write(f"{'✅' if value.get('status') == 'PASS' else '⚠️'} **{metric}** — {value.get('detail', '')}")
        with tabs[3]:
            st.json(result.get("ocr_profile", {}), expanded=False)

else:
    st.info("Upload a PDF, choose an OCR engine, then click Run.")
