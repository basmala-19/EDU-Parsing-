# 📚 EDU2 Educational RAG Parsing Pipeline
> **High-Fidelity Pre-RAG Document Parsing & Semantic Chunking for Arabic and English Educational Textbooks.**

---

## 🏛️ Clean Architecture Overview

The codebase is organized into clean, single-responsibility layers:

```
edu2/
├── routing/                  # Layer 1: Document Fingerprinting & Intelligent Routing
│   ├── probe.py              # PDF analysis (pages, visual density, digital vs scan)
│   └── router.py             # Content-based language detection & parser chain builder
│
├── parsers/                  # Layer 2: Pluggable Parsing & OCR Engines
│   ├── base.py               # Abstract BaseParser interface
│   ├── liteparse_parser.py   # Fast spatial parser (Default for Born-Digital English)
│   ├── docling_parser.py     # Local layout-aware deep parser
│   ├── ocr_parser.py         # Local Tesseract OCR (with True Otsu Binarization)
│   ├── gemini_ocr_parser.py  # VLM Parser (Gemini 3 Flash / Qwen-VL for Arabic & Scans)
│   ├── openrouter_vlm_client.py # Cloud VLM client
│   └── llama_parser.py       # Cloud emergency fallback parser
│
├── educational/              # Layer 3: Educational Structure & Semantic Extraction
│   ├── rule_based_parser.py  # Converts raw Markdown into structured EducationalDocument
│   └── llm_parser.py         # High-fidelity visual parser & Qwen-VL element refiner
│
├── schema/                   # Layer 4: Strongly-Typed Data Models
│   └── models.py             # Pydantic schemas (EducationalDocument, Chapter, Lesson, Element)
│
├── chunking/                 # Layer 5: RAG Semantic Chunking
│   └── chunker.py            # Hierarchical chunking retaining Chapter, Lesson & Page Provenance
│
├── evaluation/               # Layer 6: Quality Metrics & Assurance
│   ├── metrics.py            # Precision, Recall, OCR confidence, and schema validation
│   └── reports.py            # Quality report generator
│
├── pipeline.py               # Core Orchestrator (Single entry point for the pipeline)
├── api_main.py               # FastAPI REST Microservice (Ready for RAG integration)
├── app.py                    # Streamlit Visual Review Studio (Page-sync viewer)
├── environment.yml           # Unified Conda Environment Specification
├── requirements.txt          # Unified Pip Requirements Specification
└── .env.example              # Configuration Template
```

---

## 🚀 Routing Policy (Matrix 2×2)

The parser chain is selected dynamically from the document's internal structure:

| Document Type | Free / Local Chain | With `OPENROUTER_API_KEY` (Recommended) |
|---|---|---|
| **English Born-Digital** | `LiteParse` (Fast, 100% accurate, Free) | `LiteParse` (Retained as free & fastest) |
| **Arabic Born-Digital** | `LiteParse` → `Docling` → `Tesseract` | **`GeminiOCRParser`** (Bypasses BiDi letter scrambling) |
| **Scanned / Images (Any Lang)** | `Tesseract` (Local OCR) | **`GeminiOCRParser`** (Accurately parses complex diagrams) |
| **Emergency Fallback** | `LlamaParse` (Requires `LLAMA_CLOUD_API_KEY`) | `LlamaParse` |

---

## Setup and Running Guide

### 1. Install the Environment

#### Option A: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate edu2-rag
```

#### Option B: Python Virtual Environment
```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

Copy `.env.example` to `.env`, then add any optional API keys:

```bash
# Windows PowerShell:
Copy-Item .env.example .env
# Linux/macOS:
cp .env.example .env
```

Edit `.env`:
```env
# Optional but recommended for Arabic & Scans:
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash

# Optional emergency fallback:
LLAMA_CLOUD_API_KEY=your_llamacloud_key_here
```

---

### 3. Start the FastAPI Backend

Run the REST API server to integrate with your RAG pipeline or backend:

```bash
uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: Open [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

#### API Endpoints:
- `POST /api/v1/parse`: Upload PDF → Returns `EducationalDocument` JSON + Quality Report + Markdown.
- `POST /api/v1/parse-chunks`: **Direct RAG ingestion** → Returns ready-to-embed semantic chunks with metadata (`chapter`, `lesson`, `page`, `token_count`).
- `POST /api/v1/probe`: Quick document fingerprinting.

---

### 4. Start the Streamlit Review Studio

Launch the interactive review studio to visually inspect PDF pages synchronized with Markdown and RAG Chunks:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 5. Run the Pipeline from the CLI

```bash
python pipeline.py "path/to/book.pdf" --chunks
```

#### Python Code Integration:
```python
from pipeline import run_pipeline

# Run full pipeline with RAG chunking
result = run_pipeline(
    file_path="biology_grade10.pdf",
    include_chunks=True,
    include_markdown=True,
    parsing_mode="auto",
)

# 1. Structured Educational Document
edoc = result["educational_document"]
print(f"Title: {edoc['title']}, Chapters: {len(edoc['chapters'])}")

# 2. Semantic RAG Chunks ready for Vector DB
chunks = result["chunks"]
for chunk in chunks[:3]:
    print(f"[{chunk['chapter']} > {chunk['lesson']} (p.{chunk['page']})]: {chunk['text'][:100]}...")
```

---

### 6. Run the Test Suite

```bash
python -m pytest -q
```
The test suite covers language routing, image extraction, bidirectional text handling, and chunking integrity.
