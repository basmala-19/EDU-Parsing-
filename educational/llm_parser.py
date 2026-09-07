"""
Educational Parser — Version 2 (LLM-based, language-agnostic).

Difference from rule_based_parser.py:
  V1 classifies elements purely from Markdown structure:  # → heading, | → table.
  V2 sends the Markdown to an LLM (e.g. Qwen2.5) to classify each segment into:
  example / exercise / definition / concept — not just heading / paragraph.

This is a typed placeholder. The Router and pipeline are already ready to accept
it without changes to any other module — just pass an instance to the pipeline
instead of using parse_markdown_to_education().
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pymupdf as fitz
from PIL import Image

from schema.models import EducationalDocument, Element, ElementType

EDUCATIONAL_EXTRACTION_PROMPT = """\
Convert this lesson text into a structured educational JSON.
Classify each text segment as one of: heading, paragraph, definition, example, exercise.
Preserve the original language — do not translate.

Text:
{markdown_chunk}

Return only valid JSON matching this schema:
{"type": "...", "text": "..."}
"""


class LLMEducationalParser:
    """LLM-based educational parser — requires a configured LLM client.

    Not yet implemented. This placeholder documents the planned V2 interface
    so the team knows what to build next and in which module.

    Suggested backend: Qwen2.5-7B via Ollama (local) or a cloud API.
    """

    def __init__(self, model_client=None) -> None:
        self.model_client = model_client

    def parse(
        self,
        markdown_text: str,
        source_file: str,
        language: str | None = None,
    ) -> EducationalDocument:
        raise NotImplementedError(
            "V2 LLM parser is not implemented yet. "
            "Wire up a model client (Qwen2.5 / Ollama / API) and implement "
            "this method. Return type must match rule_based_parser so the "
            "chunker and evaluation layer require no changes."
        )


QWEN_REFINEMENT_PROMPT = """You are refining one already-parsed textbook element.
Use only visible evidence. Return one JSON object and no markdown:
{"type":"image|table|equation|heading|paragraph", "caption":"", "description":"",
 "educational_role":"", "headers":[], "rows":[], "latex":"", "text":""}
Do not invent facts. Keep text empty unless the supplied original text is empty or a placeholder.
"""


@dataclass
class RefinementReport:
    candidates: int = 0
    calls: int = 0
    applied: int = 0
    rejected: int = 0
    latency_seconds: float = 0.0


class QwenVLRefiner:
    """Opt-in, local Qwen2.5-VL refinement for uncertain elements only.

    The model is deliberately loaded lazily.  It never edits useful parser
    text: descriptions, structured rows, LaTeX, and type are additive fields.
    """

    PLACEHOLDERS = {"", "[image: docling placeholder]", "[image: image]"}

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        model_client: Callable[[Image.Image, str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_client = model_client
        self._model = None
        self._processor = None

    @classmethod
    def needs_refinement(cls, element: Element) -> bool:
        text = (element.text or "").strip().lower()
        return (
            element.metadata.extra.get("needs_review") is True
            or (element.type == ElementType.IMAGE and text in cls.PLACEHOLDERS)
            or (element.type == ElementType.TABLE and not element.rows)
            or (element.type == ElementType.HEADING and (not element.text or not element.level))
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen refinement needs the optional local dependencies. "
                "Install requirements-qwen.txt and use a GPU runtime."
            ) from exc
        # Colab T4 GPUs have ~15GB VRAM.  4-bit weights leave room for image
        # tokens and generation; full precision can OOM before page one.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            # Prevent downloading 7.5GB model weights on CPU if not explicitly enabled
            if os.getenv("ALLOW_LOCAL_QWEN_DOWNLOAD", "0") != "1":
                raise RuntimeError(
                    "Local Qwen model download skipped on CPU to prevent downloading 7.5GB. "
                    "Set ALLOW_LOCAL_QWEN_DOWNLOAD=1 to force local CPU loading, or configure OPENROUTER_API_KEY."
                )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        quantization = BitsAndBytesConfig(load_in_4bit=True) if torch.cuda.is_available() else None
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization,
        )

    def _generate(self, image: Image.Image, prompt: str) -> str:
        if self.model_client:
            return self.model_client(image, prompt)
        self._load()
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=512)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    @staticmethod
    def _page_image(pdf_path: str, page: int, bbox: list[float] | None, scale: float = 1.0) -> Image.Image:
        document = fitz.open(pdf_path)
        try:
            source_page = document[page - 1]
            clip = fitz.Rect(bbox) if bbox and len(bbox) == 4 else None
            pix = source_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            image = Image.open(__import__("io").BytesIO(pix.tobytes("png"))).convert("RGB")
            image.thumbnail((1280, 1280))
            return image
        finally:
            document.close()

    @staticmethod
    def _json_response(response: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE).strip()
        return json.loads(cleaned)

    @staticmethod
    def _apply(element: Element, result: dict) -> bool:
        original = (element.text or "").strip()
        candidate_text = (result.get("text") or "").strip()
        placeholder = original.lower() in QwenVLRefiner.PLACEHOLDERS
        # Existing text is immutable.  A response trying to alter it is
        # rejected instead of silently hallucinating a replacement.
        if original and not placeholder and candidate_text and candidate_text != original:
            return False
        proposed_type = result.get("type")
        if proposed_type in {member.value for member in ElementType}:
            element.type = ElementType(proposed_type)
        if placeholder and candidate_text:
            element.text = candidate_text
        extra = element.metadata.extra
        for key in ("caption", "description", "educational_role", "headers", "latex"):
            if result.get(key):
                extra[key] = result[key]
        if result.get("rows") and isinstance(result["rows"], list):
            element.rows = result["rows"]
            element.format = "rows"
        extra["refined_by"] = "qwen"
        extra["refinement_applied"] = "selective local VLM refinement"
        return True

    def refine(self, document: EducationalDocument, pdf_path: str, max_elements: int = 20) -> dict:
        report = RefinementReport()
        started = time.perf_counter()
        elements = [element for element, _, _ in document.all_elements() if self.needs_refinement(element)]
        report.candidates = len(elements)
        for element in elements[:max_elements]:
            image = self._page_image(pdf_path, element.metadata.page, element.metadata.bbox)
            prompt = QWEN_REFINEMENT_PROMPT + f"\nOriginal type: {element.type.value}\nOriginal text: {element.text or ''}"
            report.calls += 1
            try:
                response = self._json_response(self._generate(image, prompt))
                if self._apply(element, response):
                    report.applied += 1
                else:
                    report.rejected += 1
            except Exception:
                report.rejected += 1
        report.latency_seconds = round(time.perf_counter() - started, 3)
        return asdict(report)


QWEN_PAGE_PARSE_PROMPT = """Convert this single educational textbook page to faithful Markdown.
Preserve reading order and all visible text. Describe meaningful diagrams, charts,
and instructional images in concise brackets such as [Image: ...]. Keep tables as
Markdown pipe tables when their cells are visible. Do not invent missing text.
Return Markdown only; no explanation or JSON wrapper.
"""


class QwenPageParser:
    """High-fidelity local VLM parser for visual textbook pages.

    This is intentionally a separate *mode*, rather than a hidden fallback:
    it renders every page and is slower/costlier than the standard Docling path.
    """

    name = "qwen_vl"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", model_client=None) -> None:
        if model_client is None:
            from parsers.gemini_ocr_parser import openrouter_key_available
            if openrouter_key_available():
                try:
                    from parsers.openrouter_vlm_client import OpenRouterVLMClient
                    model_client = OpenRouterVLMClient()
                except Exception:
                    pass
        self.refiner = QwenVLRefiner(model_id=model_id, model_client=model_client)
        self.last_page_markers: list[int] = []

    def parse(self, file_path: str) -> str:
        pdf = fitz.open(file_path)
        try:
            page_count = len(pdf)
        finally:
            pdf.close()
        self.last_page_markers = []
        pages = []
        for page_number in range(1, page_count + 1):
            image = self.refiner._page_image(file_path, page_number, None, scale=1.0)
            markdown = self.refiner._generate(image, QWEN_PAGE_PARSE_PROMPT).strip()
            pages.append(f"<!-- page: {page_number} -->\n{markdown}")
            self.last_page_markers.append(page_number)
        return "\n\n".join(pages)
