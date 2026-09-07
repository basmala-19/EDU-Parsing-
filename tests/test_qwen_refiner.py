"""Guardrails for selective local VLM refinement."""

from educational.llm_parser import QwenVLRefiner
from schema.models import Element, ElementMetadata, ElementType


def _element(kind: ElementType, text: str | None) -> Element:
    return Element(type=kind, text=text, metadata=ElementMetadata(page=1, parser="docling"))


def test_existing_parser_text_cannot_be_silently_replaced():
    element = _element(ElementType.PARAGRAPH, "Original textbook sentence.")
    applied = QwenVLRefiner._apply(element, {"type": "paragraph", "text": "Invented replacement."})

    assert applied is False
    assert element.text == "Original textbook sentence."


def test_placeholder_image_can_gain_local_caption_and_provenance():
    element = _element(ElementType.IMAGE, "[Image: Docling placeholder]")
    applied = QwenVLRefiner._apply(
        element,
        {"type": "image", "text": "A diagram of fractions", "caption": "Fraction diagram"},
    )

    assert applied is True
    assert element.text == "A diagram of fractions"
    assert element.metadata.extra["refined_by"] == "qwen"


def test_only_flagged_elements_are_candidates():
    ordinary = _element(ElementType.PARAGRAPH, "Normal parsed text.")
    uncertain = _element(ElementType.TABLE, "| malformed |")

    assert QwenVLRefiner.needs_refinement(ordinary) is False
    assert QwenVLRefiner.needs_refinement(uncertain) is True
