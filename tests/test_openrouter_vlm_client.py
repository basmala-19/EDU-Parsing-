"""Guardrails for the OpenRouter cloud VLM client."""

import pytest
from PIL import Image

from parsers.openrouter_vlm_client import DEFAULT_MODEL, OpenRouterVLMClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _capture(monkeypatch, response: _FakeResponse):
    """Patch requests.post and return a list that collects (args, kwargs)."""
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr("parsers.openrouter_vlm_client.requests.post", fake_post)
    return calls


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenRouterVLMClient()


def test_default_model_is_gemini_flash(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    client = OpenRouterVLMClient()
    assert client.model == DEFAULT_MODEL == "google/gemini-3-flash-preview"


def test_env_model_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-3-pro-preview")
    client = OpenRouterVLMClient()
    assert client.model == "google/gemini-3-pro-preview"


def test_call_sends_image_and_prompt_and_returns_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    response = _FakeResponse(200, {"choices": [{"message": {"content": "extracted table text"}}]})
    calls = _capture(monkeypatch, response)

    client = OpenRouterVLMClient()
    image = Image.new("RGB", (10, 10), color="white")
    result = client(image, "Extract this table.")

    assert result == "extracted table text"
    assert len(calls) == 1
    _, kwargs = calls[0]
    payload = kwargs["json"]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["messages"][0]["content"][0]["type"] == "image_url"
    assert payload["messages"][0]["content"][1]["text"] == "Extract this table."
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_non_200_status_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _FakeResponse(429, {"error": "rate limited"})
    _capture(monkeypatch, response)

    client = OpenRouterVLMClient()
    image = Image.new("RGB", (10, 10), color="white")
    with pytest.raises(RuntimeError, match="429"):
        client(image, "prompt")


def test_unexpected_response_shape_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _FakeResponse(200, {"unexpected": "shape"})
    _capture(monkeypatch, response)

    client = OpenRouterVLMClient()
    image = Image.new("RGB", (10, 10), color="white")
    with pytest.raises(RuntimeError, match="Unexpected OpenRouter response shape"):
        client(image, "prompt")
