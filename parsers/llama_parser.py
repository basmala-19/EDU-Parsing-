"""
LlamaParser — complex-layout and scanned documents via the LlamaCloud REST API.

Supports language selection ('ar', 'en', 'auto') to optimize multilingual extraction.
API key is loaded automatically from .env or LLAMA_CLOUD_API_KEY environment variable.

Free tier: 1000 pages/day at no cost.
API docs: https://docs.cloud.llamaindex.ai
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from parsers.base import BaseParser

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

_UPLOAD_URL = "https://api.cloud.llamaindex.ai/api/v1/parsing/upload"
_JOB_URL = "https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}"
_RESULT_URL = "https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/markdown"
_POLL_INTERVAL = 2
_MAX_POLLS = 60


class LlamaParser(BaseParser):
    """Parse complex-layout documents via the LlamaCloud REST API with language support."""

    name = "llamaparse"

    def __init__(self, api_key: str | None = None, language: str = "auto") -> None:
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        self.language = language

    def parse(self, file_path: str) -> str:
        """Parse a PDF document and return clean Markdown.

        Args:
            file_path: Path to the PDF document.

        Returns:
            Full Markdown string from LlamaCloud.
        """
        if not self.api_key:
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY is not set. "
                "Register for free at https://cloud.llamaindex.ai"
            )

        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {}
        if self.language and self.language != "auto":
            data["language"] = self.language

        with open(file_path, "rb") as f:
            upload_res = requests.post(
                _UPLOAD_URL,
                files={"file": (Path(file_path).name, f, "application/pdf")},
                data=data,
                headers=headers,
                timeout=60,
            )
        upload_res.raise_for_status()
        job_id = upload_res.json()["id"]

        for _ in range(_MAX_POLLS):
            status_res = requests.get(
                _JOB_URL.format(job_id=job_id),
                headers=headers,
                timeout=30,
            )
            status_res.raise_for_status()
            status = status_res.json().get("status")

            if status == "SUCCESS":
                break
            if status in ("ERROR", "FAILED"):
                raise RuntimeError(
                    f"LlamaCloud parsing job {job_id} failed: "
                    f"{status_res.json().get('error_message')}"
                )
            time.sleep(_POLL_INTERVAL)
        else:
            raise RuntimeError(
                f"LlamaCloud parsing job {job_id} timed out after "
                f"{_MAX_POLLS * _POLL_INTERVAL}s"
            )

        result_res = requests.get(
            _RESULT_URL.format(job_id=job_id),
            headers=headers,
            timeout=60,
        )
        result_res.raise_for_status()
        return result_res.json().get("markdown", "")
