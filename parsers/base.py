"""
BaseParser interface.

Any parser (Docling, LlamaParse, OCR, MinerU, …) must subclass BaseParser
and implement parse() only. The Router knows nothing about parser internals —
it interacts exclusively through this interface.
"""
from abc import ABC, abstractmethod


class BaseParser(ABC):
    name: str = "base"

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parse a document and return clean Markdown.

        The output must preserve heading structure (# / ## / ###) and tables
        so the Educational Parser can infer document structure reliably.

        Returns:
            Markdown string.
        """
        raise NotImplementedError
