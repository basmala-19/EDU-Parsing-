"""
Convert an existing educational_document.json (edoc.json) into the Markdown
shape EDU-RAG's document loader/chunker actually expects.

Use this on artifacts you already have (no OpenRouter key / re-parse needed):

    python scripts/edoc_to_rag_markdown.py path/to/*_edoc.json

Writes, next to each input file:
    <name>.rag.md               <- feed this to EDU-RAG, not the edoc.json
    <name>.rag_metadata.json    <- pass as extra_metadata=/curriculum_id= to
                                    IngestionService.ingest() for this file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema.rag_export import educational_document_to_markdown, rag_extra_metadata


def convert(edoc_path: Path) -> tuple[Path, Path]:
    edoc = json.loads(edoc_path.read_text(encoding="utf-8"))
    stem = edoc_path.name
    for suffix in ("_edoc.json", ".edoc.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    md_path = edoc_path.with_name(f"{stem}.rag.md")
    meta_path = edoc_path.with_name(f"{stem}.rag_metadata.json")

    md_path.write_text(educational_document_to_markdown(edoc), encoding="utf-8")
    meta_path.write_text(
        json.dumps(rag_extra_metadata(edoc), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md_path, meta_path


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            print(f"skip (not found): {path}")
            continue
        md_path, meta_path = convert(path)
        print(f"{path} -> {md_path.name}, {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
