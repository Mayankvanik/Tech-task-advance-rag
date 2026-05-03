import re
from pathlib import Path

import pymupdf4llm
from bs4 import BeautifulSoup
from app.core.config import get_settings

settings = get_settings()

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".html", ".htm", ".txt"}


async def parse_document(file_path: str) -> list[dict]:
    """Parse a document and return list of {text, metadata} dicts."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        # pymupdf4llm: converts PDF → clean markdown (preserves headers, tables, code blocks)
        full_text = pymupdf4llm.to_markdown(str(path))

    elif ext in {".html", ".htm"}:
        raw_html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "head"]):
            tag.decompose()
        full_text = soup.get_text(separator="\n")

    else:
        # .md / .txt
        full_text = path.read_text(encoding="utf-8")

    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    metadata = extract_metadata(full_text, path.name, ext)
    chunks = chunk_text(full_text, metadata)
    return chunks


def extract_metadata(text: str, filename: str, ext: str) -> dict:
    """Extract doc type, version, sections, and code block presence."""
    version_match = re.search(r"v(?:ersion\s*)?(\d+\.\d+[\.\d]*)", text, re.IGNORECASE)
    version = version_match.group(1) if version_match else "unknown"

    # Markdown-style headers (also present in pymupdf4llm output)
    headers = re.findall(r"^#{1,2}\s+(.+)$", text, re.MULTILINE)

    has_code = bool(re.search(r"```[\s\S]*?```|`[^`]+`", text))

    doc_type_map = {".pdf": "pdf", ".md": "markdown", ".html": "html", ".htm": "html", ".txt": "text"}

    return {
        "source": filename,
        "doc_type": doc_type_map.get(ext, "unknown"),
        "version": version,
        "sections": headers[:10],
        "has_code": has_code,
    }


def chunk_text(text: str, base_metadata: dict, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Chunk text with overlap, tagging section headers per chunk."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current_chunk = []
    current_len = 0
    current_section = "Introduction"

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        header_match = re.match(r"^#{1,3}\s+(.+)$", para)
        if header_match:
            current_section = header_match.group(1)

        words = para.split()
        if current_len + len(words) > chunk_size:
            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": {**base_metadata, "section": current_section},
                })
                current_chunk = current_chunk[-overlap:]
                current_len = len(current_chunk)

        current_chunk.extend(words)
        current_len += len(words)

    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "metadata": {**base_metadata, "section": current_section},
        })

    return chunks