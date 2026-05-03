import re
from pathlib import Path

import pymupdf4llm
from bs4 import BeautifulSoup
from app.core.config import get_settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def chunk_text(text: str, base_metadata: dict):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    docs = splitter.create_documents([text])

    chunks = []
    current_section = "Introduction"

    for doc in docs:
        content = doc.page_content.strip()

        # detect section
        header_match = re.match(r"^#{1,3}\s+(.+)", content)
        if header_match:
            current_section = header_match.group(1)

        chunks.append({
            "text": content,
            "metadata": {
                **base_metadata,
                "section": current_section
            }
        })

    return chunks