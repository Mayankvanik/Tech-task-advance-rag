import re
from pathlib import Path
from llama_parse import LlamaParse
from app.core.config import get_settings

settings = get_settings()

# LlamaParser instance (supports PDF, MD, HTML)
parser = LlamaParse(
    api_key=settings.llama_cloud_api_key,
    result_type="markdown",
    verbose=False,
)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".html", ".htm", ".txt"}


async def parse_document(file_path: str) -> list[dict]:
    """Parse a document and return list of {text, metadata} dicts."""
    path = Path(file_path)
    ext = path.suffix.lower()
    print("bfooooooooooooooooooooooo")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext in {".pdf", ".html", ".htm"}:
        docs = await parser.aload_data(str(path))
        full_text = "\n\n".join(d.text for d in docs)
    else:
        full_text = path.read_text(encoding="utf-8")

    metadata = extract_metadata(full_text, path.name, ext)
    chunks = chunk_text(full_text, metadata)
    return chunks


def extract_metadata(text: str, filename: str, ext: str) -> dict:
    """Extract doc type, version, sections, and code block presence."""
    # Detect version pattern like v1.0, version 2.3
    version_match = re.search(r"v(?:ersion\s*)?(\d+\.\d+[\.\d]*)", text, re.IGNORECASE)
    version = version_match.group(1) if version_match else "unknown"

    # Extract top-level headers (Markdown style)
    headers = re.findall(r"^#{1,2}\s+(.+)$", text, re.MULTILINE)

    # Detect code blocks
    has_code = bool(re.search(r"```[\s\S]*?```|`[^`]+`", text))

    doc_type_map = {".pdf": "pdf", ".md": "markdown", ".html": "html", ".htm": "html", ".txt": "text"}

    return {
        "source": filename,
        "doc_type": doc_type_map.get(ext, "unknown"),
        "version": version,
        "sections": headers[:10],  # store top 10 section headers
        "has_code": has_code,
    }


def chunk_text(text: str, base_metadata: dict, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Chunk text with overlap, tagging section headers per chunk."""
    # Split on double newlines (paragraphs)
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current_chunk = []
    current_len = 0
    current_section = "Introduction"

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Track section headers
        header_match = re.match(r"^#{1,3}\s+(.+)$", para)
        if header_match:
            current_section = header_match.group(1)

        words = para.split()
        if current_len + len(words) > chunk_size:
            if current_chunk:
                chunk_text_str = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text_str,
                    "metadata": {**base_metadata, "section": current_section},
                })
                # Overlap: keep last `overlap` words
                current_chunk = current_chunk[-overlap:]
                current_len = len(current_chunk)

        current_chunk.extend(words)
        current_len += len(words)

    # Last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "metadata": {**base_metadata, "section": current_section},
        })

    return chunks
