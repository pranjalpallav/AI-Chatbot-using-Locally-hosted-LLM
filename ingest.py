"""PDF parsing, cleaning and chunking."""
import re
import fitz  # PyMuPDF


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text.strip()


def load_pdf(source) -> list[dict]:
    """source: file path (str) or raw PDF bytes. Returns [{"page": int, "text": str}]."""
    doc = fitz.open(stream=source, filetype="pdf") if isinstance(source, (bytes, bytearray)) \
        else fitz.open(source)
    pages = []
    with doc:
        for i, page in enumerate(doc, start=1):
            text = clean(page.get_text())
            if text:
                pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict], source: str,
                size: int = 200, overlap: int = 40, min_words: int = 15) -> list[dict]:
    """Split each page into overlapping word windows, keeping file + page metadata."""
    chunks, step = [], size - overlap
    for p in pages:
        words = p["text"].split()
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + size])
            if len(piece.split()) >= min_words:    # skip tiny fragments
                chunks.append({"text": piece, "source": source, "page": p["page"]})
            if start + size >= len(words):
                break
    return chunks
