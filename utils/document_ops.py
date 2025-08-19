from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from fastapi import UploadFile
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

# Supported file types we can load
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load documents from given file paths (PDF, DOCX, TXT).
    Automatically uses the right loader based on file extension.
    """
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()

            # Decide which loader to use based on file type
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                # Skip unsupported file types
                log.warning("Unsupported extension skipped", path=str(p))
                continue

            # Add loaded pages (documents can have multiple pages)
            docs.extend(loader.load())

        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        # Wrap and re-raise errors with custom exception
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e


def concat_for_analysis(docs: List[Document]) -> str:
    """
    Combine multiple documents into a single text string,
    including their source info for easier analysis.
    """
    parts = []
    for d in docs:
        # Try to find source info from metadata, fallback to "unknown"
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")

    return "\n".join(parts)


def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    """
    Prepare text for comparing two sets of documents
    (e.g., reference vs actual).
    """
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"


# ---------- Helpers ----------
class FastAPIFileAdapter:
    """
    Adapter to make FastAPI's UploadFile work like
    a simple file object with `.name` and `.getbuffer()`.

    Useful for reusing code that expects normal file objects.
    """

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename

    def getbuffer(self) -> bytes:
        """Read all file contents as bytes."""
        self._uf.file.seek(0)  # rewind file pointer
        return self._uf.file.read()


def read_pdf_via_handler(handler, path: str) -> str:
    """
    Generic helper to read a PDF using a custom handler.
    Supports both `read_pdf` and `read_` methods.
    """
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")
