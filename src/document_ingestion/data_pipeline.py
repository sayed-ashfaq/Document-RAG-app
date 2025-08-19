"""
Module: data_pipeline.py

This module provides classes for:
1. Managing FAISS vector stores (for semantic search & retrieval).
2. Ingesting and splitting documents for retrieval-based chat (ChatIngestor).
3. Handling PDF saving/reading for analysis (DocHandler).
4. Managing document comparison with session-based versioning (DocumentComparator).

It uses LangChain, FAISS, PyMuPDF (fitz), and custom utilities.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}  # allowed file formats


# ================================================================
# FAISS Manager
# ================================================================
class FaissManager:
    """
    Manages a FAISS vector store (load, create, update) for storing embeddings.

    Responsibilities:
    - Create or load a FAISS index.
    - Add new documents without duplicating existing ones.
    - Maintain metadata for ingested docs.
    """

    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        """
        Initialize FAISS manager.

        Args:
            index_dir (Path): Directory where FAISS index files and metadata are stored.
            model_loader (ModelLoader, optional): Loader for embeddings. Defaults to None.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)  # create if not exists

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}  # stores fingerprints of added docs

        # Load existing metadata if available
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}  # fallback to empty if broken

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        """Check if FAISS index files exist in storage."""
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Generate a unique fingerprint for a document.

        Args:
            text (str): Document content.
            md (dict): Metadata (file path, row id, etc.).

        Returns:
            str: Unique fingerprint string.
        """
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        """Persist metadata (ingested docs info) to disk."""
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[Document]):
        """
        Add new documents to FAISS index (idempotent, avoids duplicates).

        Args:
            docs (List[Document]): Documents to add.

        Returns:
            int: Number of new documents added.
        """
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:  # skip duplicates
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        """
        Load an existing FAISS index or create a new one.

        Args:
            texts (List[str], optional): Texts to create a new FAISS index if none exists.
            metadatas (List[dict], optional): Metadata for texts.

        Returns:
            FAISS: Loaded or newly created FAISS index.
        """
        if self._exists():  # load existing
            self.vs = FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )

            return self.vs

        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)

        # create new index
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs


# ================================================================
# Chat Ingestor
# ================================================================
class ChatIngestor:
    """
    Handles document ingestion, splitting, and vector store creation
    for retrieval-augmented generation (RAG).
    """

    def __init__(self, temp_base: str = "data", faiss_base: str = "faiss_index",
                 use_session_dirs: bool = True, session_id: Optional[str] = None):
        """
        Args:
            temp_base (str): Base directory for temporary files.
            faiss_base (str): Base directory for FAISS indexes.
            use_session_dirs (bool): Whether to use session-specific directories.
            session_id (str, optional): Custom session id (auto-generated if None).
        """
        try:
            self.model_loader = ModelLoader()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            # Ensure directories exist
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                     session_id=self.session_id,
                     temp_dir=str(self.temp_dir),
                     faiss_dir=str(self.faiss_dir),
                     sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e

    def _resolve_dir(self, base: Path):
        """Create (if needed) and return a session-specific or global directory."""
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base

    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        """
        Split documents into overlapping text chunks.

        Args:
            docs (List[Document]): Input documents.
            chunk_size (int): Max characters per chunk.
            chunk_overlap (int): Overlap between chunks.

        Returns:
            List[Document]: Chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    def built_retriver(self, uploaded_files: Iterable, *, chunk_size: int = 1000,
                       chunk_overlap: int = 200, k: int = 5):
        """
        Build a retriever from uploaded files using FAISS.

        Args:
            uploaded_files (Iterable): Uploaded file objects.
            chunk_size (int): Max characters per chunk.
            chunk_overlap (int): Overlap between chunks.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            Retriever: FAISS retriever object.
        """
        try:
            # Step 1: Save uploaded files
            paths = save_uploaded_files(uploaded_files, self.temp_dir)

            # Step 2: Load into LangChain Document objects
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            # Step 3: Split docs into chunks
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Step 4: Initialize FAISS manager
            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            # Step 5: Load or create FAISS index
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)

            # Step 6: Add docs to FAISS
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


# ================================================================
# Document Handler (PDF for Analysis)
# ================================================================
class DocHandler:
    """
    Handle saving and reading PDFs for analysis, organized by session.
    """

    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        """
        Args:
            data_dir (str, optional): Base storage path.
            session_id (str, optional): Session ID (auto-generated if None).
        """
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH",
                                              os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        """
        Save an uploaded PDF to the session directory.

        Args:
            uploaded_file: File-like object.

        Returns:
            str: Path where the file was saved.
        """
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        """
        Read a PDF and return its text content page by page.

        Args:
            pdf_path (str): Path to the PDF.

        Returns:
            str: Extracted text.
        """
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process PDF: {pdf_path}", e) from e


# ================================================================
# Document Comparator
# ================================================================
class DocumentComparator:
    """
    Save, read & combine PDFs for comparison with session-based versioning.
    """

    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        """
        Save two uploaded PDF files (reference and actual) into the session folder.

        Args:
            reference_file: File-like reference PDF.
            actual_file: File-like actual PDF.

        Returns:
            Tuple[Path, Path]: Paths of saved reference and actual files.
        """
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        except Exception as e:
            log.error("Error saving PDF files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        """
        Read and extract text from a PDF file.

        Args:
            pdf_path (Path): Path to the PDF.

        Returns:
            str: Extracted text.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # type: ignore
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e

    def combine_documents(self) -> str:
        """
        Combine multiple PDFs from the session into a single text string.

        Returns:
            str: Combined text from all PDFs.
        """
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int = 3):
        """
        Delete older session folders, keeping only the latest few.

        Args:
            keep_latest (int): Number of latest sessions to keep.
        """
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e

# ================================================================
# End of module
# ================================================================
