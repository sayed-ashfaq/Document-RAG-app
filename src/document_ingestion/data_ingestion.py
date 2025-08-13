

## 1. Create a Skeleton code
## 2. Import statements

from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional

import fitz
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException


class FaissManager:
    def __init__(self):
        pass

    def _exist(self):
        pass

    @staticmethod
    def _fingerprint():
         pass

    def _save_meta(self):
        pass

    def add_documents(self):
        pass

    def load_or_create(self):
        pass


class DocHandler:
    def __init__(self):
        pass

    def save_pdf(self):
        pass

    def read_pdf(self):
        pass

class DocumentComparator:
    def __init__(self):
        pass

    def save_uploaded_files(self):
        pass

    def read_pdf(self):
        pass

    def combine_document(self):
        pass

    def clean_old_session(self):
        pass

class ChatIngestor:
    def __init__(self):
        pass

    def resolve_dir(self):
        pass

    def _split(self):
        pass

    def build_retriever(self):
        pass