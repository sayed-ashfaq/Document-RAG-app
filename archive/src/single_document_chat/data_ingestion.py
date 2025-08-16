import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DocumentPortalException
from utils.model_loader import ModelLoader
from datetime import datetime


class SingleDocIngestor:
    def __init__(self, data_dir= "data/single_document_chat",faiss_dir: str= "faiss_index"):
        self.log = CustomLogger().get_logger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir = Path(faiss_dir)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)  # creates directory not file
        try:
            self.model_loader= ModelLoader()
            self.log.info("Model Initialization Completed", temp_path= str(self.data_dir), faiss_path= str(self.faiss_dir))
        except Exception as e:
            self.log.error("Failed to initialize the Single doc Ingestor", error= str(e))
            raise DocumentPortalException("Failed to initialize the Single doc Ingestor", sys)


    def ingest_files(self, uploaded_files: list):
        try:
            document= []
            for uploaded_file in uploaded_files:
                unique_filename= f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                temp_path= self.data_dir/unique_filename

                with open(temp_path, "wb") as f_out:
                    f_out.write(uploaded_file.read())
                self.log.info("PDF saved for ingestion", filename= uploaded_file.name)
                loader= PyPDFLoader(str(temp_path))
                docs= loader.load()
                document.extend(docs)

            self.log.info("PDF files Loaded Successfully.", count= len(document))
            return self._create_retriever(document)
        except Exception as e:
            self.log.error("Failed to create retriever", error= str(e))
            raise DocumentPortalException("Failed to create retriever", sys)

    def _create_retriever(self, docs: list):
        try:
            splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks= splitter.split_documents(docs)
            self.log.info("Document split into chunks.", count= len(chunks))

            embeddings= self.model_loader.load_embedding()
            vector_store= FAISS.from_documents(documents=chunks, embedding=embeddings)

            vector_store.save_local(str(self.faiss_dir))
            self.log.info("FAISS index created and saved", faiss_path= str(self.faiss_dir))

            retriever= vector_store.as_retriever(search_type= "similarity", search_kwargs= {"k":5})
            self.log.info("Retriever created and saved", retriever= str(retriever))
            return retriever

        except Exception as e:
            self.log.error("Failed to create retriever", error= str(e))
            raise DocumentPortalException("Failed to create retriever", sys)



































