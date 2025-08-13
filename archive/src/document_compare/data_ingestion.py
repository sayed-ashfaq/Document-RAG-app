import sys
from pathlib import Path
import fitz # manipulate the pdf part of PyMuPDF
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from datetime import datetime
import uuid


class DocumentComparator:
    def __init__(self, base_dir:str= "data/document_compare", session_id= None):
        self.log= CustomLogger().get_logger(__name__)
        self.base_dir= Path(base_dir)
        self.session_id= session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_path= self.base_dir/self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

        self.log.info("Document Comparator initialized", session_path= str(self.session_path))
    # def delete_existing_files(self):
    #     """Delete existing files"""
    #     try:
    #         if self.base_dir.exists() and self.base_dir.is_dir():
    #             for file in self.base_dir.iterdir():
    #                 if file.is_file():
    #                     file.unlink() # delete file
    #                     self.log.info("File deleted successfully", path= str(file))
    #             self.log.info("Directory cleaned", directory= str(self.base_dir))
    #     except Exception as e:
    #         self.log.error("Error deleting existing files: {}".format(e))
    #         raise CustomException("Error deleting existing files", sys)
    def save_uploaded_files(self, reference_file, actual_file):
        """Save uploaded files"""
        try:
            # self.delete_existing_files()
            # self.log.info("Existing files deleted successfully")

            ref_path= self.session_path/reference_file.name
            act_path= self.session_path/actual_file.name

            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise ValueError("Only PDF files are allowed.")

            with open(ref_path, "wb") as f:
                f.write(reference_file.get_buffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.get_buffer())

            self.log.info("File saved", reference= str(ref_path), actual= str(act_path))
            return ref_path, act_path


        except Exception as e:
            self.log.error("Error saving uploaded files: {}".format(e))
            raise CustomException("Error occurred while saving the pdf", sys)

    def read_pdf(self, pdf_path: Path) -> str:
        """Read PDF file and extract text from each page"""
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                all_text= []
                for page_num in range(doc.page_count):
                    page= doc.load_page(page_num)
                    text= page.get_text()
                    if text.strip():
                        all_text.append(f"\n --- page {page_num} ---\n{text}")
                self.log.info("PDF read successfully", file= str(pdf_path), pages= len(all_text))
                return "/n".join(all_text)

        except Exception as e:
            self.log.error("Error reading PDF file: {}".format(e))
            raise CustomException("Error reading PDF file", sys)

    def combine_documents(self)-> str:

        try:
            doc_parts= []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower()== ".pdf":
                    content= self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text= "\n\n".join(doc_parts)
            self.log.info("Documents combined", count= len(doc_parts), session= self.session_id)
            return combined_text
        except Exception as e:
            self.log.error("Error combining documents: {}".format(e))
            raise CustomException("Error while combining the documents", sys)

    def clean_old_sessions(self, keep_latest: int= 3):
        try:
            session_folders= sorted(
                [f for f in self.base_dir.iterdir() if f.is_dir()],
                reverse=True
            )
            for folder in session_folders[keep_latest:]:
                for file in folder.iterdir():
                    file.unlink()
                folder.rmdir()
                self.log.info("Old session folder deleted", path= str(folder))

        except Exception as e:
            self.log.error("Error cleaning old sessions: {}".format(e))
            raise CustomException("Error cleaning old sessions", sys)