# import os
# from pathlib import Path
#
# from jupyterlab.semver import comparator
# from langchain_community.retrievers.kendra import combined_text
#
# from src.document_analyzer.data_ingestion import DocumentHandler
# from src.document_analyzer.data_analysis import DocumentAnalyzer

# # path to pdf
# pdf_path= r"C:\Users\302sy\Desktop\Generative AI\RAG_ETE\data\document_analysis\attention_is_all_you_need.pdf"
#
# # Dummy wrapper to simulate uploaded file (streamlit style)
# class DummyFile:
#     def __init__(self, filepath):
#         self.name= Path(filepath).name
#         self._file_path= filepath
#
#     def getbuffer(self):
#         return open(self._file_path, 'rb').read()
#
# def main():
#     try:
#         #------data ingestion------
#         print("Starting PDF ingestion...")
#         dummy_pdf= DummyFile(pdf_path)
#
#         handler= DocumentHandler(session_id="Testing ingestion")
#         saved_path= handler.save_pdf(dummy_pdf)
#         print(f"PDF ingestion saved to {saved_path}")
#
#         text_content= handler.read_pdf(saved_path)
#         print(f"Extracted text length: {len(text_content)} chars/n")
#
#         # -----------data analysis-------
#         print("Starting metadata analysis....")
#         analyzer= DocumentAnalyzer()
#         analyzer_result= analyzer.analyze_document(text_content)
#         print(f"Analyzer result: {analyzer_result}")
#
#         # -----------Display results-------
#         print("/n=====METADATA ANALYSIS RESULT=====")
#         for key, value in analyzer_result.items():
#             print(f"{key}: {value}")
#
#     except Exception as e:
#         print(f"test failed: {e}")
#
# if __name__ == "__main__":
#     main()


# ## ---------- Testing Single Doc chat ---------------##
# import sys
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
#
# # from notebook.experiments import retriever
# from src.single_document_chat.data_ingestion import SingleDocIngestor
# from src.single_document_chat.retrieval import ConversationalRAG
# from utils.model_loader import ModelLoader
#
# FAISS_INDEX_PATH= Path("faiss_index")
#
#
# def test_conversational_rag_on_pdf(pdf_path: str, question: str):
#     try:
#         model_loader= ModelLoader()
#         if FAISS_INDEX_PATH.exists():
#             print("Loading existing FAISS index")
#             embeddings= model_loader.load_embedding()
#             vectorstore= FAISS.load_local(folder_path= str(FAISS_INDEX_PATH), embeddings= embeddings, allow_dangerous_deserialization=True)
#             retriever= vectorstore.as_retriever(search_type= "similarity", search_kwargs={"k": 5})
#         else:
#             # Step 2: Ingest document and create retriever
#             print("FAISS index not found, ingesting pdf and creating FAISS index...")
#             with open(pdf_path, 'rb') as f:
#                 upload_files= [f]
#                 ingestor= SingleDocIngestor()
#                 retriever= ingestor.ingest_files(uploaded_files=upload_files)
#         print("Running conversational RAG....")
#         session_id= "test_conversational_rag_on_pdf"
#         rag = ConversationalRAG(retriever=retriever, session_id=session_id)
#         response= rag.invoke(question)
#         print(f"\nQuestion:{question},\nResponse:{response}")
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)
#
# if __name__ == "__main__":
#     ## Example pdf path and question
#     pdf_path= "C:\\Users\\302sy\\Desktop\\Generative AI\\RAG_ETE\\data\\single_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
#     question= "What is the main topic of the document?"
#     if not Path(pdf_path).exists():
#         print(f"PDF file does not exists at {pdf_path}")
#         sys.exit(1)
#     else:
#         while question not in ("q", "exit","quit"):
#             test_conversational_rag_on_pdf(pdf_path, question)
#             question = input("User:\n")


### -------------------Testing Multi-Doc chat module----------------------##
import sys
from pathlib import Path

# from notebook.experiments import session_id
# from notebook.experiments import retriever
from archive.src.multi_document_chat.data_ingestion import DocumentIngestor
from archive.src.multi_document_chat.retrieval import ConversationalRAG






def test_document_ingestion_and_rag():
    try:
        test_files= [
            "C:\\Users\\302sy\\Desktop\\LLMOps.pdf",
            "C:\\Users\\302sy\\Desktop\\Generative AI\\RAG_ETE\\data\\single_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf",
            "S:\\Reading\\AVIATION\\TIMEMANAGEMENT-210130-161657.pdf",
            "S:\\Reading\\AVIATION\\Airlinesname-210130-161441.pdf"
        ]
        uploaded_files= []
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, 'rb'))

            else:
                print(f"File {file_path} does not exist")

        if not uploaded_files:
            print("NO valid files are uploaded")
            sys.exit(1)


        ingestor= DocumentIngestor()
        retriever= ingestor.ingest_files(uploaded_files)

        for f in uploaded_files:
            f.close()

        return retriever
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Testing Multi Document Chat...")
    retriever = test_document_ingestion_and_rag()
    session_id = "Testing multi document chat"
    rag = ConversationalRAG(session_id=session_id,retriever= retriever)
    question= 'what is meaning of lLM ops'
    while question not in ("q", "exit", "quit"):
        question = input("User:\n")
        answer= rag.invoke(question)
        print("Answer:\n", answer)
