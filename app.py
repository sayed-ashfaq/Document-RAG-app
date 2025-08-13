import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS

# === Importing modules here ===
from archive.src.document_analyzer.data_ingestion import DocumentHandler
from archive.src.document_analyzer.data_analysis import DocumentAnalyzer
from archive.src.single_document_chat.data_ingestion import SingleDocIngestor
from archive.src.single_document_chat.retrieval import ConversationalRAG as SingleDocRAG
from archive.src.multi_document_chat.data_ingestion import DocumentIngestor
from archive.src.multi_document_chat.retrieval import ConversationalRAG as MultiDocRAG
from utils.model_loader import ModelLoader

FAISS_INDEX_PATH = Path("faiss_index")

st.set_page_config(page_title="RAG Test App", layout="wide")
st.title("📄 RAG Modules Testing App")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📑 Document Analysis",
    "📄 Single Document Chat",
    "📚 Multi Document Chat"
])

# === Tab 1: Document Analysis ===
with tab1:
    st.header("📑 Document Analysis")
    uploaded_pdf = st.file_uploader("Upload a PDF for analysis", type=["pdf"], key="analysis_pdf")

    if uploaded_pdf:
        if st.button("Run Analysis", key="analyze_btn"):
            try:
                handler = DocumentHandler(session_id="Streamlit_DocAnalysis")
                saved_path = handler.save_pdf(uploaded_pdf)
                text_content = handler.read_pdf(saved_path)

                analyzer = DocumentAnalyzer()
                analysis_result = analyzer.analyze_document(text_content)

                st.subheader("Analysis Result")
                for key, value in analysis_result.items():
                    st.markdown(f"**{key}:** {value}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# === Tab 2: Single Document Chat ===
with tab2:
    st.header("📄 Single Document Chat")
    single_pdf = st.file_uploader("Upload a PDF for single doc chat", type=["pdf"], key="single_pdf")
    question = st.text_input("Ask a question about the document", key="single_question")

    if single_pdf and question and st.button("Get Answer", key="single_answer_btn"):
        try:
            model_loader = ModelLoader()

            # Load existing FAISS index or create a new one
            if FAISS_INDEX_PATH.exists():
                embeddings = model_loader.load_embedding()
                vectorstore = FAISS.load_local(
                    folder_path=str(FAISS_INDEX_PATH),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            else:
                ingestor = SingleDocIngestor()
                retriever = ingestor.ingest_files(uploaded_files=[single_pdf])

            rag = SingleDocRAG(retriever=retriever, session_id="Streamlit_SingleDoc")
            answer = rag.invoke(question)

            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# === Tab 3: Multi Document Chat ===
with tab3:
    st.header("📚 Multi Document Chat")
    multi_pdfs = st.file_uploader(
        "Upload multiple PDFs for multi-doc chat",
        type=["pdf"],
        accept_multiple_files=True,
        key="multi_pdfs"
    )
    multi_question = st.text_input("Ask a question across all documents", key="multi_question")

    if multi_pdfs and multi_question and st.button("Get Multi-Doc Answer", key="multi_answer_btn"):
        try:
            ingestor = DocumentIngestor()
            retriever = ingestor.ingest_files(uploaded_files=multi_pdfs)

            rag = MultiDocRAG(session_id="Streamlit_MultiDoc", retriever=retriever)
            answer = rag.invoke(multi_question)

            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
