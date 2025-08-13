# Document Portal: Industry level End to End RAG application
# 📄 RAG Modules Testing App

An interactive **Streamlit-based application** that demonstrates three different **Retrieval-Augmented Generation (RAG)** workflows:  
1. **Document Analysis** – Extract and analyze metadata from a PDF.  
2. **Single Document Chat** – Ask questions about a single PDF using a FAISS vector index.  
3. **Multi Document Chat** – Ask questions across multiple PDFs with a consolidated RAG pipeline.
4. **Document Compare** - Upload two similar different version of documents and check the changes instantly. 

---

## 🚀 Features

### 1. Document Analysis
- Upload a PDF and extract its text content.
- Analyze document metadata (e.g., title, author, content stats).
- Display results in a clean and minimal UI.

### 2. Single Document Chat
- Upload one PDF and create a FAISS vector store.
- Reuse existing FAISS index if available to save processing time.
- Ask natural language questions about the document.
- Retrieve answers using a **Conversational RAG pipeline**.

### 3. Multi Document Chat
- Upload multiple PDFs simultaneously.
- Ingest all documents into a combined vector store.
- Ask questions spanning across all uploaded files.
- Get context-aware answers with references.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** – UI framework
- **LangChain** – RAG orchestration
- **FAISS** – Vector store for embeddings
- **Custom ingestion and retrieval modules** for each workflow

---
