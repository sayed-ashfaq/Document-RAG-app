---

# Document Portal: End-to-End RAG Application

---

## 📄 Overview

**Document Portal** is an interactive, **AI-powered platform** that demonstrates **Retrieval-Augmented Generation (RAG)** workflows on PDF documents. It enables users to:

* Analyze documents and extract key metadata.
* Ask questions about one or multiple documents using a conversational AI interface.
* Compare different versions of documents to quickly spot changes.
* Operate with robust industry-standard logging, error handling, and session management.

This application is built for **research, knowledge management, and enterprise workflows**, offering a clean, intuitive interface for document-centric AI operations.

---

## 🚀 Key Features

* **Document Analysis & Chat**
  Upload PDFs to extract text and metadata, and interactively query their content using natural language. Supports both **single-document** and **multi-document** workflows.

* **Document Comparison**
  Upload two versions of a document and instantly detect differences, making version control and review easier.

* **Efficient & Scalable RAG Pipeline**
  Uses **FAISS** for vector storage and **LangChain** for retrieval and answer generation. Existing indexes are reused to reduce processing time.

* **Interactive UI**
  Built with **Streamlit**, providing a responsive and minimal interface for smooth user interaction.

---

## 🏗️ How It Works

```
[Upload PDFs] --> [Data Ingestion & Preprocessing] --> [FAISS Vectorization] --> [RAG Query] --> [Answers / Comparison]
```

**Step-by-step workflow:**

1. **Upload Documents** – Users can upload one or multiple PDFs.
2. **Data Ingestion & Processing** – Documents are parsed and structured for embeddings.
3. **Vectorization** – FAISS creates vector representations for fast similarity search.
4. **RAG Pipeline** – LangChain retrieves context and generates natural language responses.
5. **Output** – Answers, analysis, or document differences are displayed interactively.

---

## 🖼️ Screenshots / Visuals

**Document Chat Example:**
<img width="1903" height="929" alt="Screenshot 2025-08-19 113716" src="https://github.com/user-attachments/assets/7e2bd4dd-37d9-4166-915d-d86d6d626ea6" />

**Document Comparison:**
<img width="1903" height="929" alt="Screenshot 2025-08-19 113716" src="https://github.com/user-attachments/assets/a3c3b2f6-2e0d-4b31-8b7e-cc93922f12ca" />

**Document Summarizer:**
<img width="1894" height="928" alt="Screenshot 2025-08-19 104629" src="https://github.com/user-attachments/assets/12bb72e0-ed60-42c9-8f43-e718d9753df1" />


**Project Structure Overview:**
![Project Structure](path/to/project_structure_image.png)

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit** – Interactive frontend
* **LangChain** – RAG orchestration
* **FAISS** – Vector database for embeddings
* **Custom Modules** – Document ingestion, analysis, retrieval, and comparison

---

## ⚙️ Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/sayed-ashfaq/document-RAG-app.git
   cd document-portal
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:

   ```bash
   streamlit run app.py
   ```
4. Open the Streamlit interface and start uploading PDFs, chatting with documents, or comparing versions.

---

## 💡 Notes & Recommendations

* Existing FAISS indexes are reused for faster querying.
* Designed for both single and multi-document workflows in a single interface.
* Ideal for research, document management, and AI-powered knowledge extraction.
* Easily extendable with new RAG workflows or document types.

---

