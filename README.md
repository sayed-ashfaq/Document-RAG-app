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
```
project-name/
│
├── .github/
│   └── workflows/                  # GitHub Actions CI/CD workflows
│       ├── aws.yml                 # AWS deployment workflow
│       ├── task_definition.json    # ECS or container task definition
│       └── template.yml            # Template for workflows
│
├── .idea/                          # PyCharm IDE config files (can be ignored in Git)
│   └── ...
│
├── api/
│   └── main.py                     # Main API entrypoint (FastAPI/Flask)
│   └── ...                         # Other API endpoints, routers
│
├── archive/
│   └── src/                        # Old or backup source code for reference
│       └── ...
│
├── config/
│   └── config.yaml                 # Core configuration settings
│   └── ...                         # Additional config files (JSON/YAML)
│
├── data/
│   ├── doc_analysis/               # Data for document analysis workflow
│   ├── multidoc_chat/              # Data for multi-document chat workflow
│   └── single_doc/                 # Data for single-document chat workflow
│
├── exception/
│   ├── __init__.py                 # Makes folder a Python package
│   └── Custom_exception.py         # Custom exception classes for robust error handling
│
├── faiss_index/
│   ├── index.faiss                  # FAISS vector index for embeddings
│   ├── index.pkl                    # Serialized FAISS object for fast loading
│   └── ...                          # Any session or auxiliary files
│
├── logger/
│   ├── __init__.py                 # Python package initializer
│   └── custom_logger.py            # Logging utilities for debugging & monitoring
│
├── model/
│   ├── __init__.py                 # Python package initializer
│   └── models.py                   # Trained ML/DL models or model utilities
│
├── notebook/
│   └── ...                         # Jupyter notebooks for experiments and testing
│
├── prompts/
│   ├── __init__.py                 # Python package initializer
│   └── prompts.py                  # Prompt templates and utilities for RAG pipelines
│
├── src/
│   ├── document_analyzer/
│   │   ├── __init__.py
│   │   └── data_analysis.py        # Core logic for document analysis
│   │
│   ├── document_chat/
│   │   ├── __init__.py
│   │   └── retrieval.py            # Single & multi-document chat workflows
│   │
│   ├── documents_compare/
│   │   ├── __init__.py
│   │   └── document_comparator.py  # Logic to compare different versions of documents
│   │
│   └── document_ingestion/
│       ├── __init__.py
│       └── data_ingestion.py       # Parsing & preprocessing documents for RAG
│
├── static/
│   └── style.css                   # Frontend CSS styling
│               
│
├── templates/
│   └── index.html                  # HTML templates for FastAPI/Flask or frontend rendering
│
├── utils/
│   ├── __init__.py                 # Python package initializer
│   ├── config_loaders.py           # Load and manage configuration files
│   ├── document_ops.py             # Helper functions for document processing
│   ├── file_io.py                  # File reading/writing utilities
│   └── model_loader.py             # Load ML/DL models efficiently
│
├── .dockerignore                   # Docker ignore rules
├── .gitattributes                  # Git attributes
├── .gitignore                      # Git ignore rules
├── Dockerfile                      # Docker container setup
├── README.md                        # Project README (documentation)
├── app.py                          # Main app entrypoint (Streamlit/FastAPI/Flask)
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup script
└── test.py                          # Test scripts or Streamlit UI for quick prototyping

```

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

