# AI Document Assistant

A local-first AI application for asking questions about PDF documents, generating summaries, and exploring document content with source-aware answers.

## Features

- **PDF Question Answering:** Ask natural-language questions about uploaded documents.
- **Semantic Search:** Find relevant document passages using sentence embeddings and FAISS.
- **Source Citations:** Display supporting document filenames and page numbers with answers.
- **Multi-document Support:** Search across a collection or select an individual PDF.
- **Document Summarization:** Generate detailed summaries, short summaries, and key points.
- **Page-range Summaries:** Summarize a selected range of pages.
- **Chapter Summaries:** Detect chapter headings and summarize selected chapters.
- **Chat Export:** Export questions, answers, and citations into a PDF report.
- **Summary Export:** Download summary reports as PDFs.
- **Persistent Indexing:** Save document metadata, embeddings, and the search index locally.
- **Local LLM:** Use Ollama with the `llama3.2:3b` model for answer generation.

## Tech Stack

- Python
- Streamlit
- Ollama
- Llama 3.2
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- NumPy
- pypdf
- ReportLab

## How It Works

1. Upload one or more PDF documents.
2. Extract text and divide it into overlapping chunks.
3. Convert the chunks into vector embeddings.
4. Store embeddings in a FAISS similarity-search index.
5. Retrieve relevant passages for a user's question.
6. Generate an answer using the local Ollama model and retrieved context.
7. Show source references and allow users to export reports.

## Requirements

- Python 3.13 or another compatible Python version for the installed dependencies
- Git
- Ollama
- The `llama3.2:3b` model
- Sufficient RAM and disk space for the language model and embedding model

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/imaadmohd22/ai-document-assistant.git
cd ai-document-assistant
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download the model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the application.

### 5. Launch the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Project Structure

```text
ai-document-assistant/
├── app.py
├── processor.py
├── requirements.txt
├── README.md
├── .gitignore
└── storage/       # Created locally; not committed to Git
```

## Privacy and Data Handling

The application is designed to run locally. Documents and saved indexes may be stored on the local machine in the application's storage directory.

Do not upload confidential documents, saved indexes, credentials, or private customer information to a public repository. Review the application's storage and deployment configuration before using it with sensitive business data.

## Deployment Considerations

The application currently depends on a locally running Ollama service and local document-index storage. A standard Streamlit hosting environment will not automatically have this model or persistent local storage.

A public deployment requires an appropriate inference setup, suitable compute resources, and a plan for persistent storage. Do not publish a public demo until these dependencies have been configured and tested.

## Current Limitations

- PDF text extraction depends on the quality and structure of the source PDF.
- Chapter detection depends on recognizable headings.
- Answers may be incomplete when the document does not contain sufficient information.
- Model response speed depends on local hardware.
- Public hosting requires additional deployment configuration.

## Future Improvements

- Remote model-provider integration
- Improved OCR support for scanned PDFs
- User authentication and isolated document collections
- Document-level access controls
- Automated tests and deployment pipeline
- Production-grade persistent storage

## Author

**Mohd Imaad**

GitHub: [imaadmohd22](https://github.com/imaadmohd22)

---

Built as a practical project exploring document retrieval, local language models, semantic search, and AI-powered productivity tools.