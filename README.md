# AI Document Assistant

An AI-powered PDF assistant that lets users ask questions about documents, search for relevant passages, generate summaries, and export answers with source citations.

Built with Python, Streamlit, FAISS, Sentence Transformers, and Ollama.

## Features

- **Multi-PDF support:** Work with multiple PDF documents in one session.
- **Question answering:** Ask natural-language questions about document contents.
- **Semantic search:** Retrieve relevant passages using vector embeddings and FAISS.
- **Source citations:** Display source document and page references for answers.
- **Document summaries:** Generate full-document, page-range, and chapter summaries.
- **Follow-up questions:** Use conversational context to ask related questions.
- **PDF exports:** Export chat reports and document summaries.
- **Persistent indexing:** Save and reuse document indexes to reduce repeated processing.
- **Local LLM inference:** Generate answers using Ollama and Llama 3.2.

## Tech Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Programming language | Python |
| PDF extraction | pypdf |
| Text embeddings | Sentence Transformers |
| Vector search | FAISS |
| Language model | Ollama + Llama 3.2 |
| Numerical processing | NumPy |
| PDF report generation | ReportLab |

## How It Works

1. Upload one or more PDF documents.
2. Extract text and divide it into overlapping chunks.
3. Generate embeddings for the chunks.
4. Index the embeddings for similarity-based retrieval.
5. Retrieve relevant passages for a user's question.
6. Generate an answer using the local language model.
7. Display source references and allow summaries or PDF exports.

## Run Locally

### Prerequisites

- Python installed
- Git
- Ollama installed and running

### 1. Clone the repository

```bash
git clone https://github.com/imaadmohd22/ai-document-assistant.git
cd ai-document-assistant
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the language model

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the application.

### 5. Start the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Project Structure

```text
ai-document-assistant/
├── app.py
├── processor.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Design Considerations

- Retrieval is used to ground answers in relevant document passages.
- Source references help users verify answers against the original PDFs.
- Persistent indexes help avoid repeating embedding work for unchanged content.
- A local model provides an option for document processing without relying on a hosted LLM API.

## Limitations

- Answer quality depends on PDF text extraction, document quality, retrieval, and model output.
- Scanned PDFs may require a separately configured OCR pipeline.
- Local inference performance depends on the computer's available resources.
- Public cloud deployment requires a compatible hosted inference setup or a server that can run the model; installing Python dependencies alone does not provide an Ollama model.

## Future Improvements

- Configurable OCR for scanned documents
- Automated evaluation of retrieval and answer quality
- Improved document comparison and metadata filters
- Hosted inference and public deployment
- Automated tests and deployment checks

## Privacy Note

The application is designed for local use with Ollama. Review the code and storage configuration before processing confidential documents or exposing the application publicly. Do not upload private documents, generated indexes, credentials, or API keys to the public repository.

## Author

**Mohd Imaad**

[GitHub](https://github.com/imaadmohd22)
