import os
import re
import json
import tempfile
from io import BytesIO

import faiss
import numpy as np
import llm_provider as ollama
import pytesseract

from pdf2image import convert_from_bytes
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

INDEX_FILE = os.path.join(STORAGE_DIR, "faiss.index")
DATA_FILE = os.path.join(STORAGE_DIR, "document_data.json")
EMBEDDINGS_FILE = os.path.join(STORAGE_DIR, "embeddings.npy")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

PUBLIC_DEMO = os.getenv(
    "PUBLIC_DEMO", "false"
).strip().lower() in {"1", "true", "yes", "on"}


def public_demo_enabled():
    """Disable persistent document storage on shared public deployments."""
    return PUBLIC_DEMO


# OCR settings
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", "220"))

# Optional explicit paths. Normally PATH is enough.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip()

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

_embedding_model = None


# ============================================================
# EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


# ============================================================
# PDF EXTRACTION + OCR FALLBACK
# ============================================================

def _ocr_pdf_page(pdf_bytes, page_number):
    """OCR one PDF page, using its original 1-based page number."""
    images = convert_from_bytes(
        pdf_bytes,
        dpi=OCR_DPI,
        first_page=page_number,
        last_page=page_number,
        fmt="png",
        poppler_path=POPPLER_PATH or None,
        thread_count=1,
    )

    if not images:
        return ""

    try:
        text = pytesseract.image_to_string(
            images[0],
            lang=OCR_LANGUAGE,
        )
        return (text or "").strip()
    finally:
        for image in images:
            image.close()


def extract_pdf_pages(pdf_file):
    """
    Extract normal PDF text first. If a page has no extractable text,
    attempt OCR on that page.

    Return format:
    [{"filename": str, "page": int, "text": str}, ...]
    """
    pdf_bytes = pdf_file.getvalue()
    filename = getattr(pdf_file, "name", "document.pdf")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as error:
        raise ValueError(
            f"Cannot open PDF '{filename}': {error}"
        ) from error

    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
            if not result:
                raise ValueError(
                    f"'{filename}' is password-protected. "
                    "Please upload an unprotected PDF."
                )
        except Exception as error:
            raise ValueError(
                f"Cannot read encrypted PDF '{filename}': {error}"
            ) from error

    pages = []
    ocr_attempted = 0
    ocr_failures = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        # Use OCR only when ordinary extraction finds no text.
        if not text and OCR_ENABLED:
            ocr_attempted += 1

            try:
                text = _ocr_pdf_page(pdf_bytes, page_number)
            except Exception as error:
                ocr_failures.append(
                    f"Page {page_number}: {error}"
                )
                text = ""

        if text:
            pages.append({
                "filename": filename,
                "page": page_number,
                "text": text,
            })

    if not pages:
        if not OCR_ENABLED:
            raise ValueError(
                f"No readable text was found in '{filename}'. "
                "OCR is disabled. Set OCR_ENABLED=true to enable it."
            )

        details = ""
        if ocr_failures:
            details = (
                "\nOCR errors:\n" + "\n".join(ocr_failures[:3])
            )

        raise ValueError(
            f"No readable text could be extracted from '{filename}'. "
            "The PDF may be blank, damaged, password-protected, or "
            "require a different OCR language. Check Tesseract, "
            "Poppler, and OCR_LANGUAGE settings."
            + details
        )

    # A mixed PDF may contain pages that OCR could not read.
    if ocr_failures:
        print(
            f"OCR warnings for {filename}: "
            + "; ".join(ocr_failures[:5])
        )

    if ocr_attempted:
        print(
            f"OCR processed {ocr_attempted} page(s) in {filename}; "
            f"{len(pages)} page(s) yielded readable text."
        )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(pages, chunk_size=1000, overlap=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    for page_data in pages:
        text = page_data["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()

            if piece:
                chunks.append({
                    "filename": page_data["filename"],
                    "page": page_data["page"],
                    "text": piece,
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# EMBEDDINGS AND FAISS
# ============================================================

def create_embeddings(chunks):
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def create_faiss_index(embeddings):
    if embeddings is None or len(embeddings) == 0:
        return None

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def rebuild_index_from_existing_embeddings(embeddings):
    """Rebuild FAISS without regenerating existing embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return None

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# ============================================================
# PERSISTENT STORAGE
# ============================================================

def ensure_storage_directory():
    if public_demo_enabled():
        return

    os.makedirs(STORAGE_DIR, exist_ok=True)


def _atomic_save_json(path, data):
    ensure_storage_directory()

    fd, temp_path = tempfile.mkstemp(
        prefix="tmp_", suffix=".json", dir=STORAGE_DIR
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

        os.replace(temp_path, path)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def save_document_index(
    pages,
    chunks,
    embeddings,
    index,
    processed_files,
    chapters,
    chunk_size,
    overlap,
):
    # Never persist visitors' uploaded documents on a shared public app.
    if public_demo_enabled():
        return

    if not chunks or index is None:
        raise ValueError("Cannot save an empty collection.")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of embeddings must match the number of chunks."
        )

    if index.ntotal != len(chunks):
        raise ValueError("The FAISS index and chunks are out of sync.")

    ensure_storage_directory()

    metadata = {
        "version": 2,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "pages": pages,
        "chunks": chunks,
        "processed_files": processed_files,
        "chapters": chapters,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

    temporary_files = {}

    try:
        fd, temp_index = tempfile.mkstemp(
            prefix="tmp_", suffix=".index", dir=STORAGE_DIR
        )
        os.close(fd)

        faiss.write_index(index, temp_index)
        temporary_files[INDEX_FILE] = temp_index

        fd, temp_embeddings = tempfile.mkstemp(
            prefix="tmp_", suffix=".npy", dir=STORAGE_DIR
        )
        os.close(fd)

        np.save(temp_embeddings, embeddings)
        temporary_files[EMBEDDINGS_FILE] = temp_embeddings

        fd, temp_data = tempfile.mkstemp(
            prefix="tmp_", suffix=".json", dir=STORAGE_DIR
        )

        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)

        temporary_files[DATA_FILE] = temp_data

        for destination, temporary in temporary_files.items():
            os.replace(temporary, destination)

    finally:
        for temporary in temporary_files.values():
            if os.path.exists(temporary):
                os.remove(temporary)


def load_document_index():
    # Public visitors should always start with a fresh in-memory collection.
    if public_demo_enabled():
        return None

    required_files = [INDEX_FILE, DATA_FILE, EMBEDDINGS_FILE]

    if not all(os.path.exists(path) for path in required_files):
        return None

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("version") != 2:
            return None

        if data.get("embedding_model") != EMBEDDING_MODEL_NAME:
            return None

        pages = data["pages"]
        chunks = data["chunks"]
        processed_files = data["processed_files"]
        chapters = data["chapters"]

        embeddings = np.load(
            EMBEDDINGS_FILE, allow_pickle=False
        ).astype(np.float32)

        index = faiss.read_index(INDEX_FILE)

        if len(chunks) != len(embeddings):
            return None

        if index.ntotal != len(chunks):
            return None

        if len(chunks) and index.d != embeddings.shape[1]:
            return None

        return {
            "pages": pages,
            "chunks": chunks,
            "embeddings": embeddings,
            "processed_files": processed_files,
            "chapters": chapters,
            "chunk_size": data.get("chunk_size", 1000),
            "overlap": data.get("overlap", 150),
            "index": index,
        }

    except Exception as error:
        print(f"Could not load saved collection: {error}")
        return None


def persistent_index_exists():
    if public_demo_enabled():
        return False

    return all(
        os.path.exists(path)
        for path in [INDEX_FILE, DATA_FILE, EMBEDDINGS_FILE]
    )


def delete_document_index():
    if public_demo_enabled():
        return

    for path in [INDEX_FILE, DATA_FILE, EMBEDDINGS_FILE]:
        if os.path.exists(path):
            os.remove(path)


# ============================================================
# VALIDATE EXISTING EMBEDDINGS
# ============================================================

def validate_embeddings(chunks, embeddings):
    """Return embeddings aligned with chunks."""
    if embeddings is None:
        return create_embeddings(chunks)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(chunks) != len(embeddings):
        return create_embeddings(chunks)

    return embeddings


# ============================================================
# CHAPTER DETECTION
# ============================================================

ROMAN_NUMBERS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii",
    "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
    "xvi", "xvii", "xviii", "xix", "xx",
}

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty",
}


def is_chapter_heading(line):
    match = re.match(
        r"^chapter\s+(.+)$", line.strip(), re.IGNORECASE
    )

    if not match:
        return False

    value = match.group(1).strip().lower()

    return (
        value.isdigit()
        or value in ROMAN_NUMBERS
        or value in NUMBER_WORDS
    )


def clean_chapter_title(line):
    return re.sub(r"\s+", " ", line.strip())


def detect_chapters(pages):
    chapters = []

    for page in pages:
        for line in page["text"].splitlines():
            line = line.strip()

            if is_chapter_heading(line):
                chapters.append({
                    "filename": page["filename"],
                    "title": clean_chapter_title(line),
                    "start_page": page["page"],
                    "end_page": None,
                })
                break

    for chapter in chapters:
        later = [
            item for item in chapters
            if (
                item["filename"] == chapter["filename"]
                and item["start_page"] > chapter["start_page"]
            )
        ]

        next_chapter = min(
            later,
            key=lambda item: item["start_page"],
            default=None,
        )

        if next_chapter:
            chapter["end_page"] = next_chapter["start_page"] - 1
        else:
            chapter["end_page"] = max(
                (
                    page["page"] for page in pages
                    if page["filename"] == chapter["filename"]
                ),
                default=chapter["start_page"],
            )

    return chapters


def get_chapter_pages(pages, chapter):
    return [
        page for page in pages
        if (
            page["filename"] == chapter["filename"]
            and chapter["start_page"] <= page["page"]
            <= chapter["end_page"]
        )
    ]


# ============================================================
# ADD / REPLACE DOCUMENT
# ============================================================

def add_document_to_collection(
    filename,
    new_pages,
    pages,
    chunks,
    embeddings,
    index,
    processed_files,
    chapters,
    chunk_size=1000,
    overlap=150,
):
    """Add or replace a document and reuse unaffected embeddings."""
    embeddings = validate_embeddings(chunks, embeddings)

    keep_chunk_ids = [
        i for i, chunk in enumerate(chunks)
        if chunk["filename"] != filename
    ]

    chunks = [chunks[i] for i in keep_chunk_ids]
    embeddings = embeddings[keep_chunk_ids]

    pages = [
        page for page in pages
        if page["filename"] != filename
    ]

    new_chunks = chunk_text(
        new_pages, chunk_size=chunk_size, overlap=overlap
    )

    new_embeddings = create_embeddings(new_chunks)

    pages.extend(new_pages)
    chunks.extend(new_chunks)

    if len(new_embeddings):
        if len(embeddings):
            embeddings = np.vstack([embeddings, new_embeddings])
        else:
            embeddings = new_embeddings

    index = rebuild_index_from_existing_embeddings(embeddings)
    processed_files = sorted({page["filename"] for page in pages})
    chapters = detect_chapters(pages)

    if chunks:
        save_document_index(
            pages=pages,
            chunks=chunks,
            embeddings=embeddings,
            index=index,
            processed_files=processed_files,
            chapters=chapters,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    else:
        delete_document_index()

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "processed_files": processed_files,
        "chapters": chapters,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }


# ============================================================
# REMOVE DOCUMENT
# ============================================================

def remove_document_from_collection(
    filename,
    pages,
    chunks,
    embeddings,
    chunk_size=1000,
    overlap=150,
):
    """Remove a PDF while preserving other chunk embeddings."""
    embeddings = validate_embeddings(chunks, embeddings)

    keep_chunk_ids = [
        i for i, chunk in enumerate(chunks)
        if chunk["filename"] != filename
    ]

    chunks = [chunks[i] for i in keep_chunk_ids]
    embeddings = embeddings[keep_chunk_ids]

    pages = [
        page for page in pages
        if page["filename"] != filename
    ]

    index = rebuild_index_from_existing_embeddings(embeddings)
    processed_files = sorted({page["filename"] for page in pages})
    chapters = detect_chapters(pages)

    if chunks:
        save_document_index(
            pages=pages,
            chunks=chunks,
            embeddings=embeddings,
            index=index,
            processed_files=processed_files,
            chapters=chapters,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    else:
        delete_document_index()
        embeddings = np.empty((0, 384), dtype=np.float32)

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "processed_files": processed_files,
        "chapters": chapters,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
            normalize_text(text),
        )
    )


def lexical_score(query, text):
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def detect_query_intent(query):
    q = query.lower()

    if any(term in q for term in [
        "beginning", "start", "opening", "first part", "initial"
    ]):
        return "beginning"

    if any(term in q for term in [
        "ending", "final", "last part", "conclusion"
    ]):
        return "ending"

    if any(term in q for term in [
        "summary", "summarize", "overview", "overall", "plot"
    ]):
        return "summary"

    if any(term in q for term in [
        "who is", "who was", "character", "describe"
    ]):
        return "character"

    return "specific"


def intent_score(query, chunk):
    intent = detect_query_intent(query)
    page = chunk["page"]

    if intent == "beginning":
        if page <= 15:
            return 1.0
        if page <= 30:
            return 0.6
        if page <= 50:
            return 0.2

    if intent == "ending" and page >= 1:
        return 0.3

    if intent == "summary" and page <= 15:
        return 0.5

    return 0.0


def retrieve_relevant_chunks(
    query,
    chunks,
    index,
    top_k=5,
    selected_document=None,
):
    if not chunks or index is None:
        return []

    if index.ntotal != len(chunks):
        return []

    if selected_document in (None, "All Documents"):
        allowed_indices = list(range(len(chunks)))
    else:
        allowed_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk["filename"] == selected_document
        ]

    if not allowed_indices:
        return []

    model = get_embedding_model()

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    allowed_vectors = np.asarray(
        [index.reconstruct(i) for i in allowed_indices],
        dtype=np.float32,
    )

    semantic_scores = allowed_vectors @ query_vector[0]
    candidates = []

    for local_i, global_i in enumerate(allowed_indices):
        chunk = chunks[global_i]

        semantic = float(semantic_scores[local_i])
        lexical = lexical_score(query, chunk["text"])
        intent = intent_score(query, chunk)

        score = 0.60 * semantic + 0.20 * lexical + 0.20 * intent

        candidates.append({
            "chunk_index": global_i,
            "filename": chunk["filename"],
            "page": chunk["page"],
            "text": chunk["text"],
            "semantic_score": semantic,
            "lexical_score": lexical,
            "intent_score": intent,
            "final_score": score,
        })

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    selected = []
    seen_pages = set()

    for item in candidates:
        page_key = (item["filename"], item["page"])

        if page_key not in seen_pages:
            selected.append(item)
            seen_pages.add(page_key)

        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        selected_ids = {
            item["chunk_index"] for item in selected
        }

        for item in candidates:
            if item["chunk_index"] not in selected_ids:
                selected.append(item)

            if len(selected) >= top_k:
                break

    return selected


# ============================================================
# CHAT HISTORY / FOLLOW-UP QUESTIONS
# ============================================================

def format_chat_history(chat_history, max_messages=6):
    if not chat_history:
        return "No previous conversation."

    lines = []

    for message in chat_history[-max_messages:]:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role in ("user", "assistant"):
            lines.append(f"{role.title()}: {content}")

    return "\n".join(lines)


def rewrite_question(question, chat_history=None):
    if not chat_history:
        return question

    prompt = f"""
Rewrite the latest question as a standalone document-search question.
Resolve references using the conversation. Do not answer the question.
Return only the rewritten question.

Conversation:
{format_chat_history(chat_history)}

Latest question:
{question}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )

        rewritten = response["message"]["content"].strip()
        return rewritten or question

    except Exception:
        return question


# ============================================================
# SOURCE-AWARE ANSWERS
# ============================================================

SOURCE_AWARE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "page": {"type": "integer"},
                },
                "required": ["filename", "page"],
            },
        },
    },
    "required": ["answer", "citations"],
}


def format_sources_for_llm(retrieved_chunks):
    sections = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        sections.append(
            f"[SOURCE {i}]\n"
            f"[DOCUMENT: {chunk['filename']}]\n"
            f"[PAGE: {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(sections)


def validate_citations(citations, retrieved_chunks):
    valid_sources = {
        (chunk["filename"], int(chunk["page"]))
        for chunk in retrieved_chunks
    }

    validated = []

    for citation in citations:
        try:
            filename = citation["filename"]
            page = int(citation["page"])
        except (KeyError, TypeError, ValueError):
            continue

        if (filename, page) in valid_sources:
            item = {"filename": filename, "page": page}

            if item not in validated:
                validated.append(item)

    return validated


def generate_answer(question, retrieved_chunks, chat_history=None):
    if not retrieved_chunks:
        return {
            "answer": (
                "I couldn't find enough relevant information "
                "in the selected document to answer that."
            ),
            "citations": [],
        }

    context = format_sources_for_llm(retrieved_chunks)

    prompt = f"""
You are a helpful document assistant.

Answer using ONLY the supplied document sources.
Do not add outside knowledge.
If the sources are insufficient, say so.
Use exact filenames and page numbers when citing.
Return JSON matching the requested schema.

Conversation:
{format_chat_history(chat_history)}

Sources:
{context}

Question:
{question}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format=SOURCE_AWARE_SCHEMA,
            options={"temperature": 0.3},
        )

        data = json.loads(response["message"]["content"])
        answer = str(data.get("answer", "")).strip()

        if not answer:
            raise ValueError("The model returned an empty answer.")

        citations = validate_citations(
            data.get("citations", []),
            retrieved_chunks,
        )

        return {"answer": answer, "citations": citations}

    except Exception:
        try:
            response = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": (
                        "Answer using only these document sources. "
                        "If the answer is not supported, say so.\n\n"
                        f"{context}\n\nQuestion: {question}"
                    ),
                }],
                options={"temperature": 0.3},
            )

            return {
                "answer": response["message"]["content"].strip(),
                "citations": [],
            }

        except Exception as error:
            return {
                "answer": f"Unable to generate an answer: {error}",
                "citations": [],
            }


# ============================================================
# SUMMARIZATION
# ============================================================

def build_summary_sections(pages, max_chars=12000):
    sections = []
    current_text = ""
    current_label = None

    for page in pages:
        label = f"{page['filename']} — Page {page['page']}"
        text = page["text"]

        if current_text and len(current_text) + len(text) > max_chars:
            sections.append({
                "label": current_label,
                "text": current_text,
            })

            current_text = ""
            current_label = None

        if not current_label:
            current_label = label

        current_text += f"\n\n[{label}]\n{text}"

    if current_text:
        sections.append({
            "label": current_label,
            "text": current_text,
        })

    return sections


def summarize_section(section_text, summary_type="Detailed"):
    if summary_type == "Short":
        instruction = (
            "Write a concise summary of the most important information."
        )
    elif summary_type == "Key Points":
        instruction = (
            "Summarize the important facts and ideas as bullet points."
        )
    else:
        instruction = (
            "Write a detailed but readable summary of key ideas and facts."
        )

    prompt = f"""
{instruction}
Use only the supplied text. Do not add outside knowledge.

TEXT:
{section_text}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )

        return response["message"]["content"].strip()

    except Exception as error:
        return f"Unable to summarize section: {error}"


def summarize_document(pages, summary_type="Detailed", max_sections=8):
    sections = build_summary_sections(pages)[:max_sections]

    return [
        {
            "label": section["label"],
            "summary": summarize_section(
                section["text"], summary_type
            ),
        }
        for section in sections
    ]


def summarize_page_range(
    pages,
    start_page,
    end_page,
    summary_type="Detailed",
    filename=None,
):
    selected_pages = [
        page for page in pages
        if (
            start_page <= page["page"] <= end_page
            and (
                filename is None
                or page["filename"] == filename
            )
        )
    ]

    return [
        {
            "label": section["label"],
            "summary": summarize_section(
                section["text"], summary_type
            ),
        }
        for section in build_summary_sections(selected_pages)
    ]


def summarize_chapter(pages, chapter, summary_type="Detailed"):
    chapter_pages = get_chapter_pages(pages, chapter)

    return [
        {
            "label": section["label"],
            "summary": summarize_section(
                section["text"], summary_type
            ),
        }
        for section in build_summary_sections(chapter_pages)
    ]