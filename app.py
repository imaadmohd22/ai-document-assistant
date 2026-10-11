import streamlit as st
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak,
)

from processor import (
    extract_pdf_pages,
    chunk_text,
    create_embeddings,
    create_faiss_index,
    retrieve_relevant_chunks,
    rewrite_question,
    generate_answer,
    detect_chapters,
    summarize_document,
    summarize_page_range,
    summarize_chapter,
    save_document_index,
    load_document_index,
    delete_document_index,
    persistent_index_exists,
    add_document_to_collection,
    remove_document_from_collection,
)


# ============================================================
# PDF REPORT HELPERS
# ============================================================

def register_report_font():
    """Register a Unicode-capable font when one is available."""
    candidates = [
        (r"C:\Windows\Fonts\arial.ttf", "ReportArial"),
        (r"C:\Windows\Fonts\calibri.ttf", "ReportCalibri"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "ReportDejaVu"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "ReportLiberation"),
    ]
    for path, name in candidates:
        if os.path.exists(path):
            try:
                if name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                pass
    return "Helvetica"


def format_report_text(value):
    """Escape text for ReportLab Paragraph while preserving line breaks."""
    value = "" if value is None else str(value)
    value = escape(value)
    return value.replace("\r\n", "\n").replace("\n", "<br/>")


def make_report_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9DEE7"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#687386"))
    canvas.drawString(18 * mm, 10 * mm, "AI Document Assistant")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def report_styles(font_name):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName=font_name,
        fontSize=21, leading=27, alignment=TA_CENTER,
        textColor=colors.HexColor("#172033"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportMeta", parent=styles["Normal"], fontName=font_name,
        fontSize=9, leading=13, textColor=colors.HexColor("#596579"),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportHeading", parent=styles["Heading2"], fontName=font_name,
        fontSize=13, leading=17, textColor=colors.HexColor("#244A82"),
        spaceBefore=10, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontName=font_name,
        fontSize=9.5, leading=14, textColor=colors.HexColor("#202938"),
        spaceAfter=8,
    ))
    return styles


def build_summary_pdf(
    summary_items, title, document_name, summary_style, page_range=None
):
    buffer = BytesIO()
    font_name = register_report_font()
    styles = report_styles(font_name)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=title,
        author="AI Document Assistant",
    )

    story = [
        Paragraph(format_report_text(title), styles["ReportTitle"]),
        Paragraph(
            f"<b>Document:</b> {format_report_text(document_name)}",
            styles["ReportMeta"],
        ),
        Paragraph(
            f"<b>Summary style:</b> {format_report_text(summary_style)}",
            styles["ReportMeta"],
        ),
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["ReportMeta"],
        ),
    ]

    if page_range:
        story.append(Paragraph(
            f"<b>Page range:</b> {format_report_text(page_range)}",
            styles["ReportMeta"],
        ))

    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#D9DEE7"),
    ))
    story.append(Spacer(1, 4 * mm))

    for item in summary_items or []:
        label = item.get("label", "Summary")
        summary = item.get("summary", "")
        story.append(Paragraph(
            format_report_text(label), styles["ReportHeading"]
        ))
        story.append(Paragraph(
            format_report_text(summary), styles["ReportBody"]
        ))

    if not summary_items:
        story.append(Paragraph(
            "No summary content was generated.", styles["ReportBody"]
        ))

    doc.build(
        story,
        onFirstPage=make_report_footer,
        onLaterPages=make_report_footer,
    )
    buffer.seek(0)
    return buffer.getvalue()


def build_chat_pdf(chat_history, document_scope):
    buffer = BytesIO()
    font_name = register_report_font()
    styles = report_styles(font_name)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="AI Document Assistant Chat Report",
        author="AI Document Assistant",
    )

    story = [
        Paragraph(
            "AI Document Assistant — Chat Report",
            styles["ReportTitle"],
        ),
        Paragraph(
            f"<b>Search scope:</b> {format_report_text(document_scope)}",
            styles["ReportMeta"],
        ),
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["ReportMeta"],
        ),
        Spacer(1, 5 * mm),
    ]

    for message in chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role not in ("user", "assistant"):
            continue

        story.append(Paragraph(
            format_report_text(role.title()), styles["ReportHeading"]
        ))
        story.append(Paragraph(
            format_report_text(content), styles["ReportBody"]
        ))

        citations = message.get("citations", [])
        if citations:
            citation_text = "; ".join(
                f"{c.get('filename', 'Unknown')} — "
                f"Page {c.get('page', '?')}"
                for c in citations
            )
            story.append(Paragraph(
                f"<b>Sources:</b> {format_report_text(citation_text)}",
                styles["ReportMeta"],
            ))

        story.append(Spacer(1, 2 * mm))

    doc.build(
        story,
        onFirstPage=make_report_footer,
        onLaterPages=make_report_footer,
    )
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "pages": [],
    "chunks": [],
    "embeddings": None,
    "index": None,
    "processed": False,
    "processed_files": [],
    "chat_history": [],
    "chapters": [],
    "last_document_filter": "All Documents",
    "storage_loaded": False,
    "chunk_size": 1000,
    "overlap": 150,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# LOAD PERSISTENT COLLECTION
# ============================================================

if not st.session_state.storage_loaded:
    saved_data = load_document_index()

    if saved_data:
        for key in [
            "pages",
            "chunks",
            "embeddings",
            "index",
            "processed_files",
            "chapters",
            "chunk_size",
            "overlap",
        ]:
            st.session_state[key] = saved_data[key]

        st.session_state.processed = bool(saved_data["chunks"])

    st.session_state.storage_loaded = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_collection(result):
    for key, value in result.items():
        st.session_state[key] = value

    st.session_state.processed = bool(st.session_state.chunks)
    st.session_state.chat_history = []
    st.session_state.last_document_filter = "All Documents"


def process_initial_uploads(files, chunk_size, overlap):
    pages = []
    chunks = []

    for uploaded_file in files:
        filename = uploaded_file.name
        extracted_pages = extract_pdf_pages(uploaded_file)

        if not extracted_pages:
            st.warning(
                f"No extractable text found in {filename}."
            )
            continue

        # Same-name uploads in this batch replace earlier copies.
        pages = [
            page for page in pages
            if page["filename"] != filename
        ]
        chunks = [
            chunk for chunk in chunks
            if chunk["filename"] != filename
        ]

        pages.extend(extracted_pages)

        chunks.extend(
            chunk_text(
                extracted_pages,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    if not chunks:
        raise ValueError(
            "No readable text was found. Scanned PDFs may require OCR."
        )

    embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    processed_files = sorted({
        page["filename"] for page in pages
    })
    chapters = detect_chapters(pages)

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
# HEADER
# ============================================================

st.title("📚 AI Document Assistant")

st.caption(
    "Ask questions about PDFs, summarize content, "
    "view sources, and manage your document collection."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=st.session_state.chunk_size,
        step=100,
    )

    overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=st.session_state.overlap,
        step=50,
    )

    top_k = st.slider(
        "Context Sections",
        min_value=2,
        max_value=10,
        value=5,
    )

    st.session_state.chunk_size = chunk_size
    st.session_state.overlap = overlap

    st.divider()
    st.subheader("📄 Search Scope")

    document_options = [
        "All Documents",
        *sorted(st.session_state.processed_files),
    ]

    current_filter = st.session_state.last_document_filter

    if current_filter not in document_options:
        current_filter = "All Documents"

    document_filter = st.selectbox(
        "Search in",
        document_options,
        index=document_options.index(current_filter),
    )

    if document_filter != st.session_state.last_document_filter:
        st.session_state.last_document_filter = document_filter
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("💾 Storage")

    if persistent_index_exists():
        st.success("Saved collection available")
    else:
        st.info("No saved collection available")

    if st.button(
        "🗑️ Delete Saved Collection",
        use_container_width=True,
    ):
        delete_document_index()

        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value

        st.session_state.storage_loaded = True
        st.rerun()

    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Local model: llama3.2:3b")


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

st.header("1️⃣ Document Management")

add_tab, remove_tab = st.tabs([
    "➕ Add / Process PDFs",
    "➖ Remove PDF",
])


# ============================================================
# ADD DOCUMENTS
# ============================================================

with add_tab:
    st.subheader("Add PDFs to your collection")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="document_uploader",
    )

    if uploaded_files:
        st.caption(
            "A PDF with the same filename will replace the old version."
        )

        if st.button(
            "🚀 Add / Process Documents",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Extracting text, creating chunks, and indexing..."
                ):
                    result = process_initial_uploads(
                        uploaded_files,
                        chunk_size,
                        overlap,
                    )

                apply_collection(result)
                st.success(
                    f"Processed {len(st.session_state.processed_files)} "
                    "document(s) successfully."
                )
                st.rerun()

            except Exception as error:
                st.error(f"Could not process documents: {error}")


# ============================================================
# REMOVE DOCUMENTS
# ============================================================

with remove_tab:
    st.subheader("Remove a document")

    if not st.session_state.processed_files:
        st.info("No documents are currently indexed.")
    else:
        file_to_remove = st.selectbox(
            "Select document to remove",
            st.session_state.processed_files,
            key="remove_document_select",
        )

        if st.button(
            "🗑️ Remove Selected PDF",
            use_container_width=True,
        ):
            try:
                with st.spinner("Removing document and rebuilding index..."):
                    result = remove_document_from_collection(
                        file_to_remove,
                        st.session_state.pages,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        st.session_state.chunk_size,
                        st.session_state.overlap,
                    )

                apply_collection(result)
                save_document_index(
                    pages=st.session_state.pages,
                    chunks=st.session_state.chunks,
                    embeddings=st.session_state.embeddings,
                    index=st.session_state.index,
                    processed_files=st.session_state.processed_files,
                    chapters=st.session_state.chapters,
                    chunk_size=st.session_state.chunk_size,
                    overlap=st.session_state.overlap,
                )
                st.success(f"Removed {file_to_remove}.")
                st.rerun()

            except Exception as error:
                st.error(f"Could not remove document: {error}")


# ============================================================
# COLLECTION STATUS
# ============================================================

st.header("2️⃣ Collection Status")

if st.session_state.processed:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Documents",
        len(st.session_state.processed_files),
    )
    col2.metric(
        "Pages",
        len(st.session_state.pages),
    )
    col3.metric(
        "Chunks",
        len(st.session_state.chunks),
    )
    col4.metric(
        "Chapters",
        len(st.session_state.chapters),
    )

    with st.expander("📄 Current Documents"):
        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")

    with st.expander("📖 Detected Chapters"):
        if st.session_state.chapters:
            for chapter in st.session_state.chapters:
                st.write(
                    f"**{chapter['filename']}** — "
                    f"{chapter['title']} "
                    f"(Pages {chapter['start_page']}–"
                    f"{chapter['end_page']})"
                )
        else:
            st.write("No chapter headings detected.")


# ============================================================
# APPLICATION FEATURES
# ============================================================

if st.session_state.processed:
    chat_tab, summary_tab, chapter_tab, preview_tab = st.tabs([
        "💬 Ask Questions",
        "📝 Summarize",
        "📖 Chapters",
        "🔍 Chunk Preview",
    ])

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    with chat_tab:
        st.subheader("Ask questions about your documents")
        st.info(f"Search scope: {document_filter}")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                citations = message.get("citations", [])

                if citations:
                    with st.expander("📚 Sources"):
                        for citation in citations:
                            st.write(
                                f"📄 {citation['filename']} — "
                                f"Page {citation['page']}"
                            )

        question = st.chat_input(
            "Ask something about your PDFs..."
        )

        if question:
            previous_history = (
                st.session_state.chat_history.copy()
            )

            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.spinner("Searching documents..."):
                    interpreted_question = rewrite_question(
                        question,
                        previous_history,
                    )

                    retrieved_chunks = retrieve_relevant_chunks(
                        interpreted_question,
                        st.session_state.chunks,
                        st.session_state.index,
                        top_k=top_k,
                        selected_document=(
                            None
                            if document_filter == "All Documents"
                            else document_filter
                        ),
                    )

                with st.spinner("Generating answer..."):
                    result = generate_answer(
                        question,
                        retrieved_chunks,
                        previous_history,
                    )

                answer = result["answer"]
                citations = result["citations"]

            except Exception as error:
                answer = f"Unable to answer the question: {error}"
                citations = []
                retrieved_chunks = []

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
            })

            with st.chat_message("assistant"):
                st.markdown(answer)

                if citations:
                    st.markdown("**📚 Sources**")

                    for citation in citations:
                        st.write(
                            f"📄 {citation['filename']} — "
                            f"Page {citation['page']}"
                        )

                with st.expander("🔍 Retrieval Details"):
                    st.write("Original question:", question)
                    st.write(
                        "Interpreted question:",
                        interpreted_question
                        if "interpreted_question" in locals()
                        else question,
                    )
                    st.write("Search scope:", document_filter)
                    st.write(
                        "Retrieved sections:",
                        len(retrieved_chunks),
                    )

                    for i, chunk in enumerate(
                        retrieved_chunks,
                        start=1,
                    ):
                        st.markdown(
                            f"**Source {i}: {chunk['filename']} — "
                            f"Page {chunk['page']}**"
                        )
                        st.caption(
                            f"Semantic: {chunk['semantic_score']:.3f} | "
                            f"Lexical: {chunk['lexical_score']:.3f} | "
                            f"Final: {chunk['final_score']:.3f}"
                        )
                        st.text(chunk["text"][:1000])

        if st.session_state.chat_history:
            report_bytes = build_chat_pdf(
                st.session_state.chat_history,
                document_filter,
            )
            st.download_button(
                label="⬇️ Download Chat Report (PDF)",
                data=report_bytes,
                file_name=(
                    "ai_document_chat_report_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="download_chat_pdf_report",
            )
        else:
            st.info("Ask at least one question before exporting a report.")

    # --------------------------------------------------------
    # SUMMARIZATION
    # --------------------------------------------------------

    with summary_tab:
        st.subheader("📝 Document Summarization")

        summary_mode = st.radio(
            "Summary Scope",
            ["Whole Document", "Page Range"],
            horizontal=True,
        )

        summary_type = st.selectbox(
            "Summary Style",
            ["Detailed", "Short", "Key Points"],
        )

        if summary_mode == "Whole Document":
            summary_scope = st.selectbox(
                "Document",
                ["All Documents"]
                + st.session_state.processed_files,
            )

            if st.button("📝 Generate Summary", type="primary"):
                pages = (
                    st.session_state.pages
                    if summary_scope == "All Documents"
                    else [
                        page for page in st.session_state.pages
                        if page["filename"] == summary_scope
                    ]
                )

                with st.spinner("Generating summary..."):
                    summaries = summarize_document(
                        pages,
                        summary_type,
                    )

                for item in summaries:
                    st.markdown(f"### {item['label']}")
                    st.write(item["summary"])
                    st.divider()

                report_bytes = build_summary_pdf(
                    summaries,
                    title="Document Summary Report",
                    document_name=summary_scope,
                    summary_style=summary_type,
                )
                st.download_button(
                    "⬇️ Download Summary PDF",
                    data=report_bytes,
                    file_name="document_summary.pdf",
                    mime="application/pdf",
                    key="download_whole_summary_pdf",
                )

        else:
            selected_file = st.selectbox(
                "Document",
                st.session_state.processed_files,
                key="range_document",
            )

            file_pages = [
                page for page in st.session_state.pages
                if page["filename"] == selected_file
            ]

            min_page = min(page["page"] for page in file_pages)
            max_page = max(page["page"] for page in file_pages)

            col1, col2 = st.columns(2)

            with col1:
                start_page = st.number_input(
                    "Start Page",
                    min_value=min_page,
                    max_value=max_page,
                    value=min_page,
                )

            with col2:
                end_page = st.number_input(
                    "End Page",
                    min_value=min_page,
                    max_value=max_page,
                    value=max_page,
                )

            if start_page > end_page:
                st.error("Start Page cannot exceed End Page.")

            elif st.button(
                "📝 Summarize Page Range",
                type="primary",
            ):
                with st.spinner("Generating summary..."):
                    summaries = summarize_page_range(
                        st.session_state.pages,
                        int(start_page),
                        int(end_page),
                        summary_type,
                        filename=selected_file,
                    )

                for item in summaries:
                    st.markdown(f"### {item['label']}")
                    st.write(item["summary"])
                    st.divider()

                report_bytes = build_summary_pdf(
                    summaries,
                    title="Page Range Summary Report",
                    document_name=selected_file,
                    summary_style=summary_type,
                    page_range=f"{int(start_page)}–{int(end_page)}",
                )
                st.download_button(
                    "⬇️ Download Page Range Summary PDF",
                    data=report_bytes,
                    file_name="page_range_summary.pdf",
                    mime="application/pdf",
                    key="download_page_range_summary_pdf",
                )

    # --------------------------------------------------------
    # CHAPTERS
    # --------------------------------------------------------

    with chapter_tab:
        st.subheader("📖 Chapter Summaries")

        chapters = st.session_state.chapters

        if not chapters:
            st.info(
                "No chapter headings were detected automatically."
            )

        else:
            labels = [
                f"{chapter['filename']} | {chapter['title']} | "
                f"Pages {chapter['start_page']}–{chapter['end_page']}"
                for chapter in chapters
            ]

            selected_chapter_index = st.selectbox(
                "Select Chapter",
                range(len(chapters)),
                format_func=lambda i: labels[i],
            )

            chapter_style = st.selectbox(
                "Chapter Summary Style",
                ["Detailed", "Short", "Key Points"],
            )

            if st.button(
                "📝 Summarize Selected Chapter",
                type="primary",
            ):
                chapter = chapters[selected_chapter_index]

                with st.spinner("Summarizing chapter..."):
                    summaries = summarize_chapter(
                        st.session_state.pages,
                        chapter,
                        chapter_style,
                    )

                st.markdown(
                    f"## {chapter['filename']} — {chapter['title']}"
                )

                for item in summaries:
                    st.markdown(f"### {item['label']}")
                    st.write(item["summary"])

                report_bytes = build_summary_pdf(
                    summaries,
                    title="Chapter Summary Report",
                    document_name=(
                        f"{chapter['filename']} — {chapter['title']}"
                    ),
                    summary_style=chapter_style,
                    page_range=(
                        f"{chapter['start_page']}–{chapter['end_page']}"
                    ),
                )
                st.download_button(
                    "⬇️ Download Chapter Summary PDF",
                    data=report_bytes,
                    file_name="chapter_summary.pdf",
                    mime="application/pdf",
                    key="download_chapter_summary_pdf",
                )

            if st.button("📚 Generate All Chapter Summaries"):
                for chapter in chapters:
                    st.markdown(
                        f"## {chapter['filename']} — "
                        f"{chapter['title']}"
                    )

                    with st.spinner(
                        f"Summarizing {chapter['title']}..."
                    ):
                        summaries = summarize_chapter(
                            st.session_state.pages,
                            chapter,
                            chapter_style,
                        )

                    for item in summaries:
                        st.markdown(f"### {item['label']}")
                        st.write(item["summary"])

                    st.divider()

    # --------------------------------------------------------
    # CHUNK PREVIEW
    # --------------------------------------------------------

    with preview_tab:
        st.subheader("🔍 Chunk Preview")

        for i, chunk in enumerate(
            st.session_state.chunks[:10],
            start=1,
        ):
            with st.expander(
                f"Chunk {i} — {chunk['filename']} — "
                f"Page {chunk['page']}"
            ):
                st.write(chunk["text"])

else:
    st.info(
        "Upload a PDF above and click "
        "'Add / Process Documents' to get started."
    )