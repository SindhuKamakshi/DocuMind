import streamlit as st
from pathlib import Path

from rag import (
    extract_text_from_pdf,
    create_chunks,
    store_chunks,
    answer_question
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="DocuMind | Document AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# DOCUMENT FOLDER
# ==========================================================

DOCUMENTS_FOLDER = Path("data/documents")

DOCUMENTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ==========================================================
# LOAD DOCUMENTS FROM DISK
# ==========================================================

available_documents = sorted(
    [
        file.name
        for file in DOCUMENTS_FOLDER.glob("*.pdf")
    ]
)

st.session_state["documents"] = available_documents

st.session_state["document_processed"] = (
    len(available_documents) > 0
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* App title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    /* Document card */
    .document-card {
        padding: 0.7rem;
        border-radius: 0.6rem;
        border: 1px solid #dddddd;
        margin-bottom: 0.5rem;
    }

    /* Source card */
    .source-card {
        padding: 0.7rem;
        border-radius: 0.6rem;
        border: 1px solid #dddddd;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Small information text */
    .small-text {
        font-size: 0.85rem;
        color: #777777;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        "## 🧠 DocuMind"
    )

    st.caption(
        "Personal Document AI Assistant"
    )

    st.divider()

    # ------------------------------------------------------
    # UPLOAD SECTION
    # ------------------------------------------------------

    st.markdown(
        "### 📤 Add Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or multiple PDF documents."
    )

    if uploaded_files:

        st.info(
            f"{len(uploaded_files)} "
            f"document(s) selected"
        )

        if st.button(
            "🚀 Process Documents",
            use_container_width=True,
            type="primary"
        ):

            processed_documents = []
            total_chunks = 0

            progress = st.progress(
                0,
                text="Starting document processing..."
            )

            total_files = len(
                uploaded_files
            )

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                progress.progress(
                    index / total_files,
                    text=f"Processing {uploaded_file.name}..."
                )

                pdf_path = (
                    DOCUMENTS_FOLDER
                    / uploaded_file.name
                )

                # Save PDF
                with open(
                    pdf_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                # Extract text
                text = extract_text_from_pdf(
                    str(pdf_path)
                )

                if not text.strip():

                    st.warning(
                        f"⚠️ {uploaded_file.name} "
                        "does not contain extractable text."
                    )

                    continue

                # Create chunks
                chunks = create_chunks(
                    text
                )

                # Store chunks
                stored = store_chunks(
                    chunks,
                    uploaded_file.name
                )

                processed_documents.append(
                    uploaded_file.name
                )

                total_chunks += stored

            progress.progress(
                1.0,
                text="Processing completed!"
            )

            # Refresh document list
            available_documents = sorted(
                [
                    file.name
                    for file in DOCUMENTS_FOLDER.glob("*.pdf")
                ]
            )

            st.session_state[
                "documents"
            ] = available_documents

            st.session_state[
                "document_processed"
            ] = len(
                available_documents
            ) > 0

            # Clear previous chat
            st.session_state[
                "messages"
            ] = []

            if processed_documents:

                st.success(
                    f"✅ Processed "
                    f"{len(processed_documents)} "
                    f"document(s) • "
                    f"{total_chunks:,} knowledge sections"
                )

                st.rerun()

    # ------------------------------------------------------
    # DOCUMENT LIBRARY
    # ------------------------------------------------------

    st.divider()

    st.markdown(
        "### 📚 Document Library"
    )

    if st.session_state["documents"]:

        for document in st.session_state[
            "documents"
        ]:

            st.markdown(
                f"""
                <div class="document-card">
                    📄 {document}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.caption(
            "No documents available."
        )

    # ------------------------------------------------------
    # SEARCH SCOPE
    # ------------------------------------------------------

    if st.session_state["documents"]:

        st.divider()

        st.markdown(
            "### 🔍 Search Scope"
        )

        search_option = st.radio(
            "Search through:",
            [
                "All documents",
                "One document"
            ],
            label_visibility="collapsed"
        )

        if search_option == "One document":

            selected_document = st.selectbox(
                "Choose document:",
                st.session_state[
                    "documents"
                ]
            )

        else:

            selected_document = None

    else:

        search_option = "All documents"
        selected_document = None

    # ------------------------------------------------------
    # CHAT CONTROLS
    # ------------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state[
            "messages"
        ] = []

        st.rerun()

    st.divider()

    st.caption(
        "RAG-powered document question answering"
    )


# ==========================================================
# MAIN HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🧠 DocuMind</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Chat with your documents using semantic search and AI.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# DASHBOARD INFORMATION
# ==========================================================

if st.session_state["documents"]:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 Documents",
            len(
                st.session_state[
                    "documents"
                ]
            )
        )

    with col2:

        st.metric(
            "💬 Questions",
            sum(
                1
                for message in st.session_state["messages"]
                if message["role"] == "user"
            )
        )

    with col3:

        st.metric(
            "🔎 Search",
            "Semantic"
        )


st.divider()


# ==========================================================
# CHAT HEADER
# ==========================================================

st.markdown(
    "### 💬 Chat with your documents"
)

if not st.session_state["documents"]:

    st.info(
        "👈 Upload and process a PDF from the sidebar "
        "to start chatting with your documents."
    )


# ==========================================================
# PREVIOUS MESSAGES
# ==========================================================

for message in st.session_state[
    "messages"
]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown(
                "**📚 Sources**"
            )

            displayed_sources = set()

            for source in message[
                "sources"
            ]:

                source_name = source[
                    "source"
                ]

                if source_name not in displayed_sources:

                    displayed_sources.add(
                        source_name
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">
                            📄 {source_name}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # --------------------------------------------------
            # RELEVANT PASSAGES
            # --------------------------------------------------

            if message.get(
                "retrieved_chunks"
            ):

                with st.expander(
                    "🔎 View relevant passages"
                ):

                    for index, passage in enumerate(
                        message[
                            "retrieved_chunks"
                        ],
                        start=1
                    ):

                        st.markdown(
                            f"**Relevant passage {index}**"
                        )

                        st.write(
                            passage
                        )

                        if index < len(
                            message[
                                "retrieved_chunks"
                            ]
                        ):

                            st.divider()

            st.caption(
                f"Generated using: "
                f"{message['model']}"
            )


# ==========================================================
# CHAT INPUT
# ==========================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ==========================================================
# PROCESS QUESTION
# ==========================================================

if question:

    if not st.session_state[
        "document_processed"
    ]:

        st.warning(
            "Please upload and process at least one PDF first."
        )

        st.stop()

    # Determine search scope
    if search_option == "One document":

        search_sources = [
            selected_document
        ]

    else:

        search_sources = st.session_state[
            "documents"
        ]

    # ------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )

    st.session_state[
        "messages"
    ].append(
        {
            "role": "user",
            "content": question
        }
    )

    # ------------------------------------------------------
    # ASSISTANT RESPONSE
    # ------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            try:

                result = answer_question(
                    question,
                    top_k=5,
                    sources=search_sources
                )

                # Answer
                st.write(
                    result["answer"]
                )

                # Sources
                if result["sources"]:

                    st.markdown(
                        "**📚 Sources**"
                    )

                    displayed_sources = set()

                    for source in result[
                        "sources"
                    ]:

                        source_name = source[
                            "source"
                        ]

                        if source_name not in displayed_sources:

                            displayed_sources.add(
                                source_name
                            )

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    📄 {source_name}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Relevant passages
                if result[
                    "retrieved_chunks"
                ]:

                    with st.expander(
                        "🔎 View relevant passages"
                    ):

                        for index, passage in enumerate(
                            result[
                                "retrieved_chunks"
                            ],
                            start=1
                        ):

                            st.markdown(
                                f"**Relevant passage {index}**"
                            )

                            st.write(
                                passage
                            )

                            if index < len(
                                result[
                                    "retrieved_chunks"
                                ]
                            ):

                                st.divider()

                # Model information
                st.caption(
                    f"Generated using: "
                    f"{result['model']}"
                )

            except Exception as error:

                st.error(
                    "Something went wrong while "
                    "processing your question."
                )

                st.caption(
                    str(error)
                )

                result = {
                    "answer": (
                        "An error occurred while "
                        "processing the question."
                    ),
                    "sources": [],
                    "retrieved_chunks": [],
                    "model": "Error"
                }

    # ------------------------------------------------------
    # SAVE RESPONSE
    # ------------------------------------------------------

    st.session_state[
        "messages"
    ].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": result[
                "retrieved_chunks"
            ],
            "model": result["model"]
        }
    )