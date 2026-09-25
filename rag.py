import os
import time
import fitz
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


# ==========================================================
# ENVIRONMENT / GEMINI
# ==========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

client = genai.Client(api_key=api_key)


# ==========================================================
# EMBEDDING MODEL
# ==========================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# CHROMADB
# ==========================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documind_knowledge_base"
)


# ==========================================================
# RETRIEVAL SETTINGS
# ==========================================================

# ChromaDB uses cosine distance for this collection.
# Smaller distance = more similar.
#
# This is an initial threshold.
# We will tune it after testing.

MAX_DISTANCE = 0.75


# ==========================================================
# PDF TEXT EXTRACTION
# ==========================================================

def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    return text


# ==========================================================
# CHUNKING
# ==========================================================

def create_chunks(text):

    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


# ==========================================================
# STORE DOCUMENT
# ==========================================================

def store_chunks(chunks, source):

    # Remove previous version of this document
    try:

        collection.delete(
            where={
                "source": source
            }
        )

    except Exception:

        pass

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    )

    ids = [
        f"{source}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source,
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return len(chunks)


# ==========================================================
# RETRIEVE FROM DOCUMENTS
# ==========================================================

def retrieve_context(
    query,
    top_k=5,
    sources=None
):

    query_embedding = embedding_model.encode(
        query
    )

    # ------------------------------------------------------
    # Search selected documents
    # ------------------------------------------------------

    if sources:

        if len(sources) == 1:

            where_filter = {
                "source": sources[0]
            }

        else:

            where_filter = {
                "$or": [
                    {
                        "source": source
                    }
                    for source in sources
                ]
            }

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            where=where_filter,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    # ------------------------------------------------------
    # Search all documents
    # ------------------------------------------------------

    else:

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    # ------------------------------------------------------
    # Apply relevance threshold
    # ------------------------------------------------------

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        if distance <= MAX_DISTANCE:

            filtered_documents.append(
                document
            )

            filtered_metadatas.append(
                metadata
            )

            filtered_distances.append(
                distance
            )

    # Return filtered results
    return {
        "documents": filtered_documents,
        "metadatas": filtered_metadatas,
        "distances": filtered_distances
    }


# ==========================================================
# ANSWER QUESTION
# ==========================================================

def answer_question(
    query,
    top_k=5,
    sources=None
):

    results = retrieve_context(
        query,
        top_k,
        sources
    )

    retrieved_chunks = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]

    # ======================================================
    # NO RELEVANT INFORMATION
    # ======================================================

    if not retrieved_chunks:

        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "model": "Not used"
        }

    # ======================================================
    # BUILD CONTEXT
    # ======================================================

    context = "\n\n".join(
        retrieved_chunks
    )

    # ======================================================
    # RAG PROMPT
    # ======================================================

    prompt = f"""
You are DocuMind, a helpful document
question-answering assistant.

Answer the user's question using ONLY
the information provided in the context below.

If the answer is not present in the context,
say exactly:

"I don't have enough information in the provided documents."

Do not use outside knowledge.

Do not invent information.

If information comes from multiple documents,
combine the relevant information clearly.

Keep the answer clear and easy to understand.

Context:
{context}

Question:
{query}

Answer:
"""

    # ======================================================
    # GEMINI MODELS
    # ======================================================

    models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite"
    ]

    response = None
    used_model = None

    # ======================================================
    # RETRY LOGIC
    # ======================================================

    for model in models:

        for attempt in range(3):

            try:

                print(
                    f"Trying {model} "
                    f"(attempt {attempt + 1}/3)..."
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                print(
                    f"Success with {model}"
                )

                used_model = model

                break

            except Exception:

                print(
                    f"{model} temporarily unavailable."
                )

                if attempt < 2:

                    time.sleep(5)

        if response is not None:

            break

    # ======================================================
    # GEMINI FAILED
    # ======================================================

    if response is None:

        raise Exception(
            "Gemini is temporarily unavailable. "
            "Please try again later."
        )

    # ======================================================
    # SOURCE INFORMATION
    # ======================================================

    sources_info = []

    for i in range(
        len(retrieved_chunks)
    ):

        sources_info.append(
            {
                "source": metadatas[i]["source"],
                "chunk_id": metadatas[i]["chunk_id"],
                "distance": round(
                    distances[i],
                    4
                )
            }
        )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {
        "answer": response.text,
        "sources": sources_info,
        "retrieved_chunks": retrieved_chunks,
        "model": used_model
    }