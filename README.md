# 🧠 DocuMind — Personal Document RAG Assistant

DocuMind is a **Retrieval-Augmented Generation (RAG)** based document question-answering application.

It allows users to upload PDF documents and ask questions about their content using natural language. Instead of relying only on the language model's general knowledge, DocuMind retrieves relevant information from the uploaded documents and uses it as context for generating grounded answers.

---

## 📌 Project Overview

Reading and searching through large PDF documents manually can be time-consuming.

DocuMind provides a simple conversational interface where users can:

- 📄 Upload PDF documents
- ✂️ Automatically split documents into meaningful chunks
- 🧠 Convert text into vector embeddings
- 🗄️ Store embeddings in ChromaDB
- 🔎 Perform semantic similarity search
- 🤖 Generate answers using Gemini
- 📚 View the documents used as sources
- 🔍 View the relevant passages retrieved from the documents
- 📖 Search across multiple documents
- 🎯 Search within a specific document

---

## 🎯 Problem Statement

Traditional document search mainly depends on exact keyword matching.

For example, searching for:

> "What is discrete data?"

may fail to find relevant information if the document uses different wording.

DocuMind uses **semantic search**, allowing the system to retrieve information based on meaning rather than only exact keywords.

---

# 🔄 RAG Architecture

The DocuMind pipeline follows:

```text
                    PDF DOCUMENT
                         │
                         ▼
                ┌─────────────────┐
                │ Text Extraction │
                │    PyMuPDF      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Text Cleaning  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Chunking     │
                │ Recursive Split │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │
                │ MiniLM-L6-v2    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                │ Vector Database │
                └────────┬────────┘
                         │
                         │
User Question ───────────┤
                         ▼
                ┌─────────────────┐
                │ Query Embedding │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Semantic Search │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Relevant Chunks │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Gemini      │
                │      LLM        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Grounded Answer │
                │   + Sources     │
                └─────────────────┘
                ✨ Features
📤 PDF Upload

Users can upload one or multiple PDF documents through the Streamlit interface.

Each uploaded document is:

Saved locally
Processed using PyMuPDF
Converted into text
Split into chunks
Embedded
Stored in ChromaDB
✂️ Recursive Chunking

DocuMind uses:

RecursiveCharacterTextSplitter

with:

Chunk size: 1000
Chunk overlap: 200

The splitter attempts to preserve meaningful text boundaries using separators such as:

Paragraph
↓
Line
↓
Sentence
↓
Word
↓
Character

Chunk overlap helps preserve context between neighboring chunks.

🧠 Semantic Embeddings

DocuMind uses:

sentence-transformers/all-MiniLM-L6-v2

to convert text into numerical vectors.

The embedding dimension used by the model is:

384

These vectors allow semantically similar pieces of text to be retrieved even when they do not contain exactly the same words as the user's question.

🗄️ ChromaDB Vector Database

The generated embeddings and their corresponding document chunks are stored in ChromaDB.

Each stored chunk contains:

Document text
Source document
Chunk ID
Vector embedding

ChromaDB is then used to perform similarity search during question answering.

🔎 Semantic Retrieval

When a user asks a question:

The question is converted into an embedding.
ChromaDB searches for similar document chunks.
The most relevant chunks are retrieved.
A relevance threshold is applied.
The remaining chunks are provided to Gemini as context.

The current initial relevance threshold is:

Maximum cosine distance = 0.75

Smaller cosine distance means greater similarity.

The threshold is configurable and can be tuned using evaluation results.

🤖 Gemini-Powered Answer Generation

DocuMind uses Google's Gemini API to generate the final answer.

The application is instructed to:

Use only the retrieved document context
Avoid using outside knowledge
Avoid inventing information
Clearly answer the user's question
Return a fixed response when the documents do not contain enough information

If relevant information cannot be found, DocuMind responds:

I don't have enough information in the provided documents.
📚 Source Display

For every generated answer, DocuMind can display the source documents used during retrieval.

Users can also expand:

🔎 View relevant passages

to inspect the retrieved document sections.

This makes the RAG process more transparent.

📖 Multi-Document Search

DocuMind supports multiple PDF documents.

Users can choose between:

All Documents

Search across all processed documents.

One Document

Restrict the search to a selected PDF.

This allows users to control the scope of their questions.

💬 Example
User Question
What is discrete data?
DocuMind Process
Question
   ↓
Question Embedding
   ↓
ChromaDB Semantic Search
   ↓
Relevant Document Chunks
   ↓
Gemini
   ↓
Grounded Answer

The interface also displays the source document and relevant passages used for the answer.

🛠️ Technology Stack
Technology	Purpose
Python	Application development
Streamlit	Web application interface
PyMuPDF	PDF text extraction
LangChain Text Splitters	Recursive document chunking
Sentence Transformers	Text embeddings
all-MiniLM-L6-v2	Embedding model
ChromaDB	Vector database
Google Gemini	Answer generation
python-dotenv	Environment variable management
Git & GitHub	Version control
📁 Project Structure
DocuMind/
│
├── app.py
│
├── rag.py
│
├── evaluate_rag.py
│
├── test_ingestion.py
│
├── test_retrieval.py
│
├── requirements.txt
│
├── .gitignore
│
└── data/
    └── documents/
        └── .gitkeep
app.py

Contains the Streamlit user interface.

Responsibilities include:

PDF upload
Document library
Search scope selection
Chat interface
Source display
Relevant passage display
Conversation history
rag.py

Contains the core RAG pipeline.

Responsibilities include:

PDF extraction
Text cleaning
Chunking
Embedding generation
ChromaDB storage
Semantic retrieval
Relevance filtering
Gemini answer generation
evaluate_rag.py

Runs a collection of questions to evaluate the RAG pipeline, including both document-related questions and questions outside the document's knowledge.

test_ingestion.py

Tests:

PDF → Text → Chunks → ChromaDB
test_retrieval.py

Tests semantic retrieval from the vector database.

requirements.txt

Contains the Python dependencies required to run the project.

⚙️ Installation
1. Clone the repository
git clone https://github.com/SindhuKamakshi/DocuMind.git

Move into the project directory:

cd DocuMind
2. Create a virtual environment
Windows
python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
🔑 Gemini API Key Setup

DocuMind requires a Gemini API key.

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Do not commit the .env file to GitHub.

The project .gitignore already excludes environment files.

▶️ Running the Application

Start the Streamlit application:

streamlit run app.py

Streamlit will provide a local URL similar to:

http://localhost:8501

Open the URL in your browser.

📄 Using DocuMind
Step 1

Upload one or more PDF documents.

Step 2

Click:

🚀 Process Documents
Step 3

Select the search scope:

All documents

or:

One document
Step 4

Ask a question.

For example:

What is supervised learning?
Step 5

DocuMind retrieves relevant passages and generates an answer using Gemini.

Step 6

Expand:

🔎 View relevant passages

to inspect the retrieved context.

🧪 Evaluation

DocuMind includes an evaluation script:

python evaluate_rag.py

The evaluation includes questions such as:

What is discrete data?
What is continuous data?
What is supervised learning?
What is unsupervised learning?
What is classification?
What is regression?
What is clustering?
Explain the difference between supervised and unsupervised learning.
What is the capital of France?
Who is the current President of the United States?

The evaluation helps test:

Relevant document retrieval
Semantic search
Answer generation
Out-of-document question handling
Relevance filtering
🔐 Security

The following files are intentionally excluded from Git:

.env
venv/
chroma_db/
__pycache__/
*.pdf

This prevents API keys, local environments, generated vector databases, and uploaded documents from being committed to the repository.

🚀 Current RAG Pipeline

The complete implementation currently follows:

PDF
 │
 ▼
PyMuPDF
 │
 ▼
Text Cleaning
 │
 ▼
Recursive Chunking
 │
 ▼
MiniLM Embeddings
 │
 ▼
ChromaDB
 │
 ▼
Semantic Retrieval
 │
 ▼
Relevance Filtering
 │
 ▼
Retrieved Context
 │
 ▼
Gemini
 │
 ▼
Answer + Sources
🔮 Future Improvements

Possible future improvements include:

📑 Support for additional document formats
🧠 Improved retrieval strategies
🔄 Reranking retrieved chunks
📊 More comprehensive RAG evaluation
💾 Persistent external vector database
🔐 User authentication
👥 Multi-user document collections
📈 Retrieval and answer-quality metrics
🌐 Production deployment
🧩 Advanced conversational memory
📌 Page-level source citations
🖼️ Support for image-based/scanned PDFs
🔍 Hybrid keyword + semantic search
🎓 Learning Objectives

This project was built to practice the core concepts behind modern RAG systems:

Text preprocessing
Document chunking
Embeddings
Vector representations
Cosine similarity
Vector databases
Semantic search
Retrieval-Augmented Generation
LLM prompting
Grounded question answering
Streamlit application development
Git and GitHub
AI application deployment
👩‍💻 Author

Sindhu Kamakshi

B.Tech Computer Science & Engineering — AIML

⭐ Project

DocuMind — Personal Document RAG Assistant

Built using Python, Streamlit, ChromaDB, Sentence Transformers, and Google Gemini.