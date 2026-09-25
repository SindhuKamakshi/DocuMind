from pathlib import Path
from rag import extract_text_from_pdf, create_chunks, store_chunks


# Find the PDF
pdf_path = Path("data/documents/ML_merged.pdf")


# Check that the PDF exists
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")


print("PDF found:", pdf_path)


# Extract text
text = extract_text_from_pdf(str(pdf_path))

print("Characters extracted:", len(text))


# Create chunks
chunks = create_chunks(text)

print("Chunks created:", len(chunks))


# Store chunks in ChromaDB
stored = store_chunks(
    chunks,
    pdf_path.name
)

print("Chunks stored in ChromaDB:", stored)

print("\nIngestion completed successfully!")