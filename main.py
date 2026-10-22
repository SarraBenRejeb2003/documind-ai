
from pathlib import Path
import hashlib
import re

import chromadb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from generate_answer import generate_answer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "company_documents"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".adoc"}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 3
BATCH_SIZE = 500


# ============================================================
# 2. EXTRACTION DU TEXTE
# ============================================================

def read_document(file_path: Path) -> str:
    """Extract readable text from a supported document."""

    extension = file_path.suffix.lower()

    if extension == ".html":
        html = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        soup = BeautifulSoup(html, "html.parser")

        # Remove elements that usually contain scripts or navigation.
        for element in soup(["script", "style", "nav", "footer", "noscript"]):
            element.decompose()

        # Prefer the main content when the HTML provides it.
        content = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        text = content.get_text(separator="\n", strip=True)

    else:
        # TXT, Markdown and AsciiDoc are text-based formats.
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    # Normalize line endings and excessive whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 3. DECOUPAGE EN CHUNKS
# ============================================================

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Prefer a natural boundary instead of cutting mid-word.
        if end < text_length:
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )

            # Use the boundary only if it is reasonably far ahead.
            if boundary > start + int(chunk_size * 0.6):
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Keep some text from the previous chunk as context.
        start = max(start + 1, end - overlap)

    return chunks


# ============================================================
# 4. CHARGEMENT DES DOCUMENTS
# ============================================================

def load_documents():
    """Recursively load supported files and preserve their metadata."""

    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    files = sorted(
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            f"No supported documents found in {DOCUMENTS_DIR}"
        )

    chunks = []
    ids = []
    metadatas = []

    loaded_files = 0
    skipped_files = 0

    print(f"Found {len(files)} supported files.")

    for file_path in files:
        try:
            text = read_document(file_path)

            if not text:
                print(f"Skipping empty document: {file_path.name}")
                skipped_files += 1
                continue

            relative_path = file_path.relative_to(DOCUMENTS_DIR)
            source = relative_path.as_posix()

            # The first directory identifies the domain.
            domain = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "general"
            )

            document_chunks = split_into_chunks(text)

            if not document_chunks:
                skipped_files += 1
                continue

            # A stable prefix prevents ID collisions between files.
            file_hash = hashlib.sha1(
                source.encode("utf-8")
            ).hexdigest()[:12]

            for index, chunk in enumerate(document_chunks):
                chunks.append(chunk)
                ids.append(f"{file_hash}_{index}")

                metadatas.append({
                    "source": source,
                    "domain": domain,
                    "file_type": file_path.suffix.lower(),
                    "chunk": index,
                })

            loaded_files += 1

        except (OSError, UnicodeError, ValueError) as error:
            print(f"Warning: could not read {file_path}: {error}")
            skipped_files += 1

    if not chunks:
        raise RuntimeError(
            "No text chunks could be extracted from the documents."
        )

    print(f"Successfully loaded files: {loaded_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Total chunks created: {len(chunks)}")

    return chunks, ids, metadatas


# ============================================================
# 5. CREATION DES EMBEDDINGS ET INDEXATION
# ============================================================

def build_vector_database(model, chunks, ids, metadatas):
    """Create embeddings and rebuild the ChromaDB collection."""

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print("Embeddings generated.")

    # Rebuilding this collection removes its previous contents.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # The collection may not exist on the first run.
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add batches to avoid sending all records in one operation.
    for start in range(0, len(chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(chunks))

        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"Indexed {end}/{len(chunks)} chunks.")

    print("\nVector database successfully rebuilt.")

    return collection


# ============================================================
# 6. QUESTION-REPONSE AVEC LE RAG
# ============================================================

def main():
    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunks, ids, metadatas = load_documents()

    collection = build_vector_database(
        model,
        chunks,
        ids,
        metadatas,
    )

    print("\nDocuMind AI is ready!")
    print("Ask a question about Python, FastAPI, React,")
    print("PostgreSQL, Docker or Git.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        # Convert the question into an embedding.
        question_embedding = model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

        # Retrieve the closest chunks from ChromaDB.
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(RETRIEVAL_TOP_K, collection.count()),
        )

        retrieved_texts = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]

        context_chunks = [
            {
                "text": text,
                "source": metadata["source"],
            }
            for text, metadata in zip(
                retrieved_texts,
                retrieved_metadata,
            )
        ]

        print("\nGenerating answer with Qwen3...")

        answer = generate_answer(question, context_chunks)

        print("\n=== DOCUMIND AI ANSWER ===")
        print(answer)

        print("\n=== RETRIEVED SOURCES ===")

        for source in sorted({
            metadata["source"]
            for metadata in retrieved_metadata
        }):
            print(f"- {source}")


if __name__ == "__main__":
    main()
