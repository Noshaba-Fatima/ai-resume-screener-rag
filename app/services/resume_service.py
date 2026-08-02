from pathlib import Path

from app.rag.loader import load_pdf
from app.rag.cleaner import clean_text
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingGenerator
from app.rag.vector_store import VectorStore


class ResumeService:
    """
    Handles resume ingestion.

    Responsibilities:
    - Save uploaded resume
    - Extract text
    - Clean text
    - Chunk text
    - Generate embeddings
    - Store embeddings in FAISS
    """

    def __init__(
        self,
        embedding_model: EmbeddingGenerator,
        vector_store: VectorStore,
    ) -> None:

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def upload_resume(
        self,
        file_path: str,
    ) -> str:
        """
        Process and index a resume.

        Each new upload replaces the previously indexed resume.
        """

        # Load PDF
        raw_text = load_pdf(file_path)

        # Clean text
        cleaned_text = clean_text(raw_text)

        # Chunk text
        documents = chunk_text(
            text=cleaned_text,
            source=Path(file_path).name,
        )

        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(
            documents
        )

        # Reset the vector store so only the latest resume is indexed
        self.vector_store = VectorStore(
            embedding_dimension=384,
        )

        # Store the new resume
        self.vector_store.add_documents(
            documents,
            embeddings,
        )

        # Persist to disk
        self.vector_store.save(
            index_path="vector_store/faiss.index",
            documents_path="vector_store/documents.pkl",
        )

        return "Resume indexed successfully."