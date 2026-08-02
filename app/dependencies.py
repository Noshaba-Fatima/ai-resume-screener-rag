from pathlib import Path

from app.llm import llm
from app.rag.embeddings import EmbeddingGenerator
from app.rag.evaluator import Evaluator
from app.rag.pipeline import ResumeScreeningPipeline
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.screening_service import ScreeningService


VECTOR_STORE_PATH = "vector_store/faiss.index"
METADATA_PATH = "vector_store/documents.pkl"


def get_screening_service() -> ScreeningService:
    """
    Create and return a fully configured ScreeningService.
    """

    embedding_model = EmbeddingGenerator()

    vector_store = VectorStore(embedding_dimension=384)

    if Path(VECTOR_STORE_PATH).exists() and Path(METADATA_PATH).exists():
        vector_store.load(
            VECTOR_STORE_PATH,
            METADATA_PATH,
        )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    reranker = Reranker()

    evaluator = Evaluator(
        llm=llm,
    )

    pipeline = ResumeScreeningPipeline(
        retriever=retriever,
        reranker=reranker,
        evaluator=evaluator,
    )

    return ScreeningService(
        pipeline=pipeline,
    )