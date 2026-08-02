from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-ranks retrieved documents using a CrossEncoder model.

    Unlike embedding-based retrieval, a CrossEncoder jointly
    evaluates the query and each document, producing a more
    accurate relevance score.

    Responsibilities:
    - Score retrieved documents.
    - Sort documents by relevance.
    - Return the top-ranked documents.

    This class does NOT:
    - Retrieve documents.
    - Generate embeddings.
    - Call the LLM.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """
        Initialize the CrossEncoder reranker.

        Args:
            model_name: HuggingFace CrossEncoder model.
        """

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[Document]:
        """
        Re-rank retrieved documents.

        Args:
            query: User query or Job Description.
            documents: Retrieved LangChain documents.
            top_k: Number of final documents to return.

        Returns:
            Ranked list of Document objects.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        sentence_pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(sentence_pairs)

        ranked_results = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        ranked_documents = [
            document
            for _, document in ranked_results
        ]

        if top_k is not None:

            if top_k <= 0:
                raise ValueError("top_k must be greater than zero.")

            ranked_documents = ranked_documents[:top_k]

        return ranked_documents