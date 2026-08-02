from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.rag.embeddings import EmbeddingGenerator
from app.rag.vector_store import VectorStore


class Retriever:
    """
    Performs semantic retrieval over the FAISS vector store.

    Responsibilities:
    - Convert the user query into an embedding.
    - Retrieve the most relevant document chunks.
    - Optionally filter retrieved documents using metadata.

    This class does NOT:
    - Re-rank retrieved documents.
    - Call the LLM.
    - Generate prompts.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingGenerator,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: Instance of VectorStore used for semantic search.
            embedding_model: Embedding model used to generate query embeddings.
        """

        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Retrieve the top-k most relevant document chunks.

        Args:
            query: User query or job description.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieved LangChain Document objects.

        Raises:
            ValueError: If the query is empty or top_k is invalid.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_model.embed_query(query)

        retrieved_documents = self.vector_store.search(
            query_embedding=query_embedding,
            k=top_k,
        )

        return retrieved_documents

    def retrieve_with_metadata(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Retrieve documents and filter them using metadata.

        Args:
            query: User query.
            metadata_filters: Dictionary containing metadata filters.
            top_k: Number of documents to retrieve.

        Returns:
            Filtered list of Document objects.
        """

        documents = self.retrieve(
            query=query,
            top_k=top_k,
        )

        if not metadata_filters:
            return documents

        filtered_documents: List[Document] = []

        for document in documents:

            is_match = True

            for key, value in metadata_filters.items():

                if document.metadata.get(key) != value:
                    is_match = False
                    break

            if is_match:
                filtered_documents.append(document)

        return filtered_documents