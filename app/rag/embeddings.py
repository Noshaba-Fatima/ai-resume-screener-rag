"""
Embedding utilities for the RAG pipeline.

This module converts LangChain Document objects into dense vector
embeddings using a SentenceTransformer model.
"""

from typing import List

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings for LangChain Document objects.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        documents: List[Document]
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            NumPy array of shape (num_documents, embedding_dimension).

        Raises:
            ValueError: If the document list is empty.
        """

        if not documents:
            raise ValueError("Document list cannot be empty.")

        texts = [
            document.page_content
            for document in documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:
        """
        Generate an embedding for a search query.

        Args:
            query: Recruiter search query.

        Returns:
            Query embedding as a NumPy array.

        Raises:
            ValueError: If the query is empty.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding