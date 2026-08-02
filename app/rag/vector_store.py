"""
FAISS-based vector store for the Resume Screener RAG pipeline.

This module is responsible for:
1. Creating a FAISS index
2. Storing LangChain Document objects
3. Adding embeddings to the index
4. Performing similarity search
5. Saving the index and documents to disk
6. Loading the index and documents from disk
"""

from pathlib import Path
from typing import List
import pickle

import faiss
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """
    FAISS-based vector store.

    Maintains the mapping between FAISS vector indices and
    LangChain Document objects.
    """

    def __init__(self, embedding_dimension: int) -> None:
        """
        Initialize an empty FAISS index.

        Args:
            embedding_dimension:
                Dimension of embedding vectors (e.g. 384).
        """

        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.documents: List[Document] = []

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents:
                List of LangChain Document objects.

            embeddings:
                NumPy array of shape
                (num_documents, embedding_dimension).

        Raises:
            ValueError:
                If the number of embeddings does not match
                the number of documents.
        """

        if embeddings.shape[0] != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        embeddings = embeddings.astype("float32")

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Document]:
        """
        Search for the top-k most similar documents.

        Args:
            query_embedding:
                Query embedding of shape (1, embedding_dimension).

            k:
                Number of results to retrieve.

        Returns:
            List of matching Document objects.
        """

        query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx in indices[0]:
            if idx != -1:
                results.append(
                    self.documents[idx]
                )

        return results

    def save(
        self,
        index_path: str,
        documents_path: str
    ) -> None:
        """
        Save the FAISS index and documents to disk.

        Args:
            index_path:
                File path for FAISS index.

            documents_path:
                File path for serialized documents.
        """

        Path(index_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Path(documents_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            documents_path,
            "wb"
        ) as file:
            pickle.dump(
                self.documents,
                file
            )

    def load(
        self,
        index_path: str,
        documents_path: str
    ) -> None:
        """
        Load the FAISS index and documents from disk.

        Args:
            index_path:
                Path to saved FAISS index.

            documents_path:
                Path to saved documents.
        """

        self.index = faiss.read_index(
            index_path
        )

        with open(
            documents_path,
            "rb"
        ) as file:
            self.documents = pickle.load(file)