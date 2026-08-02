"""
Chunking utilities for the RAG pipeline.

This module converts cleaned text into LangChain Document objects
using RecursiveCharacterTextSplitter.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Split cleaned text into overlapping chunks.

    Args:
        text: Cleaned document text.
        source: Original document name or path.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError:
            - If text is empty.
            - If chunk_overlap >= chunk_size.
    """

    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    documents = splitter.create_documents(
        [text],
        metadatas=[
            {
                "source": source,
            }
        ],
    )

    for index, document in enumerate(documents):
        document.metadata["chunk_id"] = index

    return documents