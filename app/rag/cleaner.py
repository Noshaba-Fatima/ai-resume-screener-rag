"""
Text cleaning utilities for the RAG pipeline.

This module prepares extracted document text for chunking and embedding
by normalizing whitespace while preserving meaningful content.
"""

import re


def clean_text(raw_text: str) -> str:
    """
    Clean extracted document text.

    The function:
    - Normalizes line endings.
    - Replaces tabs with spaces.
    - Removes extra spaces.
    - Removes excessive blank lines.
    - Trims leading and trailing whitespace.

    Args:
        raw_text: Raw text extracted from a document.

    Returns:
        Cleaned text ready for chunking.

    Raises:
        ValueError: If the input is not a string.
    """

    if not isinstance(raw_text, str):
        raise ValueError("Input must be a string.")

    if not raw_text.strip():
        return ""

    # Normalize different newline formats
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with a single space
    text = text.replace("\t", " ")

    # Replace multiple spaces with one space
    text = re.sub(r" {2,}", " ", text)

    # Keep at most one blank line between sections
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces before newlines
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text