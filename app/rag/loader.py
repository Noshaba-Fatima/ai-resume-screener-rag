from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and return all extractable text as a single string.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A string containing all extracted text from the PDF.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no readable text is found in the PDF.
        RuntimeError: If the PDF cannot be processed.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        reader = PdfReader(path)

        pages = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text and page_text.strip():
                pages.append(page_text)

        if not pages:
            raise ValueError("No readable text found in the PDF.")

        return "\n".join(pages)

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}") from e