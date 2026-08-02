from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.embeddings import EmbeddingGenerator
from app.rag.vector_store import VectorStore
from app.schemas.resume import UploadResponse
from app.services.resume_service import ResumeService

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

embedding_model = EmbeddingGenerator()

vector_store = VectorStore(
    embedding_dimension=384,
)

resume_service = ResumeService(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_resume(
    file: UploadFile = File(...),
):
    """
    Upload and index a resume.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:

        message = resume_service.upload_resume(
            str(file_path)
        )

        return UploadResponse(
            message=message,
            filename=file.filename,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )