from fastapi import FastAPI

from app.routers import resumes, screening

app = FastAPI(
    title="AI Resume Screener",
    description="AI-powered Resume Screening using RAG and LLMs.",
    version="1.0.0",
)

app.include_router(
    resumes.router,
    prefix="/resume",
    tags=["Resume"],
)

# app.include_router(
#     users.router,
#     prefix="/users",
#     tags=["Users"],
# )

app.include_router(
    screening.router,
    prefix="/screening",
    tags=["Screening"],
)

@app.get("/")
def health_check():
    return {
        "message": "AI Resume Screener API is running."
    }