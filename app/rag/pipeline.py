from typing import Any, Dict, Optional

from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.evaluator import Evaluator


class ResumeScreeningPipeline:
    """
    End-to-end resume screening pipeline.

    Responsibilities:
    -----------------
    - Retrieve relevant resume chunks.
    - Rerank retrieved chunks.
    - Evaluate the candidate using an LLM.

    This class coordinates the workflow only.
    It does NOT implement retrieval,
    reranking, prompt building or evaluation logic.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        evaluator: Evaluator,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            retriever:
                Retrieves relevant resume chunks.

            reranker:
                Improves retrieval quality.

            evaluator:
                Uses the LLM to evaluate candidates.
        """

        self.retriever = retriever
        self.reranker = reranker
        self.evaluator = evaluator

    def run(
        self,
        job_description: str,
        top_k: int = 10,
        final_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute the complete screening pipeline.

        Args:
            job_description:
                Recruiter's job description.

            top_k:
                Number of documents retrieved from FAISS.

            final_k:
                Number of documents kept after reranking.

            metadata_filter:
                Optional metadata filters.

        Returns:
            LLM generated evaluation.
        """

        if not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        try:

            # Step 1: Retrieve documents
            if metadata_filter:
                retrieved_documents = self.retriever.retrieve_with_metadata(
                    query=job_description,
                    metadata_filters=metadata_filter,
                    top_k=top_k,
                )
            else:
                retrieved_documents = self.retriever.retrieve(
                    query=job_description,
                    top_k=top_k,
                )

            # Step 2: Rerank documents
            reranked_documents = self.reranker.rerank(
                query=job_description,
                documents=retrieved_documents,
                top_k=final_k,
            )

            # Step 3: Evaluate candidate
            evaluation = self.evaluator.evaluate(
                job_description=job_description,
                documents=reranked_documents,
            )

            return evaluation

        except Exception as e:
            raise RuntimeError(
                f"Resume screening pipeline failed: {str(e)}"
            ) from e