from typing import List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.rag.prompts import PromptBuilder


class Evaluator:
    """
    Evaluates a candidate's resume against a job description using an LLM.

    Responsibilities:
    -----------------
    - Build a structured prompt.
    - Send the prompt to the LLM.
    - Return the generated evaluation.

    This class does NOT:
    --------------------
    - Retrieve documents.
    - Generate embeddings.
    - Rerank documents.
    - Perform prompt engineering logic.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the evaluator.

        Args:
            llm:
                Any LangChain compatible chat model
                (Gemini, OpenAI, Ollama, Claude, etc.)
        """
        self.llm = llm

    def evaluate(
        self,
        job_description: str,
        documents: List[Document],
    ) -> str:
        """
        Evaluate a resume against a job description.

        Args:
            job_description:
                Recruiter's job description.

            documents:
                Top-ranked resume chunks retrieved from the vector store.

        Returns:
            LLM generated evaluation.
        """

        # Defensive Programming
        if not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        if not documents:
            raise ValueError("No documents provided for evaluation.")

        # Build Prompt
        prompt = PromptBuilder.build_prompt(
            job_description=job_description,
            documents=documents,
        )

        try:
            # Send prompt to the LLM
            response = self.llm.invoke(
                [
                    HumanMessage(content=prompt)
                ]
            )

            # Return only the generated response
            return response.content

        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate resume using the LLM: {str(e)}"
            ) from e