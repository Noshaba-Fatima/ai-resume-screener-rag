from typing import List

from langchain_core.documents import Document


class PromptBuilder:
    """
    Builds prompts for the LLM.

    Responsibilities:
    - Format retrieved resume chunks.
    - Combine them with the Job Description.
    - Generate a structured prompt for evaluation.

    This class does NOT:
    - Retrieve documents.
    - Call the LLM.
    - Generate embeddings.
    """

    @staticmethod
    def build_prompt(
        job_description: str,
        documents: List[Document],
    ) -> str:
        """
        Create a structured prompt for resume evaluation.

        Args:
            job_description: Job description provided by recruiter.
            documents: Top ranked resume chunks.

        Returns:
            Prompt string to send to the LLM.
        """

        if not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        if not documents:
            raise ValueError("No resume context provided.")

        resume_context = "\n\n".join(
            [
                f"Chunk {i+1}:\n{document.page_content}"
                for i, document in enumerate(documents)
            ]
        )

        prompt = f"""
You are an expert AI Resume Screening Assistant.

Your task is to evaluate a candidate's resume against the provided Job Description.

Job Description:
----------------
{job_description}

Retrieved Resume Context:
-------------------------
{resume_context}

Instructions:
-------------
1. Identify the candidate's relevant technical skills.
2. Identify missing skills required by the job description.
3. Highlight strengths.
4. Highlight weaknesses or skill gaps.
5. Estimate an overall suitability score from 0-100.
6. Explain your reasoning clearly.

Return your response in the following format:

Suitability Score:
Strengths:
Weaknesses:
Missing Skills:
Final Recommendation:
"""

        return prompt.strip()