from typing import Any, Dict, Optional

from app.rag.pipeline import ResumeScreeningPipeline


class ScreeningService:
    """
    Service responsible for screening candidates.

    This service delegates the screening process
    to the ResumeScreeningPipeline.
    """

    def __init__(
        self,
        pipeline: ResumeScreeningPipeline,
    ) -> None:

        self.pipeline = pipeline

    def screen_candidate(
        self,
        job_description: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Screen a candidate against a job description.

        Args:
            job_description:
                Recruiter's job description.

            metadata_filter:
                Optional metadata filters.

        Returns:
            AI-generated evaluation.
        """

        return self.pipeline.run(
            job_description=job_description,
            metadata_filter=metadata_filter,
        )