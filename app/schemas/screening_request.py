from pydantic import BaseModel


class ScreeningRequest(BaseModel):
    """
    Request schema for candidate screening.
    """

    job_description: str