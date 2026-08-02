from pydantic import BaseModel


class ScreeningResponse(BaseModel):
    """
    Response schema for candidate screening.
    """

    evaluation: str