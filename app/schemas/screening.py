from typing import Any, Dict, Optional

from pydantic import BaseModel


class ScreeningRequest(BaseModel):
    job_description: str
    top_k: int = 10
    final_k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None


class ScreeningResponse(BaseModel):
    evaluation: str