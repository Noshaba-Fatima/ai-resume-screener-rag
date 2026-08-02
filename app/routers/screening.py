from fastapi import APIRouter, Depends

from app.dependencies import get_screening_service
from app.schemas.screening_request import ScreeningRequest
from app.schemas.screening_response import ScreeningResponse
from app.services.screening_service import ScreeningService

import traceback

router = APIRouter()


@router.post(
    "/evaluate",
    response_model=ScreeningResponse,
)
def evaluate_resume(
    request: ScreeningRequest,
    service: ScreeningService = Depends(get_screening_service),
):
    """
    Evaluate a resume against a job description.
    """

    try:
        result = service.screen_candidate(
            job_description=request.job_description,
        )

        return ScreeningResponse(
            evaluation=result,
        )

    except Exception:
        print("\n" + "=" * 80)
        print("SCREENING ERROR")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")
        raise