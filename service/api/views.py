from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError, ModelNotFoundError
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    """
    Test, which checks the viability of the function
    """
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    """
    Get recommendations for  a user
    :param request:
    :param model_name: srt - type of choose model
    :param user_id: int - number user for recommendations
    :return: RecoResponse int, List[int]
    """


    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name == 'random_model':
        k_recs = request.app.state.k_recs
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    reco = list(range(k_recs))
    return RecoResponse(user_id=user_id, items=reco)

def add_views(app: FastAPI) -> None:
    app.include_router(router)
