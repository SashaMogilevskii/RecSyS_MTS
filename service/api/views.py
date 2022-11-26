import os
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import joblib as jb

from service.api.exceptions import (
    ModelNotFoundError,
    NotAuthorizedError,
    UserNotFoundError,
)
from service.log import app_logger
from ..modelss.allmodels import  (
    Random2Recommend,
    StupidTop,
    NoStupidTop
    )


load_dotenv()

API_KEY = os.getenv('PROJECT_API_KEY')
assert API_KEY is not None, 'API_KEY is empty!'

api_key_query = APIKeyQuery(name='API_KEY', auto_error=False)
api_key_header = APIKeyHeader(name='API_KEY', auto_error=False)
token_bearer = HTTPBearer(auto_error=False)

list_models = ['random_model',
               'random2_model',
               'stupid_top',
               'no_stupid_top',
               'base_userknn',]

random2_model = Random2Recommend()
stupid_top = StupidTop()
no_stupid_top = NoStupidTop()
async def get_api_key(
        api_key_from_query: str = Security(api_key_query),
        api_key_from_header: str = Security(api_key_header),
        token: HTTPAuthorizationCredentials = Security(token_bearer)
) -> str:
    if api_key_from_query == API_KEY:
        return api_key_from_query
    if api_key_from_header == API_KEY:
        return api_key_from_header
    if token is not None and token.credentials == API_KEY:
        return token.credentials
    raise NotAuthorizedError()


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
    responses={404: {"description": "User not found or model not use"},
               401: {"description": "Token is wrong."}}
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: APIKey = Depends(get_api_key)
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

    # Проверяем есть ли такая модель в нашем списке
    if model_name in list_models:
        k_recs = request.app.state.k_recs
        if model_name == 'random_model': #Модель из 1ого дз (range(10))
            reco = list(range(k_recs))
        elif model_name == 'random2_model': #Тестим подгрузку моделей из класса
            reco = random2_model.predict(user_id=user_id, k=k_recs)
        elif model_name == 'stupid_top': # Простой топ из hw3.1
            reco = stupid_top.predict(user_id=user_id, k=k_recs)
        elif model_name == 'no_stupid_top':  # NoStupid топ из hw3.1
            reco = no_stupid_top.predict(user_id=user_id, k=k_recs)
        elif model_name == 'base_userknn':  # UserKNN from lecture
            model = jb.load('models/model.clf')
            predicted = model.predict(test=user_id, N_recs=k_recs)
            reco = list(predicted['item_id'].values)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")


    return RecoResponse(user_id=user_id, items=reco)

    ## Добавить 2 модели
    # Построить классы этих моделей посчитать их веса и все значения.


def add_views(app: FastAPI) -> None:
    app.include_router(router)
