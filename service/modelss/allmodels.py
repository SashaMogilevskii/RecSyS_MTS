from abc import ABC
from typing import List


class BaseModel(ABC):
    """
    Базовый класс для моделей, которые будем использовать в дальнейшем
    """
    def __init__(self):
        pass

    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """
# Модель, которая возвращает 10 красивых значений
# Сделали, чтобы проверить, что все работает
class Random2Recommend(BaseModel):
    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """
        reco = [12, 13, 123, 17, 21, 27, 31, 39, 51, 73]
        return reco
