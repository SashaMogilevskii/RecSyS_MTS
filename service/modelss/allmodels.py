import json
from abc import ABC
from typing import List

import joblib as jb
import numpy as np
import pandas as pd


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


class StupidTop(BaseModel):
    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user top with filter (notebook HW3_1)
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """
        with open('./json_files/reco.json', 'r') as f:
            json_dict = json.load(f)
            reco = json_dict['reco_top10']
        return reco


class NoStupidTop(BaseModel):
    def __init__(self):
        self.user_iteractions = pd.read_csv(
            "service/data/users_interactions.csv",
            converters={"item_id": lambda x: x.strip("[]").split(", ")},
        )

        with open('./json_files/reco.json', 'r') as f:
            json_dict = json.load(f)
            list_top_items = json_dict['reco_140']
            list_top_10 = json_dict['reco_top10']
        self.list_top_items = np.array(list_top_items)
        self.reco = list_top_10

    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user top with filter (notebook HW3_1)
        This model delete items, which user used.
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """

        # Проверяем есть ли такой юзер в БД:
        check_user = np.any(self.user_iteractions.user_id == user_id)

        if check_user:
            # Список фильмов, к-ые просмотрел юзер
            list_watched = self.user_iteractions[
                self.user_iteractions.user_id == user_id
            ].to_numpy()[0][1]
            # Находим фильмы, которые смотрел уже юзер и есть в рекомендаторе
            mask = np.array(
                [item in list_watched for item in self.list_top_items]
            )
            # Удаляем их из рекомендации и берем первые К
            reco = self.list_top_items[~mask].tolist()[:k]

            # Если у юзера меньше к рекомендаций заполняем случайными числами
            # Потом можно будет улучшить
            if len(reco) < k:
                reco = reco + range(k - len(reco))
        # Юзера нету в бд. Выдаем просто тop-К рекомендаций
        else:
            reco = self.reco

        return reco


class Knn_20(BaseModel):
    def __init__(self):
        with open('./json_files/reco.json', 'r') as f:
            json_dict = json.load(f)
            list_top_10 = json_dict['reco_top10']

        self.reco = list_top_10
        self.model = jb.load("models/model_20.clf")
        self.no_stupid_model = NoStupidTop()

    def predict(self, user_id: int, k: int) -> List[int]:
        predicted = self.model.predict(test=user_id, N_recs=k)
        if len(predicted) == 0:
            reco = self.no_stupid_model.predict(user_id=user_id, k=k)
        else:
            reco = list(predicted["item_id"].values)
            if len(reco) < k:
                reco += self.no_stupid_model.predict(user_id=user_id,
                                                     k=k - len(reco))

        return reco


class model_LightFM(BaseModel):
    def __init__(self):
        with open('./json_files/reco.json', 'r') as f:
            json_dict = json.load(f)
            list_top_items = json_dict['reco_140']
            list_top_10 = json_dict['reco_top10']
        self.list_top_items = np.array(list_top_items)
        self.reco = list_top_10
        self.predict_for_user = pd.read_csv(
            "service/data/reco_LightFM.csv",
            converters={"item_id": lambda x: x.strip("[]").split(", ")},
        )

    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user top with filter (notebook HW3_1)
        This model delete items, which user used.
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """
        check_user = np.any(self.predict_for_user.user_id == user_id)

        # Check users in data from LightFm predicted
        if check_user:
            # If True -> return list
            reco = list(self.predict_for_user[
                self.predict_for_user.user_id == user_id
                                ]['item_id'].values[0])

            print(reco, type(reco), 'user naiden')
        else:
            reco = self.reco
        return reco
