import pandas as pd
import numpy as np
import joblib as jb

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


class StupidTop(BaseModel):
    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user top with filter (notebook HW3_1)
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """
        reco = [9728, 13865, 15297, 10440, 3734,
                4151, 142, 8636, 7571, 4495]
        return reco


class NoStupidTop(BaseModel):

    def __init__(self):
        super(NoStupidTop).__init__()
        self.user_iteractions = pd.read_csv(
            'service/data/users_interactions.csv',
            converters={"item_id": lambda x: x.strip("[]").split(", ")}
                                            )
        self.list_top_items = np.array([ 15297, 10440,  9728, 13865,  3734,  4151,  2657,   142,  8636,
        7571,  7102,  4495,  4457,  7829, 12995,  1844,  6809,  7417,
        4740,  9996,  3784,  3935, 16166,  7793, 14703, 14431, 12192,
       11237, 11778,  3182, 14317,  5434,  4436, 12173, 16228,  7626,
         512, 10464, 14741, 14901,  5693,  5411, 16270, 12324,  1449,
        1132,  6402, 15464,  7582, 13018,  6455,  1916,  6162, 10761,
        1785, 12356,  5658,  1819, 11749, 10942, 10772, 14488,  9194,
         657,  5732,  5424, 12981, 13915,  1287, 14470, 12743, 12841,
        6626, 10077,  2954, 13243,  8447,  4880,  8986,  2237, 15266,
        3509, 11310, 13159, 12965,  2220,  4718,  5803, 11756, 13262,
        6443,  4471,  2722, 10878,  7310, 11640, 12623,  4696,  9900,
       10605, 11985,  4260,  9169, 12537,  7210,  4946,  1290,   366,
        7107,  1418, 11118, 14899, 14461, 11754, 12770, 10240, 12228,
       16087,  4702, 11348,  9342, 15362,  6382,  9506,   288,  8437,
        6774,  8821, 10770,  5754, 10323])

        self.reco = [9728, 13865, 15297, 10440, 3734,
                4151, 142, 8636, 7571, 4495]

    def predict(self, user_id: int, k: int) -> List[int]:
        """
        Return recommendation for user top with filter (notebook HW3_1)
        This model delete items, which user used.
        :param user_id: unique user_id
        :param k: amount recommendation
        :return: list recommendation
        """

        #Проверяем есть ли такой юзер в БД:
        check_user = np.any(self.user_iteractions.user_id == user_id)

        if check_user:
            #Список фильмов, к-ые просмотрел юзер
            list_watched = self.user_iteractions[self.user_iteractions.user_id == \
                                                 user_id].to_numpy()[0][1]
            #Находим фильмы, которые смотрел уже юзер и есть в рекомендаторе
            mask = np.array([item in list_watched for item in self.list_top_items])
            #Удаляем их из рекомендации и берем первые К
            reco = self.list_top_items[~mask].tolist()[:k]

            #Если у юзера меньше к рекомендаций заполняем случайными числами
            #Потом можно будет улучшить
            if len(reco) < k:
                reco = reco + range(k - len(reco))
        #Юзера нету в бд. Выдаем просто тop-К рекомендаций
        else:
            reco = self.reco

        return reco


class Knn_20(BaseModel):
    def __init__(self):
        super(BaseModel).__init__()

        self.model = jb.load('models/model_20.clf')
        self.reco = [9728, 13865, 15297, 10440, 3734,
                     4151, 142, 8636, 7571, 4495]

    def predict(self, user_id: int, k: int) -> List[int]:
        predicted = self.model.predict(test=user_id, N_recs=k)
        if len(predicted) == 0:
            reco = self.no_stupid_model.predict(user_id=user_id, k=k)
        else:
            reco = list(predicted['item_id'].values)
            if len(reco) < k:
                reco += self.no_stupid_modelp.predict(user_id=user_id,
                                                      k=k - len(reco))

        return reco

class Knn_20_All(BaseModel):
    def __init__(self):
        super(BaseModel).__init__()

        self.model = jb.load('models/model_20_all.clf')
        self.no_stupid_model = NoStupidTop()
        self.reco_unique = []

    def predict(self, user_id: int, k: int) -> List[int]:
        predicted = self.model.predict(test=user_id, N_recs=k)
        if len(predicted) == 0:
            self.reco_unique = self.no_stupid_model.predict(user_id=user_id, k=k)
        else:
            reco = list(predicted['item_id'].values)
            if len(reco) < k:
                reco += self.no_stupid_model.predict(user_id=user_id,
                                              k=k + 15)
            for el in reco:
                if el not in self.reco_unique:
                    self.reco_unique.append(el)

                if len(self.reco_unique) == 10:
                    break



        return self.reco_unique
