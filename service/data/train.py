import os
import click
import joblib as jb
from typing import List

import pandas as pd
from rectools.models import ImplicitALSWrapperModel, LightFMWrapperModel
from implicit.als import AlternatingLeastSquares
from lightfm import LightFM

from rectools.dataset import Dataset


os.environ["OPENBLAS_NUM_THREADS"] = "1"

RANDOM_STATE = 42
NUM_THREADS = 16
N_FACTORS = 32
N_EPOCHS = 1
USER_ALPHA = 0
ITEM_ALPHA = 0
LEARNING_RATE = 0.05


@click.command()
@click.argument("input_path", type=click.Path(exists=True), nargs=3)
@click.argument("output_path", type=click.Path())
def train(input_path: List[str], output_path: str):
    """
    Find the best hyperparameters, train model and tracking experiment by mlflow
    :param input_path: path of train and test datasets
    :param output_path: path for saving model
    :param n_users: number of users for model
    """
    df_train = pd.read_csv(input_path[0])
    user_features = pd.read_csv(input_path[1])
    item_features = pd.read_csv(input_path[2])

    dataset = Dataset.construct(
        interactions_df=df_train,
        user_features_df=user_features,
        cat_user_features=["sex", "age", "income"],
        item_features_df=item_features,
        cat_item_features=["genre", "content_type"],
    )

    # model = ImplicitALSWrapperModel(
    #                 model=AlternatingLeastSquares(
    #                     factors=N_FACTORS,
    #                     random_state=RANDOM_STATE,
    #                     num_threads=NUM_THREADS,
    #                 ),
    #                 fit_features_together=True,
    #             )

    model = LightFMWrapperModel(
            LightFM(
                no_components=N_FACTORS,
                loss='warp',
                random_state=RANDOM_STATE,
                learning_rate=LEARNING_RATE,
                user_alpha=USER_ALPHA,
                item_alpha=ITEM_ALPHA,
            ),
            epochs=N_EPOCHS,
            num_threads=NUM_THREADS,
        )

    model.fit(dataset)

    jb.dump(model, output_path)


if __name__ == "__main__":
    train()
