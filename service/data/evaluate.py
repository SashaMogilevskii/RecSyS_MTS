import click
import json
import joblib as jb
from typing import List

import pandas as pd
from rectools.metrics import Precision, Recall, MeanInvUserFreq, Serendipity, MAP, calc_metrics
from rectools import Columns
from rectools.dataset import Dataset


K_RECOS = 10


@click.command()
@click.argument("input_path", type=click.Path(exists=True), nargs=5)
@click.argument("output_path", type=click.Path())
def evaluate(input_path: List[str], output_path: str):
    """
    Saving score
    :param input_path: path of current model
    :param output_path: path for saving score
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

    df_test = pd.read_csv(input_path[3])
    model = jb.load(input_path[4])

    test_users = df_test[Columns.User].unique()
    catalog = df_train[Columns.Item].unique()

    metrics = {
    "prec@10": Precision(k=10),
    "recall@10": Recall(k=10),
    "novelty": MeanInvUserFreq(k=10),
    "serendipity": Serendipity(k=10),
    "Map@10": MAP(10)
    }

    recos = model.recommend(
        users=test_users,
        dataset=dataset,
        k=K_RECOS,
        filter_viewed=True,
    )

    metric_values = calc_metrics(
        metrics,
        reco=recos,
        interactions=df_test,
        prev_interactions=df_train,
        catalog=catalog,
    )

    with open(output_path, "w") as f:
        json.dump(metric_values, f)


if __name__ == "__main__":
    evaluate()
