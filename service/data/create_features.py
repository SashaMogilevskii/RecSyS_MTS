import click
from pathlib import Path
from typing import List
import pandas as pd
from rectools import Columns


@click.command()
@click.argument("input_path", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path(), nargs=2)
def create_features(input_path: List[str], output_path: List[str]):
    """
    Create addidional features for ALS model
    :param input_path: train and kion datasets
    :param output_path: users and items features
    :return:
    """
    input_path_train = Path(input_path[0]).resolve()
    input_path_data = Path(input_path[1]).resolve()
    output_path_users = Path(output_path[0]).resolve()
    output_path_items = Path(output_path[1]).resolve()

    train = pd.read_csv(input_path_train)
    users = pd.read_csv(input_path_data / "kion_train" / 'users.csv')
    items = pd.read_csv(input_path_data / "kion_train" / 'items.csv')

    users.fillna('Unknown', inplace=True)
    users = users.loc[users[Columns.User].isin(train[Columns.User])].copy()

    user_features_frames = []
    for feature in ["sex", "age", "income"]:
        feature_frame = users.reindex(columns=[Columns.User, feature])
        feature_frame.columns = ["id", "value"]
        feature_frame["feature"] = feature
        user_features_frames.append(feature_frame)
    user_features = pd.concat(user_features_frames)

    items = items.loc[items[Columns.Item].isin(train[Columns.Item])].copy()

    items["genre"] = items["genres"].str.lower().str.replace(", ", ",",
                                                             regex=False).str.split(
        ",")
    genre_feature = items[["item_id", "genre"]].explode("genre")
    genre_feature.columns = ["id", "value"]
    genre_feature["feature"] = "genre"

    content_feature = items.reindex(columns=[Columns.Item, "content_type"])
    content_feature.columns = ["id", "value"]
    content_feature["feature"] = "content_type"

    item_features = pd.concat((genre_feature, content_feature))

    user_features.to_csv(output_path_users, index=False)
    item_features.to_csv(output_path_items, index=False)


if __name__ == "__main__":
    create_features()
