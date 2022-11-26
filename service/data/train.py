import os
import sys
import click
import joblib as jb
from typing import List

import pandas as pd
from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender

sys.path.insert(1, 'service/modelss/')
from userknn import UserKnn


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def train(input_path: str, output_path: str):
    """
    Find the best hyperparameters, train model and tracking experiment by mlflow
    :param input_path: path of train and test datasets
    :param output_path: path for saving model
    """
    df_train = pd.read_csv(input_path)
  
    userknn_model = UserKnn(model=TFIDFRecommender(), N_users=50)
    userknn_model.fit(df_train)
    
    jb.dump(userknn_model, output_path)
    
if __name__ == "__main__":
    train()