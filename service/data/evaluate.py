# import sys
# import click
# import json
# import joblib as jb
# from typing import List
#
# import pandas as pd
# from rectools.metrics import (
#     Precision,
#     Recall,
#     MeanInvUserFreq,
#     Serendipity,
#     calc_metrics,
# )
# from rectools import Columns
#
# sys.path.insert(1, "service/modelss/")
# from userknn import UserKnn
#
#
# @click.command()
# @click.argument("input_path", type=click.Path(exists=True), nargs=3)
# @click.argument("output_path", type=click.Path())
# def evaluate(input_path: List[str], output_path: str):
#     """
#     Saving score
#     :param input_path: path of current model
#     :param output_path: path for saving score
#     """
#     df_train = pd.read_csv(input_path[0])
#     df_test = pd.read_csv(input_path[1])
#     model = jb.load(input_path[2])
#
#     catalog = df_train[Columns.Item].unique()
#
#     metrics = {
#         "prec@10": Precision(k=10),
#         "recall@10": Recall(k=10),
#         "novelty": MeanInvUserFreq(k=10),
#         "serendipity": Serendipity(k=10),
#     }
#
#     recos = model.predict(df_test)
#
#     metric_values = calc_metrics(
#         metrics,
#         reco=recos,
#         interactions=df_test,
#         prev_interactions=df_train,
#         catalog=catalog,
#     )
#
#     with open(output_path, "w") as f:
#         json.dump(metric_values, f)
#
#
# if __name__ == "__main__":
#     evaluate()
