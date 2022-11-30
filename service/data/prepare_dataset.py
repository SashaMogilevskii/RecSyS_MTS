import click
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
from rectools import Columns
from rectools.model_selection import TimeRangeSplitter
from  rectools.dataset.interactions import Interactions


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), nargs=2)
@click.argument("unit", type=click.STRING)
@click.argument("n_units", type=click.INT)
def prepare_dataset(input_path: str, output_path: List[str], unit='W', n_units=1):
    """
    Preprocessing and train/test split
    :param input_path: path of data directory
    :param output_path: path for saving train and test data
    :param unit: step time for test
    :param n_units: count of steps
    """
    input_path = Path(input_path).resolve()
    output_path_train = Path(output_path[0]).resolve()
    output_path_test = Path(output_path[1]).resolve()

    interactions = pd.read_csv(input_path / "kion_train" / "interactions.csv")

    interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                                'total_dur': Columns.Weight},
                        inplace=True)

    interactions['datetime'] = pd.to_datetime(interactions['datetime'])

    last_date = interactions[Columns.Datetime].max().normalize()
    start_date = last_date - pd.Timedelta(n_units, unit=unit)

    interactions_ = Interactions(interactions)

    periods = 2
    freq = f"{n_units}{unit}"
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq, tz=last_date.tz)

    cv = TimeRangeSplitter(
        date_range=date_range,
        filter_already_seen=True,
        filter_cold_items=True,
        filter_cold_users=True,
    )

    train_ids, test_ids, fold_info = cv.split(interactions_, collect_fold_stats=True).__next__()

    df_train = interactions.loc[train_ids]
    df_test = interactions.loc[test_ids][Columns.UserItem]

    df_train.to_csv(output_path_train, index=False)
    df_test.to_csv(output_path_test, index=False)


if __name__ == "__main__":
    prepare_dataset()
