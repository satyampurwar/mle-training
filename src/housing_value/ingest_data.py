"""
Notes
-----
Use this module for data ingestion of the housing data.
    $ python src/housing_value/ingest_data.py
optional arguments:
    -h, --help            show this help message and exit
    --raw-data RAW        directory to save fetched raw data, default val in setup.cfg: data/raw
    --processed-data PROCESSED
                          directory to save processed data, default val in setup.cfg: data/processed
    --log-level LEVEL     provide logging level, default val in setup.cfg: DEBUG
    --log-data LOG        file to store log data, default val in setup.cfg: logs/main.log if no console display is opted else default is console
                          display
    --no-console-log      do not display log on console
"""
import argparse
import configparser
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "%(asctime)s %(name)s %(filename)s.%(funcName)s(%(lineno)d) %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def fetch_housing_data(housing_url, housing_path):
    """Function to fetch data from link and store data.

    Parameters
    ----------
    housing_url : str
        The url to fetch data.
    housing_path : str
        The directory to store data.

    Returns
    -------
    none : None
        Only fetches and stores data.

    """
    logger.debug("At this function")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    return None


def load_housing_data(housing_path):
    """Function to read raw data from directory.

    Parameters
    ----------
    housing_path : str
        The directory to read raw data.

    Returns
    -------
    df : object
        The pandas dataframe of raw data.

    """
    logger.debug("At this function")
    csv_path = os.path.join(housing_path, "housing.csv")
    df = pd.read_csv(csv_path)
    return df


def data_ingestion(housing_url, housing_path):
    """Function to fetch, store and read raw data.

    Parameters
    ----------
    housing_url : str
        The url to fetch data.
    housing_path : str
        The directory to store and read raw data.

    Returns
    -------
    df : object
        The pandas dataframe of raw data.

    """
    fetch_housing_data(housing_url, housing_path)
    logger.info(f"Stored raw data at : {housing_path}")
    df = load_housing_data(housing_path)
    return df


def data_labeling(df):
    """Function to add a label of income category based on median income to raw data.

    Parameters
    ----------
    df : object
        The pandas dataframe of raw data.

    Returns
    -------
    df : object
        The pandas dataframe of labelled raw data with income categories.

    """
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return df


def save_split_data(df, processed_path=None, split_size=0.2):
    """Function to split data into train & test datasets.

    Parameters
    ----------
    df : object
        The pandas dataframe of labelled raw data with income categories.
    processed_path : str
        The directory to store train and test csv files.
    split_size : float
        The test_size.

    Returns
    -------
    strat_train_set : object
        The pandas dataframe of train data without income categories.
    strat_test_set : object
        The pandas dataframe of test data without income categories.

    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    if processed_path:
        strat_train_set.to_csv(path_or_buf=f"{processed_path}/train.csv", index=False)
        strat_test_set.to_csv(path_or_buf=f"{processed_path}/test.csv", index=False)
        logger.info(f"Stored processed data at : {processed_path}")
        return strat_train_set, strat_test_set
    else:
        return strat_train_set, strat_test_set


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("setup.cfg")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data",
        dest="RAW",
        type=str,
        help="directory to save fetched raw data, default val in setup.cfg: data/raw",
    )
    parser.add_argument(
        "--processed-data",
        dest="PROCESSED",
        type=str,
        help="directory to save processed data, default val in setup.cfg: data/processed",
    )
    parser.add_argument(
        "--log-level",
        dest="LEVEL",
        type=str,
        help="provide logging level, default val in setup.cfg: DEBUG",
    )
    parser.add_argument(
        "--log-data",
        dest="LOG",
        type=str,
        help="file to store log data, default val in setup.cfg: logs/main.log if \
        no console display is opted else default is console display ",
    )
    parser.add_argument(
        "--no-console-log",
        dest="NOCONSOLE",
        action="store_true",
        help="do not display log on console",
    )

    args = parser.parse_args()

    if not args.NOCONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        if not args.LOG:
            file_handler = logging.FileHandler(str(config["Default"]["log_data"]))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    if args.LOG:
        file_handler = logging.FileHandler(args.LOG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.LEVEL:
        logger.setLevel(level=args.LEVEL)
    else:
        logger.setLevel(level=str(config["Default"]["log_level"]))

    HOUSING_URL = str(config["Default"]["raw_data_url"])

    if args.RAW:
        HOUSING_PATH = args.RAW
    else:
        HOUSING_PATH = str(config["Default"]["raw_data"])

    if args.PROCESSED:
        PROCESSED_DATA = args.PROCESSED
    else:
        PROCESSED_DATA = str(config["Default"]["processed_data"])

    housing = data_ingestion(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = data_labeling(df=housing)
    strat_train_set, strat_test_set = save_split_data(
        df=housing, processed_path=PROCESSED_DATA, split_size=0.2
    )
