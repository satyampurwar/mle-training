import configparser
from os import path

import pandas as pd

from housing_value import ingest_data

config = configparser.ConfigParser()
config.read("setup.cfg")

HOUSING_URL = str(config["Default"]["raw_data_url"])
HOUSING_PATH = str(config["Default"]["raw_data"])


def test_fetch_housing_data():
    ingest_data.fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    assert path.exists(f"{HOUSING_PATH}/housing.csv")


def test_load_housing_data():
    df = ingest_data.load_housing_data(housing_path=HOUSING_PATH)
    assert isinstance(df, pd.DataFrame)


def test_data_ingestion():
    df = ingest_data.data_ingestion(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    assert isinstance(df, pd.DataFrame)
