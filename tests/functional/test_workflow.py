import configparser
import math
import pickle

import numpy as np
import pytest
import sklearn

from housing_value import ingest_data, score, train

config = configparser.ConfigParser()
config.read("setup.cfg")


def random_data_constructor(housing_url, housing_path):
    df = ingest_data.data_ingestion(housing_url, housing_path)
    df = df.sample(frac=0.1, random_state=101, ignore_index=True)
    df = ingest_data.data_labeling(df)
    return df


def test_workflow_data_labeling():
    df = random_data_constructor(
        housing_url=str(config["Default"]["raw_data_url"]),
        housing_path=str(config["Default"]["raw_data"]),
    )
    # Check added dimension to ingested data
    assert "income_cat" in df.columns


def split_data_constructor():
    df = random_data_constructor(
        housing_url=str(config["Default"]["raw_data_url"]),
        housing_path=str(config["Default"]["raw_data"]),
    )
    train_sample, test_sample = ingest_data.save_split_data(df=df, split_size=0.1)
    return train_sample, test_sample


def test_workflow_split_data():
    train_sample, test_sample = split_data_constructor()
    split = len(test_sample) / (len(train_sample) + len(test_sample))
    # Check train test split size of sample data
    assert math.isclose(split, 0.1, abs_tol=0.0003)


def training_data_constructor():
    train_sample, test_sample = split_data_constructor()
    features, labels = train.load_training_data(df=train_sample)
    return features, labels


def test_workflow_train_feature_data():
    features, labels = training_data_constructor()
    # Check dimensions of feature data
    assert len(features.columns) == 9


def train_feature_engineering_constructor():
    features, labels = training_data_constructor()
    imputer, feature_prepared = train.original_feature_engineering(
        df=features,
        pickle_path=str(config["Default"]["pickle_data"]),
        imputer_file="test_imputer.pkl",
    )
    return feature_prepared, labels


@pytest.fixture
def pickled():
    return str(config["Default"]["pickle_data"])


def test_workflow_imputer_file(pickled):
    train_feature_engineering_constructor()
    file = open(f"{pickled}/test_imputer.pkl", "rb")
    # Check whether correct imputer loaded
    assert isinstance(pickle.load(file), sklearn.impute.SimpleImputer)


def model_training_constructor():
    feature_prepared, labels = train_feature_engineering_constructor()
    model = train.rf_regressor_model_training(
        df_prepared=feature_prepared,
        labels=labels,
        pickle_path=str(config["Default"]["pickle_data"]),
        model_file="test_model.pkl",
    )
    return model


def test_workflow_model_file(pickled):
    model_training_constructor()
    file = open(f"{pickled}/test_model.pkl", "rb")
    # Check whether correct imputer loaded
    assert isinstance(pickle.load(file), sklearn.ensemble.RandomForestRegressor)


def testing_data_constructor():
    train_sample, test_sample = split_data_constructor()
    features, labels = score.load_scoring_data(df=test_sample)
    return features, labels


def test_workflow_test_feature_data():
    features, labels = testing_data_constructor()
    # Check dimensions of feature data
    assert len(features.columns) == 9


def test_feature_engineering_constructor():
    features, labels = testing_data_constructor()
    feature_prepared = score.replicate_feature_engineering(
        df=features,
        pickle_path=str(config["Default"]["pickle_data"]),
        imputer_file="test_imputer.pkl",
    )
    return feature_prepared, labels


def model_scoring_constructor():
    feature_prepared, labels = test_feature_engineering_constructor()
    output, rmse = score.scoring_test_data(
        df_prepared=feature_prepared,
        actuals=labels,
        pickle_path=str(config["Default"]["pickle_data"]),
        model_file="test_model.pkl",
        output_path=str(config["Default"]["output_data"]),
        output_file="test_output.csv",
    )
    return labels, rmse


def test_workflow_model_score():
    labels, rmse = model_scoring_constructor()
    # Check whether scoring is appropriate
    assert np.round(rmse / np.mean(labels), 2) < 0.3
