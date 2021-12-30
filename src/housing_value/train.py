"""
Notes
-----
Use this module for model training on the housing data.
    $ python src/housing_value/train.py
optional arguments:
    -h, --help            show this help message and exit
    --processed-data PROCESSED
                          directory to read training data, default val in setup.cfg: data/processed
    --pickle-data PICKLE  directory to save pickle files, default val in setup.cfg: artifacts
    --log-level LEVEL     provide logging level, default val in setup.cfg: DEBUG
    --log-data LOG        file to store log data, default val in setup.cfg: logs/main.log if no console display is opted else default is console
                          display
    --no-console-log      do not display log on console
"""
import argparse
import configparser
import logging
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "%(asctime)s %(name)s %(filename)s.%(funcName)s(%(lineno)d) %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def load_training_data(df=None, processed_path=None):
    """Function to read training data and return features and labels.

    Parameters
    ----------
    df : object
        The pandas dataframe of training data to bypass reading data from directory.
    processed_path : str
        The directory to read training data.

    Returns
    -------
    features : object
        The pandas dataframe of features.
    labels : object
        The pandas series of labels.

    """
    logger.debug("Reading training data")
    if processed_path:
        strat_train_set = pd.read_csv(filepath_or_buffer=f"{processed_path}/train.csv")
        logger.info(f"Read trained data from : {processed_path}")
    else:
        strat_train_set = df
    features = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    labels = strat_train_set["median_house_value"].copy()
    return features, labels


def original_feature_engineering(df, pickle_path=None, imputer_file=None):
    """Function to train imputer, impute missing data and do feature engineering.

    Parameters
    ----------
    df : object
        The pandas dataframe of features for imputation and feature engineering.
    pickle_path : str
        The directory to store imputer pickle file.
    imputer_file : str
        The name of imputer file (for e.g - something.pkl).

    Returns
    -------
    imputer : object
        The sklearn.impute.SimpleImputer object.
    df_prepared : object
        The pandas dataframe of features after imputation and feature engineering.

    """
    df_num = df.drop("ocean_proximity", axis=1)
    logger.debug("Training imputer")
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_num)
    if pickle_path and imputer_file:
        pickle.dump(imputer, open(f"{pickle_path}/{imputer_file}", "wb"))
        logger.info(f"Saved {imputer_file} at : {pickle_path}")
    elif pickle_path:
        pickle.dump(imputer, open(f"{pickle_path}/imputer.pkl", "wb"))
        logger.info(f"Saved imputer pickle file at : {pickle_path}")
    else:
        pass
    X = imputer.transform(df_num)
    df_tr = pd.DataFrame(X, columns=df_num.columns, index=df.index)
    df_tr["rooms_per_household"] = df_tr["total_rooms"] / df_tr["households"]
    df_tr["bedrooms_per_room"] = df_tr["total_bedrooms"] / df_tr["total_rooms"]
    df_tr["population_per_household"] = df_tr["population"] / df_tr["households"]
    df_cat = df[["ocean_proximity"]]
    df_prepared = df_tr.join(pd.get_dummies(df_cat, drop_first=True))
    return imputer, df_prepared


def rf_regressor_model_training(df_prepared, labels, pickle_path=None, model_file=None):
    """Function to train model using Random Forest Regressor.

    Parameters
    ----------
    df_prepared : object
        The pandas dataframe of features after imputation and feature engineering.
    labels : object
        The pandas series of labels for supervised learning.
    pickle_path : str
        The directory to store model pickle file.
    model_file : str
        The name of model file (for e.g - something.pkl).

    Returns
    -------
    model : object
        The sklearn.ensemble.RandomForestRegressor object.

    """
    logger.debug("Training model")
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(df_prepared, labels)
    model = grid_search.best_estimator_
    if pickle_path and model_file:
        pickle.dump(model, open(f"{pickle_path}/{model_file}", "wb"))
        logger.info(f"Saved {model_file} at : {pickle_path}")
    elif pickle_path:
        pickle.dump(model, open(f"{pickle_path}/model.pkl", "wb"))
        logger.info(f"Saved model pickle file at : {pickle_path}")
    else:
        pass
    return model


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("setup.cfg")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-data",
        dest="PROCESSED",
        type=str,
        help="directory to read training data, default val in setup.cfg: data/processed",
    )
    parser.add_argument(
        "--pickle-data",
        dest="PICKLE",
        type=str,
        help="directory to save pickle files, default val in setup.cfg: artifacts",
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

    if args.PROCESSED:
        PROCESSED_DATA = args.PROCESSED
    else:
        PROCESSED_DATA = str(config["Default"]["processed_data"])

    if args.PICKLE:
        PICKLE_DATA = args.PICKLE
    else:
        PICKLE_DATA = str(config["Default"]["pickle_data"])

    IMPUTER_FILE = str(config["Default"]["imputer_file"])

    MODEL_FILE = str(config["Default"]["model_file"])

    housing, housing_labels = load_training_data(processed_path=PROCESSED_DATA)
    imputer, housing_prepared = original_feature_engineering(
        df=housing, pickle_path=PICKLE_DATA, imputer_file=IMPUTER_FILE
    )
    model = rf_regressor_model_training(
        df_prepared=housing_prepared,
        labels=housing_labels,
        pickle_path=PICKLE_DATA,
        model_file=MODEL_FILE,
    )
