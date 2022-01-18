"""
Notes
-----
Use this module for scoring the housing test data.
    $ python src/housing_value/score.py
optional arguments:
    -h, --help            show this help message and exit
    --processed-data PROCESSED
                          directory to read testing data, default val in setup.cfg: data/processed
    --pickle-data PICKLE  directory to read pickle files, default val in setup.cfg: artifacts
    --output-data OUTPUT  directory to store scored data, default val in setup.cfg: data/processed
    --log-level LEVEL     provide logging level, default val in setup.cfg: DEBUG
    --log-data LOG        file to store log data, default val in setup.cfg: logs/main.log if no console display is opted else default is console
                          display
    --no-console-log      do not display log on console
"""
import argparse
import configparser
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "%(asctime)s %(name)s %(filename)s.%(funcName)s(%(lineno)d) %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def load_scoring_data(df=None, processed_path=None):
    """Function to read testing data and return features and actuals.

    Parameters
    ----------
    df : object
        The pandas dataframe of testing data to bypass reading data from directory.
    processed_path : str
        The directory to read testing data.

    Returns
    -------
    X_test : object
        The pandas dataframe of features.
    y_test : object
        The pandas series of actuals.

    """
    logger.debug("Reading testing data")
    if processed_path:
        strat_test_set = pd.read_csv(filepath_or_buffer=f"{processed_path}/test.csv")
        logger.info(f"Read testing data from : {processed_path}")
    else:
        strat_test_set = df
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    return X_test, y_test


def replicate_feature_engineering(df, pickle_path=None, imputer_file=None):
    """Function to score missing values using imputer and do feature engineering.

    Parameters
    ----------
    df : object
        The pandas dataframe of features for imputation and feature engineering.
    pickle_path : str
        The directory to read imputer pickle file.
    imputer_file : str
        The name of imputer file (for e.g - something.pkl)

    Returns
    -------
    df_prepared : object
        The pandas dataframe of features after imputation and feature engineering.

    """
    df_num = df.drop("ocean_proximity", axis=1)
    logger.debug("Imputing test Data")
    if pickle_path and imputer_file:
        file = open(f"{pickle_path}/{imputer_file}", "rb")
    elif pickle_path:
        file = open(f"{pickle_path}/imputer.pkl", "rb")
    else:
        pass
    imputer = pickle.load(file)
    X = imputer.transform(df_num)
    df_tr = pd.DataFrame(X, columns=df_num.columns, index=df.index)
    df_tr["rooms_per_household"] = df_tr["total_rooms"] / df_tr["households"]
    df_tr["bedrooms_per_room"] = df_tr["total_bedrooms"] / df_tr["total_rooms"]
    df_tr["population_per_household"] = df_tr["population"] / df_tr["households"]
    df_cat = df[["ocean_proximity"]]
    df_prepared = df_tr.join(pd.get_dummies(df_cat, drop_first=True))
    return df_prepared


def scoring_test_data(
    df_prepared,
    actuals,
    pickle_path=None,
    model_file=None,
    output_path=None,
    output_file=None,
):
    """Function to score predictions based on inputs/features provided to trained model.

    Parameters
    ----------
    df_prepared : object
        The pandas dataframe of features after imputation and feature engineering.
    actuals : object
        The pandas series of labels for comparision with predictions.
    pickle_path : str
        The directory to read model pickle file.
    model_file : str
        The name of model file (for e.g - something.pkl).
    output_path : str
        The directory to store actuals & predictions.
    output_file : str
        The name of output file (for e.g - something.csv).

    Returns
    -------
    output : object
        The pandas dataframe containing actuals & predictions.
    rmse : float
        The root mean square error.

    """
    logger.debug("Scoring test Data")
    if pickle_path and model_file:
        file = open(f"{pickle_path}/{model_file}", "rb")
        model = pickle.load(file)
    elif pickle_path:
        file = open(f"{pickle_path}/model.pkl", "rb")
        model = pickle.load(file)
    else:
        pass
    predictions = model.predict(df_prepared)
    output = pd.DataFrame()
    output["Actual"] = actuals
    output["Prediction"] = predictions
    if output_path and output_file:
        output.to_csv(path_or_buf=f"{output_path}/{output_file}", index=False)
        logger.info(f"Stored {output_file} at : {output_path}")
    elif output_path:
        output.to_csv(path_or_buf=f"{output_path}/output.csv", index=False)
        logger.info(f"Stored scored/output data at : {output_path}")
    else:
        pass
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    logger.info(f"Mean of actuals : {np.mean(actuals)}")
    logger.info(f"rmse : {rmse}")
    logger.info(f"Ratio of rmse to mean : {np.round(rmse/np.mean(actuals),2)}")
    return output, rmse


def scoring_with_pipeline(
    df, actuals, pickle_path=None, pipe_file=None, output_path=None, output_file=None,
):
    """Function to score predictions based on inputs/features provided to trained pipeline.

    Parameters
    ----------
    df : object
        The pandas dataframe of features before imputation and feature engineering.
    actuals : object
        The pandas series of labels for comparision with predictions.
    pickle_path : str
        The directory to read model pickle file.
    pipe_file : str
        The name of pipe file (for e.g - something.pkl).
    output_path : str
        The directory to store actuals & predictions.
    output_file : str
        The name of output file (for e.g - something.csv).

    Returns
    -------
    output : object
        The pandas dataframe containing actuals & predictions.
    rmse : float
        The root mean square error.

    """
    logger.debug("Scoring with Pipeline")
    if pickle_path and pipe_file:
        file = open(f"{pickle_path}/{pipe_file}", "rb")
        pipe = pickle.load(file)
    elif pickle_path:
        file = open(f"{pickle_path}/pipe.pkl", "rb")
        pipe = pickle.load(file)
    else:
        pass
    predictions = pipe.predict(df)
    output = pd.DataFrame()
    output["Actual"] = actuals
    output["Prediction"] = predictions
    if output_path and output_file:
        output.to_csv(path_or_buf=f"{output_path}/{output_file}", index=False)
        logger.info(f"Stored {output_file} at : {output_path}")
    elif output_path:
        output.to_csv(path_or_buf=f"{output_path}/output.csv", index=False)
        logger.info(f"Stored scored/output data at : {output_path}")
    else:
        pass
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    logger.info(f"Mean of actuals : {np.mean(actuals)}")
    logger.info(f"rmse : {rmse}")
    logger.info(f"Ratio of rmse to mean : {np.round(rmse/np.mean(actuals),2)}")
    return output, rmse


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("setup.cfg")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-data",
        dest="PROCESSED",
        type=str,
        help="directory to read testing data, default val in setup.cfg: data/processed",
    )
    parser.add_argument(
        "--pickle-data",
        dest="PICKLE",
        type=str,
        help="directory to read pickle files, default val in setup.cfg: artifacts",
    )
    parser.add_argument(
        "--output-data",
        dest="OUTPUT",
        type=str,
        help="directory to store scored data, default val in setup.cfg: data/processed",
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

    if args.OUTPUT:
        OUTPUT_DATA = args.OUTPUT
    else:
        OUTPUT_DATA = str(config["Default"]["output_data"])

    # IMPUTER_FILE = str(config["Default"]["imputer_file"])

    # MODEL_FILE = str(config["Default"]["model_file"])

    PIPE_FILE = str(config["Default"]["pipe_file"])

    OUTPUT_FILE = str(config["Default"]["output_file"])

    X_test, y_test = load_scoring_data(processed_path=PROCESSED_DATA)

    # X_test_prepared = replicate_feature_engineering(
    #     df=X_test, pickle_path=PICKLE_DATA, imputer_file=IMPUTER_FILE
    # )

    # output, rmse = scoring_test_data(
    #     df_prepared=X_test_prepared,
    #     actuals=y_test,
    #     pickle_path=PICKLE_DATA,
    #     model_file=MODEL_FILE,
    #     output_path=PROCESSED_DATA,
    #     output_file=OUTPUT_FILE,
    # )

    output, rmse = scoring_with_pipeline(
        df=X_test,
        actuals=y_test,
        pickle_path=PICKLE_DATA,
        pipe_file=PIPE_FILE,
        output_path=PROCESSED_DATA,
        output_file=OUTPUT_FILE,
    )
