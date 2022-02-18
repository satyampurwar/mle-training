"""
Notes
-----
Use this script to run end-end ML pipeline using mlflow.
    $ mlflow run .
optional arguments/parameters:
    split_size : value to split data into train and test data, default val: 0.2
Example to use this script to run end-end ML pipeline using mlflow with split_size parameter.
    $ mlflow run . -P split_size=0.2
"""
import configparser
import logging
import sys

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from housing_value.ingest_data import data_ingestion, data_labeling, save_split_data
from housing_value.score import load_scoring_data, scoring_with_pipeline
from housing_value.train import load_training_data, training_with_pipeline

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "%(asctime)s %(name)s %(filename)s.%(funcName)s(%(lineno)d) %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("setup.cfg")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(str(config["Default"]["log_data"]))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(level=str(config["Default"]["log_level"]))

    HOUSING_URL = str(config["Default"]["raw_data_url"])
    HOUSING_PATH = str(config["Default"]["raw_data"])
    PROCESSED_DATA = str(config["Default"]["processed_data"])
    PICKLE_DATA = str(config["Default"]["pickle_data"])
    PIPE_FILE = str(config["Default"]["pipe_file"])
    OUTPUT_FILE = str(config["Default"]["output_file"])

    split_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2

    # Create nested runs
    with mlflow.start_run(run_name="PARENT_RUN") as parent_run:
        mlflow.log_param("parent_run", "yes")
        logger.info(f"parent_run run_id : {parent_run.info.run_id}")

        with mlflow.start_run(run_name="INGEST_DATA", nested=True) as ingest_data:
            mlflow.log_param("ingest_data", "yes")
            housing = data_ingestion(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
            housing = data_labeling(df=housing)
            strat_train_set, strat_test_set = save_split_data(
                df=housing, processed_path=PROCESSED_DATA, split_size=split_size
            )
            logger.info(f"ingest_data run_id : {ingest_data.info.run_id}")
            mlflow.log_param("split_size", split_size)
            # mlflow.log_artifact(f"{HOUSING_PATH}/housing.csv")
            # mlflow.log_artifact(f"{PROCESSED_DATA}/train.csv")
            # mlflow.log_artifact(f"{PROCESSED_DATA}/test.csv")
            # logger.info(f"saving artifacts at : {mlflow.get_artifact_uri()}")

        with mlflow.start_run(run_name="TRAIN_MODEL", nested=True) as train_model:
            mlflow.log_param("train_model", "yes")
            housing, housing_labels = load_training_data(processed_path=PROCESSED_DATA)
            pipe, best_param = training_with_pipeline(
                df=housing,
                labels=housing_labels,
                pickle_path=PICKLE_DATA,
                pipe_file=PIPE_FILE,
            )
            logger.info(f"train_model run_id : {train_model.info.run_id}")
            # mlflow.log_artifact(f"{PROCESSED_DATA}/train.csv")
            mlflow.log_param("best_estimator", best_param)
            signature = infer_signature(housing, pipe.predict(housing))
            mlflow.sklearn.log_model(pipe, "model", signature=signature)
            logger.info(f"Saving artifacts at : {mlflow.get_artifact_uri()}")

        with mlflow.start_run(run_name="SCORE_MODEL", nested=True) as score_model:
            mlflow.log_param("score_model", "yes")
            X_test, y_test = load_scoring_data(processed_path=PROCESSED_DATA)
            output, rmse = scoring_with_pipeline(
                df=X_test,
                actuals=y_test,
                pickle_path=PICKLE_DATA,
                pipe_file=PIPE_FILE,
                output_path=PROCESSED_DATA,
                output_file=OUTPUT_FILE,
            )
            logger.info(f"score_model run_id : {score_model.info.run_id}")
            # mlflow.log_artifact(f"{PROCESSED_DATA}/test.csv")
            # mlflow.log_artifact(f"{PICKLE_DATA}/{PIPE_FILE}")
            mlflow.log_artifact(f"{PROCESSED_DATA}/{OUTPUT_FILE}")
            mlflow.log_metric(key="rmse", value=rmse)
            logger.info(f"Saving artifacts at : {mlflow.get_artifact_uri()}")
