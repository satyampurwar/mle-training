# run.sh
mlflow models serve -m $ARTIFACTS_STORE -h $SERVER_HOST --no-conda