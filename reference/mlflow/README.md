# Managing the end-to-end machine learning lifecycle with MLFlow
## TRACKING
- MLFlow UI (for tracking parameters, metrics, tags & artifacts)
- Run the below command in your project root and it will create baseline (mlruns folder and default metadata)
- `mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000`
## PACKAGING
- Refer MLproject file for configurations of yaml, endpoints, parameters etc.
- MLFlow Project for creating envionment, running script and doing packaging & pickling
- Run the below command and it will also create baseline (mlruns folder and default metadata) if "TRACKING" command is not executed yet
- `mlflow run . -P <parameters>`
## STANDARDIZATION
- Replace created yaml with original yaml during packaging : `cp conda.yaml mlruns/<experiment_id>/<runid>/artifacts/model/conda.yaml`
## DEPLOYMENT
- MLFlow Serve (for utilizing above environent and deployment of already created packaging & pickling)
- Run the below command below to create environment based on yml/yaml file if it doesn't exist and then it  will serve the model pickle through invocations
- `mlflow models serve -m mlruns/<experiment_id>/<runid>/artifacts/model/ -h 127.0.0.1 -p 1234`
## TESTING
- Testing endpoint from other terminal
- `curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations`
## TAGGING
- Adding of Tags like "date" and "deployed" to TRACKING
- Edit <run_id> in tagging.py
- `python tagging.py`
## REFERENCE
- https://github.com/tsterbak/pydataberlin-2019
