# Problem Statement
 - The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.
 - The following techniques have been used:
    - Linear regression
    - Decision Tree
    - Random Forest
## Research & Development
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.
 - Development code is present in <base>/mle-training/notebooks/reference/nonstandardcode
 - First draft to productionize code is present in <base>/mle-training/notebooks/personal
 - Random Forest technique is selected for deployment
## Virtual Environment
 - `cd <base>/mle-training`
 - `conda env create --file deploy/conda/linux_cpu_py39.yml`
 - `conda activate mle-dev`
 - Install dependencies using conda or pip based on requirement
 - `conda env export --name mle-dev > deploy/conda/linux_cpu_py39.yml`
 - To deactivate current environment : `conda deactivate`
 - To activate development environment : `conda activate mle-dev`
## Code Aesthetics
 - Configurations are mentioned in setup.cfg
 - Configurations are mentioned in .vscode/settings.json if using vscode
 - `black <script.py>`
 - `isort <script.py>`
 - `flake8 <script.py>`
## Packaging
 - To install package in editable mode : `pip install -e .`
 - To create package distribution : `python setup.py sdist`
## Execution of Scripts
 - `python src/housing_value/ingest_data.py --help`
 - `python src/housing_value/train.py --help`
 - `python src/housing_value/score.py --help`
## Testing of scripts
 - Configurations are mentioned in setup.cfg
 - `pytest`
 - `pytest <test_directory>/<test.py>`
## Documentation
 - `cd docs`
 - `rm source/housing_value.rst source/modules.rst`
 - `sphinx-apidoc -o ./source ../src/housing_value`
 - Amendments : make amendments in **source/housing_value.rst** by following **notebooks/reference/housing_value.rst** if required before build
 - `make clean`
 - `make html`
 - `cd ..`
 ## MLflow
 - Cleaning : `rm -r mlruns`
 - Tracking : `mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000`
 - Creating distribution of owned application :
    - `pip install -e .`
    - `python setup.py sdist`
    - Add distibution to pip list in deploy/conda/linux_cpu_py39.yml by adding : `../../dist/housing_value-0.0.0.tar.gz`
 - Packaging : `mlflow run . -P <parameters>`
 - Remember to pick correct experiment and run for standardization & deployment in below steps.
 - Standardization of deployment environment :
   - `cp dist/housing_value-0.0.0.tar.gz mlruns/<experiment_id>/<runid>/artifacts/model`
   - `cp deploy/conda/conda-lean.yaml mlruns/<experiment_id>/<runid>/artifacts/model/conda.yaml`
 - Deployment of trained model : `mlflow models serve -m mlruns/<experiment_id>/<runid>/artifacts/model/ -h 127.0.0.1 -p 1234`
 - Testing API endpoint : `curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"],"data":[[-118.39, 34.12, 29.0, 6447.0, 1012.0, 2184.0, 960.0, 8.2816, "<1H OCEAN"]]}' http://127.0.0.1:1234/invocations`