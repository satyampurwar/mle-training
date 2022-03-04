## Problem Statement
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
 - Development code is present in <base>/mle-training/reference/nonstandardcode
 - First draft to productionize code is present in <base>/mle-training/notebooks/personal
 - Random Forest technique is selected for deployment
## Virtual Environment
 - `cd <base>/mle-training`
 - `conda env create --file deploy/conda/linux_cpu_py39.yml`
 - `conda activate mle-dev`
 - Install dependencies using conda or pip based on requirement
 - Updating yml : `conda env export --name mle-dev > deploy/conda/linux_cpu_py39.yml`
 - To activate development environment : `conda activate mle-dev`
 - To deactivate current environment : `conda deactivate`
## Code Aesthetics
 - Configurations are mentioned in setup.cfg
 - Configurations are mentioned in .vscode/settings.json if using vscode
 - `black <script.py>`
 - `isort <script.py>`
 - `flake8 <script.py>`
## Execution of Scripts
 - `python src/housing_value/ingest_data.py --help`
 - `python src/housing_value/train.py --help`
 - `python src/housing_value/score.py --help`
## Testing of Scripts
 - Configurations are mentioned in setup.cfg
 - `pytest`
 - `pytest <test_directory>/<test.py>`
## Logging while Execution of Scripts
 - Logs captured while execution of scripts : `cat logs/main.log`
## Documentation using Sphinx
 - Make sure package is installed before documentation -
   - 1st Method :
      - To install package in editable mode : `pip install -e .`
   - 2nd Method :
      - `python3 -m pip install --upgrade build`
      - `python3 -m build`
      - `pip install dist/housing_value-0.0.0-py3-none-any.whl`
 - `cd docs`
 - `rm source/housing_value.rst source/modules.rst`
 - `sphinx-apidoc -o ./source ../src/housing_value`
 - Amendments : make amendments in **source/housing_value.rst** by following **reference/housing_value.rst** if required before build
 - `make clean`
 - `make html`
 - `cd ..`
## Source Packaging with setuptools
 <!-- ## Packaging with setup.py
 - To install package in editable mode : `pip install -e .`
 - To create package distribution : `python setup.py sdist` -->
 - Note : Packaging related metadata in setup.cfg (static) is preferred over setup.py (dynamic)
 - `python3 -m pip install --upgrade build`
 - `python3 -m build`
## Testing installed Package
 - Test Installation :
   - `pip uninstall housing_value`
   - `pip install dist/housing_value-0.0.0.tar.gz`
   - `pip uninstall housing_value`
   - `pip install dist/housing_value-0.0.0-py3-none-any.whl`
   - `pip uninstall housing_value`
   - `pip install dist/housing_value-0.0.0-py3-none-any.whl`
   - `python`
   - `import housing_value`
   - `exit()`
   - `pip uninstall housing_value`
 - Test the valid installation with scripts :
   - `pip install dist/housing_value-0.0.0-py3-none-any.whl`
   - Actually you must run this without src folder to check installation : `pytest`
   - `pip uninstall housing_value`
## Application Packaging with Mlflow
 - Cleaning existing mlruns if not required : `rm -r mlruns`
 - Tracking : `mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000`
 - Creating distribution of application if not created while source packaging or removed later :
   - `python3 -m pip install --upgrade build`
   - `python3 -m build`
   - Add distibution to pip list in deploy/conda/linux_cpu_py39.yml by adding : `../../dist/housing_value-0.0.0-py3-none-any.whl`
 - Packaging : `mlflow run . -P <parameters>`
 - Remember to pick correct experiment and run (i.e. train step) for standardization & deployment in below steps
 - Standardization of deployment environment :
   - `cp dist/housing_value-0.0.0-py3-none-any.whl mlruns/<experiment_id>/<run_id>/artifacts/model`
   - Source code distribution is specified in conda-lean.yaml
   - `cp deploy/conda/conda-lean.yaml mlruns/<experiment_id>/<run_id>/artifacts/model/conda.yaml`
## Logging while Mlflow Packaging
 - Logs captured while packaging of the application : `cat logs/main.log`
## Run Application with Mlflow
 - Deployment of trained model : `mlflow models serve -m mlruns/<experiment_id>/<run_id>/artifacts/model/ -h 127.0.0.1 -p 1234`
 - Testing API endpoint from other terminal : `curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"],"data":[[-118.39, 34.12, 29.0, 6447.0, 1012.0, 2184.0, 960.0, 8.2816, "<1H OCEAN"]]}' http://127.0.0.1:1234/invocations`
## Changing Configuration
 - Make changes in setup.cfg
 - Do fresh build
   - `python3 -m pip install --upgrade build`
   - `python3 -m build`
 - Run steps from **Testing installed Package** to **Logging while Mlflow Packaging**
## Bundling Deployment Artifacts for Containerization
 - `cd <base>/mle-training/deploy/docker`
 - Model Artifacts :
   - Manually copy main artifacts - **MLmodel & model.pkl** in **mlruns** folder which is a cleaned folder unlike original **mlruns** obtained while **Application Packaging with Mlflow**
   - Copy **whl file of source/application package** for installation in image/container
 - Cleaning Metadata : Remove the following irrevelant metadata in **MLmodel** file based on preferred deployment environment -
   - *env: conda.yaml* because --no-conda is opted in run.sh script while serving model
   - *python_version* because base image in Dockerfile is *FROM python:3.9-slim-bullseye* and version used while development was *python 3.9.5*
   - *model_uuid* because **mlruns** is cleaned for deployment
   - *run_id* because **mlruns** is cleaned for deployment
 - Image Developement :
   - Fix original versions of packages used during development in **requirements.txt**
   - **.dockerignore** - to ignore copying files in *WORKDIR* of image/container
   - **Dockerfile** to build image
 - Starting Container :
   - **run.sh** will start the application
     - **run.sh** is referred in *CMD* of Dockerfile
     - Make sure to bind the host server to avoid this error - curl: (52) Empty reply from server
 - Endpoint Testing :
   - **setup.cfg** is required to process the input data
     - Only Data related info is kept in this
     - This shall be present at location from where the code is relatively executed
 - Integration : Take care to integrate above reference in Dockerfile
## Containerizing Application
 - Image Developement : `docker build -t satyamta/housing:latest .`
 - Starting Container : `docker run -dit -p 8080:5000 --name my_app satyamta/housing:latest`
 - Testing the endpoint from host : `curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"],"data":[[-118.39, 34.12, 29.0, 6447.0, 1012.0, 2184.0, 960.0, 8.2816, "<1H OCEAN"]]}' http://0.0.0.0:8080/invocations`
 - Push Image to Dockerhub :
   - `docker login`
   - `docker push satyamta/housing:latest`
 - Delete Container & Image from current environment :
   - `docker rm -f my_app`
   - `docker rmi satyamta/housing:latest`
 - Retest in new environment :
   - Pull Image : `docker pull satyamta/housing:latest`
   - Starting Container : `docker run -dit -p 8080:5000 --name my_app satyamta/housing:latest`
   - Testing the endpoint from host : `curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"],"data":[[-118.39, 34.12, 29.0, 6447.0, 1012.0, 2184.0, 960.0, 8.2816, "<1H OCEAN"]]}' http://0.0.0.0:8080/invocations`