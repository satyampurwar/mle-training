## Virtual Environment
 - `cd <base>/extract`
 - `conda env create --file deploy/conda/linux_cpu_py39.yml`
 - `conda activate mle-dev`

## Testing
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
   - `pytest`
   - `pip uninstall housing_value`

## Application Packaging
 - Packaging : `mlflow run . -P <parameters>`
 - Remember to pick correct experiment and run (i.e. train step) for standardization & deployment in below steps : **experiment_id is 0 & run_id is 8adfa47e2b1d4018ac920d6200ae596d in this extract**
 - Standardization of deployment environment :
   - `cp dist/housing_value-0.0.0-py3-none-any.whl mlruns/<experiment_id>/<run_id>/artifacts/model`
   - Source code distribution is specified in conda-lean.yaml
   - `cp deploy/conda/conda-lean.yaml mlruns/<experiment_id>/<run_id>/artifacts/model/conda.yaml`

## Logging
 - Logs captured while packaging of the application : `cat logs/main.log`

## Run Application
 - Deployment of trained model : `mlflow models serve -m mlruns/<experiment_id>/<run_id>/artifacts/model/ -h 127.0.0.1 -p 1234`
 - Testing API endpoint from other terminal : `curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"],"data":[[-118.39, 34.12, 29.0, 6447.0, 1012.0, 2184.0, 960.0, 8.2816, "<1H OCEAN"]]}' http://127.0.0.1:1234/invocations`

## Changing Configuration
 - Make changes in setup.cfg
 - Add License, pyproject.toml & source code in this folder
 - Do fresh build
 - Run steps from **Testing** to **Logging**