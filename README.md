# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.
 - Development code is present in <project_folder>/mle-training/notebooks/reference
 - First draft of to productionize code is present in <project_folder>/mle-training/notebooks/personal
 - Random Forest technique is selected for deployment

## Steps to setup the development environment
 - cd <project_folder>/mle-training
 - conda env create --file deploy/conda/linux_cpu_py39.yml
 - conda activate mle-dev
 - install dependencies using conda or pip
 - conda env export --name mle-dev > deploy/conda/linux_cpu_py39.yml
## To deactivate current environment
 - conda deactivate

The above steps to setup development will not be required as you can spin up environment with help of env.yml file already built and existing in mle-training repository, so, follow below steps -

## To activate development environment
 - conda activate mle-dev

## Steps to format the script from console
 - black <script.py>
 - isort <script.py>
 - flake8 <script.py>

## To install a package in editable mode
 - pip install -e .

## To create package distribution
 - python setup.py sdist

## To execute the default scripts
 - cd <project_folder>/mle-training
 - python src/housing_value/ingest_data.py
 - python src/housing_value/train.py
 - python src/housing_value/score.py

## To see arguments that can be passed to above scripts
 - python src/housing_value/ingest_data.py --help
 - python src/housing_value/train.py --help
 - python src/housing_value/score.py --help

## Testing
 - pytest
 - pytest <test_directory>/<test.py>

## To rewrite html documentation using sphinx
 - cd <project_folder>/mle-training/docs
 - sphinx-apidoc -o ./source ../src/housing_value
 - make amendments in source/housing_value.rst if required before build
 - make clean
 - make html