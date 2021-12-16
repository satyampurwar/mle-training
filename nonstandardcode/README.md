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

## Steps to setup the development environment
conda create --name mle-dev python=3.9.5
conda activate mle-dev
cd <project_folder>/mle-training
conda env export --name mle-dev > env.yml
conda install --yes numpy
conda install --yes pandas
conda install --yes matplotlib
conda install --yes scikit-learn
conda env export --name mle-dev > env.yml
## To deactivate current environment
conda deactivate

## The above steps to setup development will not be required as you can spin up environment with help 
## of env.yml file already existing in mle-training repository, follow below steps -
## To activate development environment
conda activate mle-dev
## To excute the script
cd <project_folder>/mle-training
python nonstandardcode/nonstandardcode.py