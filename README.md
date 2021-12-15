# mle-training
Tamlep
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
## To activate development environment
conda activate mle-dev
## To excute the script
cd <project_folder>/mle-training
python nonstandardcode/nonstandardcode.py

