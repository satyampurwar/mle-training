# mle-training
Tamlep
## Steps to setup the development environment
 - conda create --name mle-dev python=3.9.5
 - conda activate mle-dev
 - cd <project_folder>/mle-training
 - conda env export --name mle-dev > env.yml /
Installing utilities:
 - conda install --yes numpy
 - conda install --yes pandas
 - conda install --yes matplotlib
 - conda install --yes scikit-learn /
Installing code formatters:
 - conda install --yes black
 - conda install --yes isort
 - conda install --yes flake8 /
Exporting environment file:
 - conda env export --name mle-dev > env.yml
## To deactivate current environment
 - conda deactivate

The above steps to setup development will not be required as you can spin up environment with help of env.yml file already built and existing in mle-training repository, so, follow below steps -

## To activate development environment
 - conda activate mle-dev
## To execute the script
 - cd <project_folder>/mle-training
 - python nonstandardcode/nonstandardcode.py

## Steps to format the script
 - black nonstandardcode/nonstandardcode.py
 - isort nonstandardcode/nonstandardcode.py
 - flake8 nonstandardcode/nonstandardcode.py
