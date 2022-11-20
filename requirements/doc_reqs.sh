#!/bin/bash
set -e -x
# This script will document the requirements for multiple conda environments
# It will capture the requirements in multiple ways each of which has pros and cons

# inputs
PROJECT_NAMES='deeptime'

for PROJECT_NAME in $PROJECT_NAMES
do
    echo $PROJECT_NAME
    PYTHON_INTERPRETER=~/miniforge3/envs/$PROJECT_NAME/bin/python
    # minimal requirement, simpler, but no versions or pip
    conda env export --no-builds --from-history > requirements/environment.min.yaml
    # extensive requirements including pip and information overload
    conda env export > requirements/environment.max.yaml
    # requirements in a modified pip spec, usefull for dependabot and so on
    $PYTHON_INTERPRETER -m pip freeze > requirements/pip.conda.txt
done

# inputs
for PROJECT_NAME in $PROJECT_NAMES
do
    echo $PROJECT_NAME
    # conda lock is good for not overspecifying version, but it misses pip
    cd requirements && conda-lock -f environment.max.yaml -p linux-64
done