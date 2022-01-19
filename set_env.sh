#!/bin/bash

conda deactivate
source ${ONEAPI_ROOT}/setvars.sh --force
conda env remove -n user_tensorflow
conda create -n user_tensorflow -c ${ONEAPI_ROOT}/conda_channel python=`python -V| awk '{print $2}'` -y
conda activate user_tensorflow
conda install -n user_tensorflow -c ${ONEAPI_ROOT}/conda_channel tensorflow python-flatbuffers -y
conda install -n user_tensorflow -c ${ONEAPI_ROOT}/conda_channel neural-compressor -y --offline
conda install -n user_tensorflow -c ${ONEAPI_ROOT}/conda_channel lpot -y --offline
conda install -n user_tensorflow runipy notebook -y