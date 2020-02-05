#!/bin/bash
export CONDA_ALWAYS_YES="true"
conda create -n kamn python=3.7
conda install -n kamn anaconda
conda install -n kamn pytorch cudatoolkit=10.1 -c pytorch
conda install ignite -c pytorch
conda install -n kamn -c anaconda gensim
unset CONDA_ALWAYS_YES 
conda init --all --dry-run --verbose
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
source activate kamn
pip install -U spacy
pip install -U transformers 
pip install -U neuralcoref --no-binary neuralcoref
pip install -U tensorflow
pip install -U tensorboardX
pip install tensorflow_datasets

python -m spacy download en_core_web_lg
