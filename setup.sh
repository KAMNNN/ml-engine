#!/bin/bash
export CONDA_ALWAYS_YES="true"
conda create -n kamn python=3.7
conda install -n kamn anaconda
conda install -n kamn pytorch cpuonly -c pytorch
conda install ignite -c pytorch
conda install -n kamn -c anaconda gensim
unset CONDA_ALWAYS_YES 
conda init --all --dry-run --verbose
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
source activate kamn
pip install transformers 
git submodule init
git submodule update
cd external/neuralcoref
pip install -r requirements.txt
pip install -e .
python -m spacy download en
cd ../../
