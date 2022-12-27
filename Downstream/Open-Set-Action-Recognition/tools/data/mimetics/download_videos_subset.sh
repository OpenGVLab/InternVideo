#!/usr/bin/env bash

# set up environment
conda env create -f ../kinetics/environment.yml
source activate kinetics
pip install --upgrade youtube-dl

DATA_DIR="../../../data/mimetics10"
ANNO_DIR="../../../data/mimetics/annotations"
SUBSET="../kinetics/subset_list.txt"
python ../kinetics/download_subset.py ${ANNO_DIR}/mimetics_v1.0.csv ${DATA_DIR}/videos --subset_file ${SUBSET} -t /ssd/data/tmp/mimetics10 -n 1 

source deactivate kinetics
# conda remove -n kinetics --all
