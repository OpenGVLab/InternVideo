#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl

DATASET=$1
if [ "$DATASET" == "kinetics400" ] ; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400!"
        exit 0
fi

DATA_DIR="../../../data/kinetics10"
ANNO_DIR="../../../data/${DATASET}/annotations"
SUBSET="./subset_list.txt"
python download_subset.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/videos_train --subset_file ${SUBSET} -t /ssd/data/tmp/kinetics10 -n 1
python download_subset.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/videos_val --subset_file ${SUBSET} -t /ssd/data/tmp/kinetics10 -n 1

source deactivate kinetics
conda remove -n kinetics --all
