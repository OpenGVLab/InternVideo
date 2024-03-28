#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "kinetics10" ] || [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

pwd_dir=$pwd

cd ../../../
PYTHONPATH=. python tools/data/data_check.py data/${DATASET}/videos_train data/${DATASET}/${DATASET}_train_list_videos.txt train
echo "Train filelist for video passed checking."

PYTHONPATH=. python tools/data/data_check.py data/${DATASET}/videos_val data/${DATASET}/${DATASET}_val_list_videos.txt val
echo "Val filelist for video passed checking."
cd ${pwd_dir}
