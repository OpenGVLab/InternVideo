#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "mimetics10" ] || [ "$DATASET" == "mimetics" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support mimetics10 or mimetics"
        exit 0
fi

pwd_dir=$pwd

cd ../../../
PYTHONPATH=. python tools/data/data_check.py data/${DATASET}/videos data/${DATASET}/${DATASET}_test_list_videos.txt test
echo "Test filelist for video passed checking."

cd ${pwd_dir}
