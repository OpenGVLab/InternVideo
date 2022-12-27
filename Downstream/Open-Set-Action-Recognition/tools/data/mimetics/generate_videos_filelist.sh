#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "mimetics10" ] || [ "$DATASET" == "mimetics" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support mimetics10, mimetics"
        exit 0
fi

pwd_dir=$pwd
cd ../../../

PYTHONPATH=. python tools/data/mimetics/build_file_list.py ${DATASET} data/${DATASET}/videos/ data/${DATASET}/${DATASET}_test_list_videos.txt
echo "test filelist for video generated."

cd ${pwd_dir}