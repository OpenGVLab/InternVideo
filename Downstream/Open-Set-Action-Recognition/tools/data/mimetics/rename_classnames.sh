#!/usr/bin/env bash

pwd_dir=$pwd
# Rename classname for convenience
DATASET=$1
if [ "$DATASET" == "mimetics10" ] || [ "$DATASET" == "mimetics" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../../../data/${DATASET}/
ls ./videos | while read class; do \
  newclass=`echo $class | tr " " "_" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "videos/${class}" "videos/${newclass}";
  fi
done

cd ${pwd_dir}
