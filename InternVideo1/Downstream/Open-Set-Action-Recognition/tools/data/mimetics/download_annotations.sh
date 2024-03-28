#!/usr/bin/env bash

DATASET='mimetics'
DATA_DIR="../../../data/${DATASET}/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://europe.naverlabs.com/wp-content/uploads/2019/12/Mimetics_release_v1.0.zip

unzip Mimetics_release_v1.0.zip -d ${DATA_DIR}/

rm Mimetics_release_v1.0.zip
