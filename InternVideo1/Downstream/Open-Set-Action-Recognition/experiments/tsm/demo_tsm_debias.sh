#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
DATASET=$2  # kinetics10, or mimetics10
case ${DATASET} in
  kinetics10)
    DATA_SPLIT='data/kinetics10/kinetics10_val_list_videos.txt'
	VIDEO_DIR='data/kinetics10/videos_val'
    ;;
  mimetics10)
    DATA_SPLIT='data/mimetics10/mimetics10_test_list_videos.txt'
	VIDEO_DIR='data/mimetics10/videos'
    ;;
  *)
    echo "Dataset is not supported: "${DATASET}
    exit
    ;;
esac

CUDA_VISIBLE_DEVICES=$1 python experiments/demo.py configs/recognition/tsm/inference_tsm_enn.py \
	--ckpt_dear work_dirs/tsm/train_kinetics10_tsm_DEAR/latest.pth \
  --ckpt_nodebias work_dirs/tsm/train_kinetics10_tsm_DEAR_noDebias/latest.pth \
	--split_file ${DATA_SPLIT} \
	--video_path ${VIDEO_DIR} \
	--result_list experiments/tsm/results_debias/demo_list_TSM_${DATASET}.txt

cd $pwd_dir
echo "Experiments finished!"