#!/bin/bash
pwd_dir=$pwd
cd ../../
GPUS=$1
PORT=${PORT:-29498}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/train.py configs/recognition/mae/finetune_ucf101_mae_edlnokl.py \
	--work-dir work_dirs/mae/ky \
	--validate \
	--seed 0 \
	--deterministic \
	--launcher pytorch \

cd $pwd_dir
echo "Experiments finished!"
