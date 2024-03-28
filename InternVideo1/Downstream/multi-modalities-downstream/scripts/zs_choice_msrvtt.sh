set -e

unset https_proxy
unset http_proxy
unset all_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ALL_PROXY
echo "Proxy is unset!"

###################### zero-shot test ######################
NUM_NODES=1
NUM_GPUS=8
BATCH_SIZE=`echo "1024*${NUM_NODES}" | bc`
JOB_NAME="zs_choice"
MODEL_PREFIX="."
OUTPUT_DIR="outputs/outputs_clip_kc_nc"
LOG_FILE="${OUTPUT_DIR}/logs/${JOB_NAME}/log.txt"
N_TASKS=`echo "${NUM_NODES}*${NUM_GPUS}" | bc`

srun --async -p video -n${N_TASKS} --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=${NUM_GPUS} --cpus-per-task=14 -o ${LOG_FILE}\
    --quotatype auto --job-name ${JOB_NAME} \
    --open-mode=append \
    python run.py with data_root=s3://video_pub/ num_gpus=${NUM_GPUS} num_nodes=${NUM_NODES} \
    per_gpu_batchsize=128 clip_kc_nc_finetune_msrvttchoice test_only=True \
    batch_size=${BATCH_SIZE} \
    num_frames=8 num_workers=12 \
    model_dir=${MODEL_PREFIX}/${OUTPUT_DIR}/models/${JOB_NAME} \
    log_dir=${OUTPUT_DIR}/result/${JOB_NAME} \
    'video_datasets=["msrvtt_choice"]' \
    clip_wiseft_coef=0.5 \
    clip=/pathto/ViT-L-14.pt \
    clip_type=kc_new \
    load_path=/pathto/InternVideo-MM-L-14.ckpt

rm batchscript-*

while [ ! -f ${LOG_FILE} ] ; do sleep 1; done
tail -f ${LOG_FILE}