NUM_NODES=1
NUM_GPUS=8
JOB_NAME="vtc_tgifqa_lr1e5"
MODEL_PREFIX="."
OUTPUT_DIR="outputs/outputs_clip_kc_nc_tgifqa"
LOG_FILE="${OUTPUT_DIR}/logs/${JOB_NAME}/log.txt"
N_TASKS=`echo "${NUM_NODES}*${NUM_GPUS}" | bc`

srun --async -p video -n${N_TASKS} --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=${NUM_GPUS} --cpus-per-task=16 -o ${LOG_FILE}\
    --quotatype reserved --job-name ${JOB_NAME} \
    --open-mode=append \
    python run.py with data_root=./meta_data num_gpus=${NUM_GPUS} num_nodes=${NUM_NODES} \
    per_gpu_batchsize=32 clip_finetune_tgifqa \
    num_frames=8 \
    num_workers=8 \
    batch_size=512 \
    max_epoch=20 \
    model_dir=${OUTPUT_DIR}/models/${JOB_NAME} \
    log_dir=${OUTPUT_DIR}/result/${JOB_NAME} \
    resume_from=${OUTPUT_DIR}/models/${JOB_NAME}/last.ckpt \
    save_checkpoints_interval=1000 \
    learning_rate=1e-5 \
    clip_qa_type=vtc_cap \
    save_last=False \
    save_top_k=0 \
    max_steps=-1 \
    clip=/pathto/ViT-L-14.pt \
    clip_type=kc_new \
    load_path=/pathto/InternVideo-MM-L-14.ckpt

rm batchscript-*

while [ ! -f ${LOG_FILE} ] ; do sleep 1; done
tail -f ${LOG_FILE}