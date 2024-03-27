torchrun --rdzv_endpoint=${MASTER_NODE}:${MASTER_PORT} \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks/pretrain.py \
    $(dirname $0)/config.py \
    wandb.enable False \
    model.vision_encoder.pretrained 'CLIP-ViT-B/16' \
    model.text_encoder.pretrained 'CLIP-ViT-B/16' \
    output_dir ${OUTPUT_DIR}
