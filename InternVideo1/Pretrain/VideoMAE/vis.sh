# Set the path to save images
OUTPUT_DIR='/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/demo/vis_k400_1_0.9'
# path to image for visualization
# IMAGE_PATH='/apdcephfs/share_1290939/elliottong/dataset/Diving48/Y7QZcr24ye0_00538.mp4'
# IMAGE_PATH='/apdcephfs/share_1290939/elliottong/dataset/20bn-something-something-v2/16977.mp4'
IMAGE_PATH='/apdcephfs/share_1290939/elliottong/kinetics_400_val_10s_320p/-B2oGkg1qSI.mp4'


# path to pretrain model
# MODEL_PATH='/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/diving_pretrain_mae_base_patch16_224_frame_16x2_tc_mask_0.9_lr_3e-4_new_e3200/checkpoint-3199.pth'
# MODEL_PATH='/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/ssv2_pretrain_mae_base_patch16_224_frame_16x2_tc_mask_0.9_new_e2400/checkpoint-2399.pth'
MODEL_PATH='/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/ssv2_pretrain_mae_base_patch16_224_frame_16x2_tc_mask_0.9_new_e2400/checkpoint-2399.pth'


# Now, it only supports pretrained models with normalized pixel targets
python3 run_mae_vis.py \
    --mask_ratio 0.9 \
    --mask_type t_consist \
    --decoder_depth 4 \
    --model pretrain_mae_base_patch16_224 \
    --mask_type t_consist \
    ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH} 