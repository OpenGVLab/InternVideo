# Fine-tuning VideoMAE 

## Original Implementation

The implementation of our VideoMAE supports **multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts folder](scripts).

-  For example, to fine-tune VideoMAE ViT-Base on **Something-Something V2** with 64 GPUs (8 nodes x 8 GPUs), you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/eval_lr_1e-3_epoch_40'
  DATA_PATH='YOUR_PATH/list_ssv2'
  MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-2399.pth'
  
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
      --master_port 12320 --nnodes=8 \
      --node_rank=0 --master_addr=$ip_node_0 \
      run_class_finetuning.py \
      --model vit_base_patch16_224 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 8 \
      --num_sample 1 \
      --input_size 224 \
      --short_side_size 224 \
      --save_ckpt_freq 10 \
      --num_frames 16 \
      --sampling_rate 2 \
      --opt adamw \
      --lr 1e-3 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 40 \
      --dist_eval \
      --test_num_segment 2 \
      --test_num_crop 3 \
      --enable_deepspeed 
  ```

  on the first node. On other nodes, run the same command with `--node_rank 1`, ..., `--node_rank 7` respectively.  `--master_addr` is set as the ip of the node 0.

- For example, to fine-tune VideoMAE ViT-Base on **Kinetics400** with 64 GPUs (8 nodes x 8 GPUs), you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/eval_lr_1e-3_epoch_100'
  DATA_PATH='YOUR_PATH/list_kinetics-400'
  MODEL_PATH='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/checkpoint-799.pth'
  
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
      --master_port 12320 --nnodes=8 \
      --node_rank=0 --master_addr=$ip_node_0 \
      run_class_finetuning.py \
      --model vit_base_patch16_224 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 8 \
      --num_sample 1 \
      --input_size 224 \
      --short_side_size 224 \
      --save_ckpt_freq 10 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 1e-3 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 100 \
      --dist_eval \
      --test_num_segment 5 \
      --test_num_crop 3 \
      --enable_deepspeed
  ```

  on the first node. On other nodes, run the same command with `--node_rank 1`, ..., `--node_rank 7` respectively.  `--master_addr` is set as the ip of the node 0.

### Note:

- We didn't use `cls token` in our implementation, and directly average the feature of last layer for video classification.
- Here total batch size = (`batch_size` per gpu) x `nodes` x (gpus per node).
- `lr` here is the base learning rate. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.

## Slurm

TODO