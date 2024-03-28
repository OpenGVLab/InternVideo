# eval
# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] bash run/cma_r2r_eval.bash

#####################################################################

# flag="--exp_name cma_r2r_slide1
#       --run-type eval
#       --exp-config exp/cma_r2r.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       GPU_NUMBERS 1

#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_r2r_slide0
#       --run-type eval
#       --exp-config exp/cma_r2r.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       GPU_NUMBERS 1

#       VIDEO_OPTION ['tensorboard']
#       EVAL_CKPT_PATH_DIR data/logs/checkpoints/cma_r2r_slide0/ckpt.31.pth

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False

#       IL.batch_size 1
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_r2r_slide0
#       --run-type inference
#       --exp-config exp/cma_r2r.yaml

#       SIMULATOR_GPU_IDS [1]
#       TORCH_GPU_IDS [1]
#       TORCH_GPU_ID 1
#       GPU_NUMBERS 1

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False

#       IL.batch_size 18
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name slide1_to_slide0
#       --run-type eval
#       --exp-config exp/cma_r2r.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [3]
#       TORCH_GPU_ID 3
#       GPU_NUMBERS 1

#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20

#       EVAL_CKPT_PATH_DIR data/logs/checkpoints/cma_r2r_slide1/ckpt.40.pth
#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
#       "
# python run.py $flag
