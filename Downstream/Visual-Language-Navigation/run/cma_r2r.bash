device=$1
# train
# srun -p GVT -n1 -c10 --gres=gpu:3 -x SH-IDC1-10-198-8-[61,62] bash run/cma_r2r.bash 0,1,2

#####################################################################

flag="--exp_name cma_r2r_slide1
      --run-type train
      --exp-config exp/cma_r2r.yaml

      SIMULATOR_GPU_IDS [$device]
      TORCH_GPU_IDS [$device]
      GPU_NUMBERS 3

      IL.batch_size 20
      IL.lr 1e-4
      IL.schedule_ratio 0.75
      IL.max_traj_len 20
      "
python -m torch.distributed.launch --nproc_per_node=3 run.py $flag

#####################################################################

# flag="--exp_name cma_r2r_slide0
#       --run-type train
#       --exp-config exp/cma_r2r.yaml

#       SIMULATOR_GPU_IDS [$device]
#       TORCH_GPU_IDS [$device]
#       GPU_NUMBERS 3

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False

#       IL.batch_size 20
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python -m torch.distributed.launch --nproc_per_node=3 run.py $flag