# srun -p GVT -n1 -c6 --gres=gpu:4 -x SH-IDC1-10-198-8-[61,62,79] bash iter_train.bash

# ngpus=4
# flag="--exp_name hamt
#       --run-type train
#       --exp-config iter_train.yaml

#       SIMULATOR_GPU_IDS [0,1,2,3]
#       TORCH_GPU_IDS [0,1,2,3]
#       GPU_NUMBERS $ngpus
#       NUM_ENVIRONMENTS 12

#       IL.iters 5000
#       IL.lr 3e-5
#       IL.log_every 100
#       IL.waypoint_aug True
#       "
# python -m torch.distributed.launch --nproc_per_node=$ngpus run.py $flag

ngpus=6
flag="--exp_name hamt_video+clip_k400_6x8_large
      --run-type train
      --exp-config iter_train.yaml

      SIMULATOR_GPU_IDS [0,1,2,3,4,5]
      TORCH_GPU_IDS [0,1,2,3,4,5]
      GPU_NUMBERS $ngpus
      NUM_ENVIRONMENTS 8

      IL.iters 10000
      IL.lr 3e-5
      IL.log_every 150
      IL.feedback teacher
      IL.resume False
      "
      
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=$ngpus run.py $flag