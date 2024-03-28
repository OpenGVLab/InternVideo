# srun -p GVT -n1 -c10 --gres=gpu:3 -x SH-IDC1-10-198-8-[61,62] bash run/vlnbert_r2r.bash

flag="--exp_name r2r_vlnbert_slide1
      --run-type train
      --exp-config exp/vlnbert_r2r.yaml

      SIMULATOR_GPU_IDS [0,1,2]
      TORCH_GPU_IDS [0,1,2]
      GPU_NUMBERS 3

      IL.batch_size 20
      IL.lr 3.5e-5
      IL.schedule_ratio 0.50
      IL.max_traj_len 20
      "
python -m torch.distributed.launch --nproc_per_node=3 run.py $flag