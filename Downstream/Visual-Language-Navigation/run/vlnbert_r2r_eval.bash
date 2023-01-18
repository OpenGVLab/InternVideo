# eval
# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] bash run/vlnbert_r2r_eval.bash

flag="--exp_name r2r_vlnbert_slide1
      --run-type eval
      --exp-config exp/vlnbert_r2r.yaml

      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      TORCH_GPU_ID 0
      GPU_NUMBERS 1

      IL.batch_size 11
      IL.lr 3.5e-5
      IL.schedule_ratio 0.50
      IL.max_traj_len 20
      "
python run.py $flag