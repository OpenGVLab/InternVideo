# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62,79] bash run/vlnbert_r2r_da_eval.bash

p=0.5
bs=32
diter=$1
epoch=$2
ngpus=4
flag="--exp_name r2r_vlnbert_da.p${p}.bs${bs}.di${diter}.ep${epoch}
      --run-type eval
      --exp-config exp/vlnbert_r2r_da.yaml

      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS $ngpus
      NUM_ENVIRONMENTS 11

      IL.lr 3.5e-5
      IL.batch_size 11
      IL.max_traj_len 20
      "
python run.py $flag