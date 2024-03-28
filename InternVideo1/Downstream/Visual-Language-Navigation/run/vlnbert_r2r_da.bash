# srun -p GVT -n1 -c5 --gres=gpu:4 -x SH-IDC1-10-198-8-[61,62,79] bash run/vlnbert_r2r_da.bash

p=0.5
bs=32
diter=$1
epoch=$2
ngpus=4
flag="--exp_name r2r_vlnbert_da.p${p}.bs${bs}.di${diter}.ep${epoch}
      --run-type train
      --exp-config exp/vlnbert_r2r_da.yaml

      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS $ngpus
      NUM_ENVIRONMENTS 15

      IL.lr 3.5e-5
      IL.batch_size $bs
      IL.max_traj_len 20
      IL.epochs $epoch
      IL.DAGGER.iterations $diter
      IL.DAGGER.update_size 6000
      IL.DAGGER.p $p
      IL.DAGGER.preload_lmdb_features False
      IL.DAGGER.lmdb_features_dir data/trajectories_dirs/r2r_vlnbert_da.p${p}.bs${bs}.di${diter}.ep${epoch}/trajectories.lmdb
      "
python -m torch.distributed.launch --nproc_per_node=$ngpus run.py $flag

# p=0.75
# bs=8
# diter=$1
# epoch=$2
# ngpus=2
# flag="--exp_name r2r_vlnbert_da.p${p}.bs${bs}.di${diter}.ep${epoch}
#       --run-type train
#       --exp-config exp/vlnbert_r2r_da.yaml

#       SIMULATOR_GPU_IDS [0,1]
#       TORCH_GPU_IDS [0,1]
#       GPU_NUMBERS $ngpus
#       NUM_ENVIRONMENTS 4

#       IL.lr 3.5e-5
#       IL.batch_size $bs
#       IL.max_traj_len 20
#       IL.epochs $epoch
#       IL.DAGGER.iterations $diter
#       IL.DAGGER.update_size 100
#       IL.DAGGER.p $p
#       IL.DAGGER.preload_lmdb_features False
#       IL.DAGGER.lmdb_features_dir data/trajectories_dirs/r2r_vlnbert_da.p${p}.bs${bs}.di${diter}.ep${epoch}/trajectories.lmdb
#       "
# python -m torch.distributed.launch --nproc_per_node=$ngpus run.py $flag
