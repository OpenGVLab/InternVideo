# train
# srun -p GVT -n1 -c10 --gres=gpu:3 -x SH-IDC1-10-198-8-[61,62] bash run/cma_rxr.bash


#####################################################################

# flag="--exp_name cma_rxr_slide0
#       --run-type train
#       --exp-config exp/cma_rxr.yaml

#       SIMULATOR_GPU_IDS [0,1,2]
#       TORCH_GPU_IDS [0,1,2]
#       GPU_NUMBERS 3

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False

#       IL.batch_size 19
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python -m torch.distributed.launch --nproc_per_node=3 run.py $flag

#####################################################################

# flag="--exp_name cma_rxr_slide1
#       --run-type train
#       --exp-config exp/cma_rxr.yaml

#       SIMULATOR_GPU_IDS [0,1,2]
#       TORCH_GPU_IDS [0,1,2]
#       GPU_NUMBERS 3

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

#       IL.batch_size 20
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python -m torch.distributed.launch --nproc_per_node=3 run.py $flag

#####################################################################
# lang=$1

# flag="--exp_name cma_rxr_${lang}_slide1_oldcamera
#       --run-type train
#       --exp-config exp/cma_rxr_${lang}.yaml

#       SIMULATOR_GPU_IDS [0,1,2]
#       TORCH_GPU_IDS [0,1,2]
#       GPU_NUMBERS 3

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV 90
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV 90
#       RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS ['CenterCropperPerSensor']

#       IL.batch_size 20
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20

#       IL.load_from_ckpt True
#       IL.ckpt_to_load $2
#       IL.is_requeue True
#       "
# python -m torch.distributed.launch --nproc_per_node=3 run.py $flag
#####################################################################

flag="--exp_name rxr.en_cma_oldcam.slide1
      --run-type train
      --exp-config exp/cma_rxr_en.yaml

      SIMULATOR_GPU_IDS [0,1,2]
      TORCH_GPU_IDS [0,1,2]
      GPU_NUMBERS 3

      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 224
      TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 224
      TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV 90
      TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 256
      TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 256
      TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV 90
      RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS ['CenterCropperPerSensor']

      IL.batch_size 20
      IL.lr 3.5e-4
      IL.schedule_ratio 0.75
      IL.max_traj_len 20
      "
python -m torch.distributed.launch --nproc_per_node=3 run.py $flag