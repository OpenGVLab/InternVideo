# eval
# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] bash run/cma_rxr_eval.bash


#####################################################################

# flag="--exp_name cma_rxr_slide0
#       --run-type eval
#       --exp-config exp/cma_rxr.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       GPU_NUMBERS 1

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False

#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_rxr_slide0
#       --run-type inference
#       --exp-config exp/cma_rxr.yaml

#       SIMULATOR_GPU_IDS [$1]
#       TORCH_GPU_IDS [$1]
#       TORCH_GPU_ID $1
#       GPU_NUMBERS 1

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False

#       IL.batch_size 1
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_rxr_en_slide1>0
#       --run-type eval
#       --exp-config exp/cma_rxr_en.yaml

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
#       EVAL_CKPT_PATH_DIR data/logs/checkpoints/cma_rxr_slide1/ckpt.15.pth

#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_rxr_slide1_oldcamera
#       --run-type eval
#       --exp-config exp/cma_rxr.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       GPU_NUMBERS 1

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV 90
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV 90
#       RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS ['CenterCropperPerSensor']

#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_rxr_en_slide1>0
#       --run-type eval
#       --exp-config exp/cma_rxr_en.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       GPU_NUMBERS 1

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV 90
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV 90
#       RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS ['CenterCropperPerSensor']
#       EVAL_CKPT_PATH_DIR data/logs/checkpoints/cma_rxr_slide1_oldcamera/ckpt.25.pth

#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

# flag="--exp_name cma_rxr_$1_slide1>0_oldcamera
#       --run-type eval
#       --exp-config exp/cma_rxr_$1.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       GPU_NUMBERS 1

#       TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 224
#       TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV 90
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 256
#       TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV 90
#       RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS ['CenterCropperPerSensor']
#       EVAL_CKPT_PATH_DIR $2
      
#       IL.batch_size 11
#       IL.lr 1e-4
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

#####################################################################

flag="--exp_name rxr.en_cma_oldcam.slide1
      --run-type eval
      --exp-config exp/cma_rxr_en.yaml

      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      TORCH_GPU_ID 0
      GPU_NUMBERS 1

      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 224
      TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 224
      TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV 90
      TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 256
      TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 256
      TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV 90
      RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS ['CenterCropperPerSensor']

      IL.batch_size 11
      IL.lr 3.5e-4
      IL.schedule_ratio 0.75
      IL.max_traj_len 20
      "
python run.py $flag
