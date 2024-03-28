# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] bash run/vlnbert_rxr_eval.bash

flag="--exp_name rxr.en_vlnbert_oldcam.slide1
      --run-type eval
      --exp-config exp/vlnbert_rxr_en.yaml

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
      IL.lr 3.5e-5
      IL.schedule_ratio 0.50
      IL.max_traj_len 20
      "
python run.py $flag