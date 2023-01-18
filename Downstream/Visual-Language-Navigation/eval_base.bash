# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] bash iter_eval.bash

ngpus=1
flag="--exp_name eval_base
      --run-type eval
      --exp-config iter_train.yaml

      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS $ngpus
      NUM_ENVIRONMENTS 11

      EVAL.SAVE_RESULTS False
      EVAL.CKPT_PATH_DIR pretrained/pretrained_models/base_ckpt.iter6900.pth
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "
python run.py $flag
