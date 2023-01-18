# srun -p GVT -n1 -c10 --gres=gpu:3 -x SH-IDC1-10-198-8-[61,62] --async \
# -J hi bash run/cma_rxr.bash hi data/logs/checkpoints/cma_rxr_hi_slide1_oldcamera/ckpt.5.pth
# srun -p GVT -n1 -c10 --gres=gpu:3 -x SH-IDC1-10-198-8-[61,62] --async \
# -J te bash run/cma_rxr.bash te data/logs/checkpoints/cma_rxr_te_slide1_oldcamera/ckpt.4.pth

# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] --async \
# -J hi_eval bash run/cma_rxr_eval.bash hi
# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] --async \
# -J te_eval bash run/cma_rxr_eval.bash te

# srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] --async \
# -J en_s1>0 bash run/cma_rxr_eval.bash

srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] --async \
-J en_infer bash run/cma_rxr_eval.bash en data/logs/checkpoints/cma_rxr_en_slide1_oldcamera/ckpt.25.pth
srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] --async \
-J hi_infer bash run/cma_rxr_eval.bash hi data/logs/checkpoints/cma_rxr_hi_slide1_oldcamera/ckpt.8.pth
srun -p GVT -n1 -c5 --gres=gpu:1 -x SH-IDC1-10-198-8-[61,62] --async \
-J te_infer bash run/cma_rxr_eval.bash te data/logs/checkpoints/cma_rxr_te_slide1_oldcamera/ckpt.9.pth