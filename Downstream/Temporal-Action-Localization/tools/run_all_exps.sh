#!/bin/bash

# THUMOS14 with I3D features
python ./train.py ./configs/thumos_i3d.yaml --output final 2>&1 | tee ./ckpt/thumos_log.txt
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_final/ 2>&1 | tee ./ckpt/thumos_results.txt
sleep 5

# EPIC-Kitchens verb with slowfast features
python ./train.py ./configs/epic_slowfast_verb.yaml --output final 2>&1 | tee ./ckpt/epic_verb_log.txt
python ./eval.py ./configs/epic_slowfast_verb.yaml ./ckpt/epic_slowfast_verb_final/ 2>&1 | tee ./ckpt/epic_verb_results.txt
sleep 5

# EPIC-Kitchens noun with slowfast features
python ./train.py ./configs/epic_slowfast_noun.yaml --output final 2>&1 | tee ./ckpt/epic_noun_log.txt
python ./eval.py ./configs/epic_slowfast_noun.yaml ./ckpt/epic_slowfast_noun_final/ 2>&1 | tee ./ckpt/epic_noun_results.txt
sleep 5

# ActivityNet 1.3 with TSP features (r(2+1d)-34)
python ./train.py ./configs/anet_tsp.yaml --output final 2>&1 | tee ./ckpt/anet_tsp_log.txt
python ./eval.py ./configs/anet_tsp.yaml ./ckpt/anet_tsp_final/ 2>&1 | tee ./ckpt/anet_tsp_results.txt
