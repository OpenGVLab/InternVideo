# Model Zoo


## Pretraining 
For $\text{InternVideo2}\_{s2}$, we load those models of $\text{InternVideo2}_{s1}$ and further pretrain them on multi-modality datasets.

For $\text{InternVideo2}\_{clip}$, we load those models of $\text{InternVideo2}_{s2}$.


| Model    | Setting     | Model  | Pretraining Script  |
| -------- | ----------- | ------ | ------------- |
| $\text{InternVideo2}_{s2}$-1B | IV-25.5M          | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) | [script](scripts/pretraining/stage2/1B/run.sh)  |
| $\text{InternVideo2}_{clip}$-1B | IV-25.5M        |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-CLIP-1B-224p-f8) | [script](scripts/pretraining/clip/1B/run.sh)  |
| $\text{InternVideo2}_{s2}$-6B | IV-400M         |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_6B-224p-f4) | [script](scripts/pretraining/stage2/6B/run.sh) |
| $\text{InternVideo2}_{clip}$-6B | IV-400M         |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-CLIP-6B-224p-f8) | [script](scripts/pretraining/clip/6B/run.sh) |
| $\text{InternVideo2}_{s2}$-S14 | IV-25.5M Distillation |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/blob/main/stage1/S14/S14_dist_1B_stage2/pytorch_model.bin) | - |
| $\text{InternVideo2}_{s2}$-B14 | IV-25.5M Distillation |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/blob/main/stage1/B14/B14_dist_1B_stage2/pytorch_model.bin) | - |
| $\text{InternVideo2}_{s2}$-L14 | IV-25.5M Distillation |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/blob/main/stage1/L14/L14_dist_1B_stage2/pytorch_model.bin) | - |
| $\text{InternVideo2}_{clip}$-S14 | IV-25.5M Distillation |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/clip/S14/pytorch_model.bin) | [script](scripts/pretraining/clip/S14/run.sh) |
| $\text{InternVideo2}_{clip}$-B14 | IV-25.5M Distillation |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/clip/B14/pytorch_model.bin) | [script](scripts/pretraining/clip/B14/run.sh) |
| $\text{InternVideo2}_{clip}$-L14 | IV-25.5M Distillation |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/clip/L14/pytorch_model.bin) | [script](scripts/pretraining/clip/L14/run.sh) |


## Zero-shot Evaluation

### Zero-Shot Video-Text Retrieval

| Model    | Dataset     |  T2V  | V2T  | Evaluation Script  |
| -------- | ----------- | ------ | ------- | ------- |
| $\text{InternVideo2}_{s2}$-1B | MSRVTT | 51.9 | 50.9 | [script](scripts/evaluation/stage2/zero_shot/1B/eval_msrvtt.sh) |
|                               | LSMDC  | 32.0 | 27.3 | [script](scripts/evaluation/stage2/zero_shot/1B/eval_lsmdc.sh) |
|                               | DiDeMo | 57.0 | 54.3 | [script](scripts/evaluation/stage2/zero_shot/1B/eval_didemo.sh) |
|                               | MSVD   | 58.1 | 83.3 | [script](scripts/evaluation/stage2/zero_shot/1B/eval_msvd.sh) |
|                               | ANet   | 60.4 | 54.8 | [script](scripts/evaluation/stage2/zero_shot/1B/eval_anet.sh) |
|                               | VATEX  | 70.4 | 85.4 | [script](scripts/evaluation/stage2/zero_shot/1B/eval_vatex.sh) |
| $\text{InternVideo2}_{s2}$-6B | MSRVTT | 55.9 | 53.7 | TBD |
|                               | LSMDC  | 33.8 | 30.1 | TBD |
|                               | DiDeMo | 57.9 | 57.1 | TBD |
|                               | MSVD   | 59.3 | 83.1 | TBD |
|                               | ANet   | 63.2 | 56.5 | TBD |
|                               | VATEX  | 71.5 | 85.3 | TBD |


| Model    | Dataset     |  T2V  | V2T  | Evaluation Script  |
| -------- | ----------- | ------ | ------- | ------- |
| $\text{InternVideo2}_{clip}$-1B | MSRVTT | 50.0 | 48.4 | [script](scripts/evaluation/clip/zero_shot/1B/eval_msrvtt.sh) |
|                               | LSMDC  | 26.4 | 23.1 | [script](scripts/evaluation/clip/zero_shot/1B/eval_lsmdc.sh) |
|                               | DiDeMo | 47.8 | 46.4 | [script](scripts/evaluation/clip/zero_shot/1B/eval_didemo.sh) |
|                               | ANet   | 49.4 | 46.2 | [script](scripts/evaluation/clip/zero_shot/1B/eval_anet.sh) |
|                               | VATEX_en  | 63.5 | 81.2 | [script](scripts/evaluation/clip/zero_shot/1B/eval_vatex_en.sh) |
|                               | VATEX_ch  | 54.9 | 76.4 | [script](scripts/evaluation/clip/zero_shot/1B/eval_vatex_ch.sh) |
| $\text{InternVideo2}_{clip}$-6B | MSRVTT | 50.9 | 50.6 | [script](scripts/evaluation/clip/zero_shot/6B/eval_msrvtt.sh) |
|                               | LSMDC  | 29.4 | 26.3 | [script](scripts/evaluation/clip/zero_shot/6B/eval_lsmdc.sh) |
|                               | DiDeMo | 50.5 | 46.8| [script](scripts/evaluation/clip/zero_shot/6B/eval_didemo.sh) |
|                               | ANet   | 50.2 | 47.5 | [script](scripts/evaluation/clip/zero_shot/6B/eval_anet.sh) |
|                               | VATEX_en  | 64.1 | 82.6 | [script](scripts/evaluation/clip/zero_shot/6B/eval_vatex_en.sh) |
|                               | VATEX_ch  | 54.6 | 76.9 | [script](scripts/evaluation/clip/zero_shot/6B/eval_vatex_ch.sh) |
| $\text{InternVideo2}_{clip}$-S14 | MSRVTT | 35.6 | 35.9 | [script](scripts/evaluation/clip/zero_shot/S14/eval_msrvtt.sh) |
|                               | LSMDC  | 14.7 | 12.8 | [script](scripts/evaluation/clip/zero_shot/S14/eval_lsmdc.sh) |
|                               | DiDeMo | 33.7 | 35.5 | [script](scripts/evaluation/clip/zero_shot/S14/eval_didemo.sh) |
|                               | ANet   | 34.5 | 23.6 | [script](scripts/evaluation/clip/zero_shot/S14/eval_anet.sh) |
|                               | VATEX_en  | 49.9 | 69.1 | [script](scripts/evaluation/clip/zero_shot/S14/eval_vatex_en.sh) |
|                               | VATEX_ch  | 1.9 | 7.6 | [script](scripts/evaluation/clip/zero_shot/S14/eval_vatex_ch.sh) |
| $\text{InternVideo2}_{clip}$-B14 | MSRVTT | 40.3 | 48.5 | [script](scripts/evaluation/clip/zero_shot/B14/eval_msrvtt.sh) |
|                               | LSMDC  | 18.7 | 16.5 | [script](scripts/evaluation/clip/zero_shot/B14/eval_lsmdc.sh) |
|                               | DiDeMo | 40.3 | 39.1 | [script](scripts/evaluation/clip/zero_shot/B14/eval_didemo.sh) |
|                               | ANet   | 41.5 | 38.8 | [script](scripts/evaluation/clip/zero_shot/B14/eval_anet.sh) |
|                               | VATEX_en  | 56.8 | 74.5 | [script](scripts/evaluation/clip/zero_shot/B14/eval_vatex_en.sh) |
|                               | VATEX_ch  | 1.8 | 8.8 | [script](scripts/evaluation/clip/zero_shot/B14/eval_vatex_ch.sh) |
| $\text{InternVideo2}_{clip}$-L14 | MSRVTT | 42.1 | 44.1 | [script](scripts/evaluation/clip/zero_shot/L14/eval_msrvtt.sh) |
|                               | LSMDC  | 21.4 | 18.9 | [script](scripts/evaluation/clip/zero_shot/L14/eval_lsmdc.sh) |
|                               | DiDeMo | 42.8 | 43.2 | [script](scripts/evaluation/clip/zero_shot/L14/eval_didemo.sh) |
|                               | ANet   | 43.6 | 40.7 | [script](scripts/evaluation/clip/zero_shot/L14/eval_anet.sh) |
|                               | VATEX_en  | 59.6 | 75.5 | [script](scripts/evaluation/clip/zero_shot/L14/eval_vatex_en.sh) |
|                               | VATEX_ch  | 1.6 | 9.8 | [script](scripts/evaluation/clip/zero_shot/L14/eval_vatex_ch.sh) |


### Zero-Shot Action Recognition

| Model    | Dataset     |  top-1  | AVG  | Script  |
| -------- | ----------- | ------ | ------- | ------- |
| $\text{InternVideo2}_{clip}$-1B | K400 | 73.1 | 82.4 | [script](scripts/evaluation/clip/zero_shot/1B/eval_k400.sh) |
|                                 | K600  | 72.8 | 81.8 | [script](scripts/evaluation/clip/zero_shot/1B/eval_k600.sh) |
|                                 | K700 | 64.9 | 75.2 | [script](scripts/evaluation/clip/zero_shot/1B/eval_k700.sh) |
|                                 | UCF101 | 88.8 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_ucf101.sh) |
|                                 | HMDB51 | 53.9 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_hmdb51.sh) |
|                                 | MiT | 31.6 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_mit.sh) |
|                                 | SSv2-MC | 61.5 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_ssv2_mc.sh) |
| $\text{InternVideo2}_{clip}$-6B | K400 | 72.7 | 82.2 | [script](scripts/evaluation/clip/zero_shot/1B/eval_k400.sh) |
|                                 | K600  | 71.7 | 81.2 | [script](scripts/evaluation/clip/zero_shot/1B/eval_k600.sh) |
|                                 | K700 | 64.2 | 75.2 | [script](scripts/evaluation/clip/zero_shot/1B/eval_k700.sh) |
|                                 | UCF101 | 89.5 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_ucf101.sh) |
|                                 | HMDB51 | 56.7 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_hmdb51.sh) |
|                                 | MiT | 32.9 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_mit.sh) |
|                                 | SSv2-MC | 63.5 | - | [script](scripts/evaluation/clip/zero_shot/1B/eval_ssv2_mc.sh) |
| $\text{InternVideo2}_{clip}$-S14 | K400 | 62.1 | 73.6 | [script](scripts/evaluation/clip/zero_shot/S14/eval_k400.sh) |
|                                 | K600  | 61.6 | 72.5 | [script](scripts/evaluation/clip/zero_shot/S14/eval_k600.sh) |
|                                 | K700 | 51.4 | 63.4 | [script](scripts/evaluation/clip/zero_shot/S14/eval_k700.sh) |
|                                 | UCF101 | 79.1 | - | [script](scripts/evaluation/clip/zero_shot/S14/eval_ucf101.sh) |
|                                 | HMDB51 | 49.2 | - | [script](scripts/evaluation/clip/zero_shot/S14/eval_hmdb51.sh) |
|                                 | MiT | 24.1 | - | [script](scripts/evaluation/clip/zero_shot/S14/eval_mit.sh) |
|                                 | SSv2-MC | 46.4 | - | [script](scripts/evaluation/clip/zero_shot/S14/eval_ssv2_mc.sh) |
| $\text{InternVideo2}_{clip}$-B14 | K400 | 67.7 | 78.0 | [script](scripts/evaluation/clip/zero_shot/B14/eval_k400.sh) |
|                                 | K600  | 66.8 | 77.0 | [script](scripts/evaluation/clip/zero_shot/B14/eval_k600.sh) |
|                                 | K700 | 57.9 | 69.3 | [script](scripts/evaluation/clip/zero_shot/B14/eval_k700.sh) |
|                                 | UCF101 | 83.4 | - | [script](scripts/evaluation/clip/zero_shot/B14/eval_ucf101.sh) |
|                                 | HMDB51 | 52.5 | - | [script](scripts/evaluation/clip/zero_shot/B14/eval_hmdb51.sh) |
|                                 | MiT | 27.9 | - | [script](scripts/evaluation/clip/zero_shot/B14/eval_mit.sh) |
|                                 | SSv2-MC | 55.9 | - | [script](scripts/evaluation/clip/zero_shot/B14/eval_ssv2_mc.sh) |
| $\text{InternVideo2}_{clip}$-L14 | K400 | 70.7 | 80.5 | [script](scripts/evaluation/clip/zero_shot/L14/eval_k400.sh) |
|                                 | K600  | 69.9 | 79.6 | [script](scripts/evaluation/clip/zero_shot/L14/eval_k600.sh) |
|                                 | K700 | 61.9 | 72.9 | [script](scripts/evaluation/clip/zero_shot/L14/eval_k700.sh) |
|                                 | UCF101 | 85.9 | - | [script](scripts/evaluation/clip/zero_shot/L14/eval_ucf101.sh) |
|                                 | HMDB51 | 53.2 | - | [script](scripts/evaluation/clip/zero_shot/L14/eval_hmdb51.sh) |
|                                 | MiT | 30.6 | - | [script](scripts/evaluation/clip/zero_shot/L14/eval_mit.sh) |
|                                 | SSv2-MC | 59.6 | - | [script](scripts/evaluation/clip/zero_shot/L14/eval_ssv2_mc.sh) |

| Model    | Dataset     |  mAP  | Script  |
| -------- | ----------- | ------ | ------- |
| $\text{InternVideo2}_{clip}$-1B | Charades | 32.9 | [script](scripts/evaluation/clip/zero_shot/1B/eval_charades_mc.sh) |
| $\text{InternVideo2}_{clip}$-6B | Charades | 34.6 | [script](scripts/evaluation/clip/zero_shot/6B/eval_charades_mc.sh) |
| $\text{InternVideo2}_{clip}$-S14 | Charades | 21.7 | [script](scripts/evaluation/clip/zero_shot/S14/eval_charades_mc.sh) |
| $\text{InternVideo2}_{clip}$-B14 | Charades | 26.1 | [script](scripts/evaluation/clip/zero_shot/B14/eval_charades_mc.sh) |
| $\text{InternVideo2}_{clip}$-L14 | Charades | 30.1 | [script](scripts/evaluation/clip/zero_shot/L14/eval_charades_mc.sh) |
