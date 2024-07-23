# Model Zoo


## Pretraining 
For $\text{InternVideo2}_{s2}$, we load those models of $\text{InternVideo2}_{s1}$ and further pretrain them on multi-modality datasets.

For $\text{InternVideo2}_{clip}$, we load those models of $\text{InternVideo2}_{s2}$.


| Model    | Setting     | Model  | Pretraining Script  |
| -------- | ----------- | ------ | ------------- |
| $\text{InternVideo2}_{s2}$-1B | IV-25.5M          | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) | [script](scripts/pretraining/stage2/1B/run.sh)  |
| $\text{InternVideo2}_{clip}$-1B | IV-25.5M        |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-CLIP-1B-224p-f8) | [script](scripts/pretraining/clip/1B/run.sh)  |
| $\text{InternVideo2}_{s2}$-6B | IV-400M         |  TBD | [script](scripts/pretraining/stage2/6B/run.sh) |
| $\text{InternVideo2}_{clip}$-6B | IV-400M         |  TBD | [script](scripts/pretraining/clip/6B/run.sh) |


### Zero-shot Evaluation

## Zero-Shot Video-Text Retrieval

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


## Zero-Shot Action Recognition

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

| Model    | Dataset     |  mAP  | Script  |
| -------- | ----------- | ------ | ------- |
| $\text{InternVideo2}_{clip}$-1B | Charades | 32.9 | [script](scripts/evaluation/clip/zero_shot/1B/eval_charades_mc.sh) |
| $\text{InternVideo2}_{clip}$-6B | Charades | 34.6 | [script](scripts/evaluation/clip/zero_shot/6B/eval_charades_mc.sh) |