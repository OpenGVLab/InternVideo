# Model Zoo


## Pretraining of $\text{InternVideo2}_{s2}$

We load those models of $\text{InternVideo2}_{s1}$ and further pretrain them on multimodality data.


| Model    | Setting     | Model  | Pretraining Script  |
| -------- | ----------- | ------ | ------------- |
| $\text{InternVideo2}_{s2}$-1B | IV-25.5M          |  🤗\[[HF link](https://huggingface.co/OpenGVLab/InternVideo2/blob/main/InternVideo2-stage2_1b-224p-f4.pt)\] | [script](scripts/pretraining/stage2/1B/run.sh)  |
| $\text{InternVideo2}_{clip}$-1B | IV-25.5M          |  TBD | TBD  |
| $\text{InternVideo2}_{s2}$-6B | IV-400M         |  TBD | [script](scripts/pretraining/stage2/6B/run.sh) |


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

## Zero-Shot Action Recognition

| Model    | Dataset     |  top-1  | AVG  | Script  |
| -------- | ----------- | ------ | ------- | ------- |
| $\text{InternVideo2}_{clip}$-1B | K400 | 73.1 | 82.4 | TBD |
|                                 | K600  | 72.8 | 81.8 | TBD |
|                                 | K700 | 64.9 | 75.2 | TBD |