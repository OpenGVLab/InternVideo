# Model Zoo

## Note

- For all the pretraining and finetuning, we adopt spaese/uniform sampling.
- `#Frame` $=$ `#input_frame` $\times$ `#crop` $\times$ `#clip`
- `#input_frame` means how many frames are input for model per inference
- `#crop` means spatial crops (e.g., 3 for left/right/center)
- `#clip` means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

## Pretraining

| Model    | Setting     | Model  | Shell  |
| -------- | ----------- | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash-1.1M 300e   |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8/blob/main/pretrain.pth) | [run.sh](./scripts/pretraining/1B_pt.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash-2M 300e   |  TBD | [run.sh](./scripts/pretraining/6B_pt.sh) |

## Distillation

| Model    | Setting     | Teacher | Model  | Shell  |
| -------- | ----------- | ------- | ------ | ------ |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash-1.1M 100e | $\text{InternVideo2}_{s2}$-1B |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_dist_1B_stage2/pytorch_model.bin) | [run.sh](./scripts/distillation/B14_dist_1B_stage2.sh) |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash-1.1M 100e | $\text{InternVideo2}_{s2}$-1B |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_dist_1B_stage2/pytorch_model.bin) | [run.sh](./scripts/distillation/S14_dist_1B_stage2.sh) |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash-1.1M 100e | $\text{InternVideo2}_{s2}$-1B |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_dist_1B_stage2/pytorch_model.bin) | [run.sh](./scripts/distillation/L14_dist_1B_stage2.sh) |


## Finetuning

### K710

| Model    | Setting  | #Frame   | Top-1  | Model  | Shell  |
| -------- | -------  | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT  | 8x3x4    | 87.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-k710/blob/main/1B_ft_k710_f8.pth) | [run.sh](./scripts/pretraining/1B_pt.sh) | [run.sh](./scripts/finetuning/full_tuning/k710/1B_ft_k710_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT  | 8x3x4    | 88.1 | TBD | [run.sh](./scripts/finetuning/full_tuning/k710/6B_ft_k710_f8.sh) |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash PT  | 8x3x4    | 79.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_ft_k710_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k710/S14_ft_k710_f8.sh) |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash PT  | 8x3x4    | 83.5 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_ft_k710_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k710/B14_ft_k710_f8.sh) |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash PT  | 8x3x4    | 86.2 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_dist_1B_stage2/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k710/L14_ft_k710_f8.sh) |


### K400

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 91.3 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-K400/blob/main/1B_ft_k710_ft_k400_f8.pth) | [run.sh](./scripts/finetuning/full_tuning/k400/1B_ft_k710_ft_k400_f8.sh) |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 91.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-K400/blob/main/1B_ft_k710_ft_k400_f16.pth) | [run.sh](./scripts/finetuning/full_tuning/k400/1B_ft_k710_ft_k400_f16.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 91.9 | TBD | [run.sh](./scripts/finetuning/full_tuning/k400/6B_ft_k710_ft_k400_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 92.1 | TBD | [run.sh](./scripts/finetuning/full_tuning/k400/6B_ft_k710_ft_k400_f16.sh) |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash PT + K710 FT  | 8x3x4    | 85.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_ft_k710_ft_k400_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k400/S14_ft_k710_ft_k400_f8.sh) |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash PT + K710 FT  | 8x3x4    | 88.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_ft_k710_ft_k400_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k400/B14_ft_k710_ft_k400_f8.sh) |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash PT + K710 FT  | 8x3x4    | 90.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k400/L14_ft_k710_ft_k400_f8.sh) |


### K600

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 91.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-K400/blob/main/1B_ft_k710_ft_k400_f8.pth) | [run.sh](./scripts/finetuning/full_tuning/k600/1B_ft_k710_ft_k600_f8.sh) |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 91.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-K600/blob/main/1B_ft_k710_ft_k600_f16.pth) | [run.sh](./scripts/finetuning/full_tuning/k600/1B_ft_k710_ft_k600_f16.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 91.7 | TBD | [run.sh](./scripts/finetuning/full_tuning/k600/6B_ft_k710_ft_k600_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 91.9 | TBD | [run.sh](./scripts/finetuning/full_tuning/k600/6B_ft_k710_ft_k600_f16.sh) |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash PT + K710 FT  | 8x3x4    | 86.0 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_ft_k710_ft_k600_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k600/S14_ft_k710_ft_k600_f8.sh) |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash PT + K710 FT  | 8x3x4    | 88.9 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_ft_k710_ft_k600_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k600/B14_ft_k710_ft_k600_f8.sh) |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash PT + K710 FT  | 8x3x4    | 90.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_ft_k710_ft_k600_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k600/L14_ft_k710_ft_k600_f8.sh) |


### K700

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 85.0 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-K700/blob/main/1B_ft_k710_ft_k700_f8.pth) | [run.sh](./scripts/finetuning/full_tuning/k700/1B_ft_k710_ft_k700_f8.sh) |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 85.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-K700/blob/main/1B_ft_k710_ft_k700_f16.pth) | [run.sh](./scripts/finetuning/full_tuning/k700/1B_ft_k710_ft_k700_f16.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 85.7 | TBD | [run.sh](./scripts/finetuning/full_tuning/k700/6B_ft_k710_ft_k700_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 85.9 | TBD | [run.sh](./scripts/finetuning/full_tuning/k700/6B_ft_k710_ft_k700_f16.sh) |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash PT + K710 FT  | 8x3x4    | 75.7 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_ft_k710_ft_k700_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k700/S14_ft_k710_ft_k700_f8.sh) |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash PT + K710 FT  | 8x3x4    | 80.5 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_ft_k710_ft_k700_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k700/B14_ft_k710_ft_k700_f8.sh) |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash PT + K710 FT  | 8x3x4    | 83.5 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_ft_k710_ft_k700_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/k700/L14_ft_k710_ft_k700_f8.sh) |


### MiT V1

| Model         | Setting              | #Frame   | Top-1  | Model  | Shell  |
| ------------- | -------------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 50.8 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-MiT) | [run.sh](./scripts/finetuning/full_tuning/mit/1B_ft_k710_ft_k400_ft_mit_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 51.0 | TBD | [run.sh](./scripts/finetuning/full_tuning/mit/6B_ft_k710_ft_k400_ft_mit_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B 336↑ | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 51.2 | TBD | [run.sh](./scripts/finetuning/full_tuning/mit/6B_ft_k710_ft_k400_ft_mit_f8_res224to336.sh) |


### SthSth V1

| Model    | Setting     | #Frame   | Top-1  | Model  | Shell  |
| -------- | ----------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT  | 8x3x4    | 68.5 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-SthSth/blob/main/1B_ft_ssv1_f8.pth) | [run.sh](./scripts/finetuning/full_tuning/ssv1/1B_ft_ssv1_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT  | 8x3x4    | 69.7 | TBD | [run.sh](./scripts/finetuning/full_tuning/ssv1/6B_ft_ssv1_f8.sh) |


### SthSth V2

| Model    | Setting     | #Frame   | Top-1  | Model  | Shell  |
| -------- | ----------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT  | 8x3x4    | 77.1 |  [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-f8-SthSth/blob/main/1B_ft_ssv2_f8.pth)  | [run.sh](./scripts/finetuning/full_tuning/ssv2/1B_ft_ssv1_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT  | 8x3x4    | 77.5 | TBD | [run.sh](./scripts/finetuning/full_tuning/ssv2/6B_ft_ssv2_f8.sh) |
| $\text{InternVideo2}_{dist}$-S/14 | K-Mash PT  | 8x3x4    | 71.6 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/S14/S14_ft_ssv2_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/ssv2/S14_ft_ssv2_f8.sh) |
| $\text{InternVideo2}_{dist}$-B/14 | K-Mash PT  | 8x3x4    | 73.5 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/B14/B14_ft_k710_ft_k700_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/ssv2/B14_ft_ssv2_f8.sh) |
| $\text{InternVideo2}_{dist}$-L/14 | K-Mash PT  | 8x3x4    | 76.4 | [:hugs: HF link](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models/resolve/main/stage1/L14/L14_ft_ssv2_f8/pytorch_model.bin) | [run.sh](./scripts/finetuning/full_tuning/ssv2/L14_ft_ssv2_f8.sh) |



### ANet

| Model         | Setting              | #Frame   | Top-1  | mAP  | Model  | Shell  |
| ------------- | -------------------- | -------- | ------ |  ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 95.9 | 98.2 | TBD | [run.sh](./scripts/finetuning/full_tuning/anet/6B_ft_k710_ft_k400_ap_anet_f8.sh) |


### HACS

| Model         | Setting              | #Frame   | Top-1  |  mAP  | Model  | Shell  |
| ------------- | -------------------- | -------- | ------ | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 97.0 | 98.8 | TBD | [run.sh](./scripts/finetuning/full_tuning/hacs/6B_ft_k710_ft_k400_ap_hacs_f8.sh) |
