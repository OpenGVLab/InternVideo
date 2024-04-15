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
| $\text{InternVideo2}_{s1}$-1B | K-Mash-1.1M 300e   |  TBD | [run.sh](./scripts/pretraining/1B_pt.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash-2M 300e   |  TBD | [run.sh](./scripts/pretraining/6B_pt.sh) |


## Finetuning

### K710

| Model    | Setting  | #Frame   | Top-1  | Model  | Shell  |
| -------- | -------  | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT  | 8x3x4    | 87.6 | TBD | [run.sh](./scripts/finetuning/full_tuning/k710/1B_ft_k710_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT  | 8x3x4    | 88.1 | TBD | [run.sh](./scripts/finetuning/full_tuning/k710/6B_ft_k710_f8.sh) |


### K400

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 91.3 | TBD | [run.sh](./scripts/finetuning/full_tuning/k400/1B_ft_k710_ft_k400_f8.sh) |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 91.6 | TBD | [run.sh](./scripts/finetuning/full_tuning/k400/1B_ft_k710_ft_k400_f16.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 91.9 | TBD | [run.sh](./scripts/finetuning/full_tuning/k400/6B_ft_k710_ft_k400_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 92.1 | TBD | [run.sh](./scripts/finetuning/full_tuning/k400/6B_ft_k710_ft_k400_f16.sh) |


### K600

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 91.4 | TBD | [run.sh](./scripts/finetuning/full_tuning/k600/1B_ft_k710_ft_k600_f8.sh) |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 91.6 | TBD | [run.sh](./scripts/finetuning/full_tuning/k600/1B_ft_k710_ft_k600_f16.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 91.7 | TBD | [run.sh](./scripts/finetuning/full_tuning/k600/6B_ft_k710_ft_k600_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 91.9 | TBD | [run.sh](./scripts/finetuning/full_tuning/k600/6B_ft_k710_ft_k600_f16.sh) |



### K700

| Model    | Setting       | #Frame   | Top-1  | Model  | Shell  |
| -------- | ------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 8x3x4    | 85.0 | TBD | [run.sh](./scripts/finetuning/full_tuning/k700/1B_ft_k710_ft_k700_f8.sh) |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT  | 16x3x4    | 85.4 | TBD | [run.sh](./scripts/finetuning/full_tuning/k700/1B_ft_k710_ft_k700_f16.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 8x3x4    | 85.7 | TBD | [run.sh](./scripts/finetuning/full_tuning/k700/6B_ft_k710_ft_k700_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT  | 16x3x4    | 85.9 | TBD | [run.sh](./scripts/finetuning/full_tuning/k700/6B_ft_k710_ft_k700_f16.sh) |


### MiT V1

| Model         | Setting              | #Frame   | Top-1  | Model  | Shell  |
| ------------- | -------------------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 50.8 | TBD | [run.sh](./scripts/finetuning/full_tuning/mit/1B_ft_k710_ft_k400_ft_mit_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 51.0 | TBD | [run.sh](./scripts/finetuning/full_tuning/mit/6B_ft_k710_ft_k400_ft_mit_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B 336↑ | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 51.2 | TBD | [run.sh](./scripts/finetuning/full_tuning/mit/6B_ft_k710_ft_k400_ft_mit_f8_res224to336.sh) |


### SthSth V1

| Model    | Setting     | #Frame   | Top-1  | Model  | Shell  |
| -------- | ----------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT  | 8x3x4    | 68.5 | TBD | [run.sh](./scripts/finetuning/full_tuning/ssv1/1B_ft_ssv1_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT  | 8x3x4    | 69.7 | TBD | [run.sh](./scripts/finetuning/full_tuning/ssv1/6B_ft_ssv1_f8.sh) |


### SthSth V2

| Model    | Setting     | #Frame   | Top-1  | Model  | Shell  |
| -------- | ----------- | -------- | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-1B | K-Mash PT  | 8x3x4    | 77.1 | TBD | [run.sh](./scripts/finetuning/full_tuning/ssv1/1B_ft_ssv1_f8.sh) |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT  | 8x3x4    | 77.5 | TBD | [run.sh](./scripts/finetuning/full_tuning/ssv1/6B_ft_ssv1_f8.sh) |



### ANet

| Model         | Setting              | #Frame   | Top-1  | mAP  | Model  | Shell  |
| ------------- | -------------------- | -------- | ------ |  ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 95.9 | 98.2 | TBD | [run.sh](./scripts/finetuning/full_tuning/anet/6B_ft_k710_ft_k400_ap_anet_f8.sh) |


### HACS

| Model         | Setting              | #Frame   | Top-1  |  mAP  | Model  | Shell  |
| ------------- | -------------------- | -------- | ------ | ------ | ------ | ------ |
| $\text{InternVideo2}_{s1}$-6B | K-Mash PT + K710 FT + K400 FT  | 8x3x4    | 97.0 | 98.8 | TBD | [run.sh](./scripts/finetuning/full_tuning/hacs/6B_ft_k710_ft_k400_ap_hacs_f8.sh) |