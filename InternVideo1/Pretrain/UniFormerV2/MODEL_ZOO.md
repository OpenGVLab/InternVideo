# Model Zoo

## Note

- All the `config.yaml` in our `exp` are **NOT** the training config actually used, since some hyperparameters are **changed** in the `run.sh` or `test.sh`.
-  \#Frame = \#input_frame x \#crop x \#clip
  - \#input_frame means how many frames are input for model per inference
  - \#crop means spatial crops (e.g., 3 for left/right/center)
  - \#clip means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

## K710

| Model                       | Frame | Model    | Shell      | Config          |
| --------------------------- | ------ | ----- | -------- | ---------- |
| UniFormerV2-B/16            | 8     | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k710/k710_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k710/k710_b16_f8x224/run.sh) | [config.yaml](./exp/k710/k710_b16_f8x224/config.yaml) |
| UniFormerV2-L/14            | 8     | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k710/k710_uniformerv2_l14_8x224.pyth) | [run.sh](./exp/k710/k710_l14_f8x224/run.sh) | [config.yaml](./exp/k710/k710_l14_f8x224/config.yaml) |
| UniFormerV2-L/14@336        | 8     | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k710/k710_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k710/k710_l14_f8x336/run.sh) | [config.yaml](./exp/k710/k710_l14_f8x336/config.yaml) |
| **Frozen** UniFormerV2-L/14@336 | 8     | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k710/frozen_k710_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k710/frozen_k710_l14_f8x336/run.sh) | [config.yaml](./exp/k710/frozen_k710_l14_f8x336/config.yaml) |

## K400

| Model                | Pretraining    | #Frame | Top-1 | Model                                                        | Shell                                             | Config                                                      |
| -------------------- | -------------- | ------ | ----- | ------------------------------------------------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| UniFormerV2-B/16     | CLIP-400M      | 8x3x4  | 84.4  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k400/k400_b16_f8x224/run.sh)       | [config.yaml](./exp/k400/k400_b16_f8x224/config.yaml)       |
| UniFormerV2-B/16     | CLIP-400M+K710 | 8x3x4  | 85.6  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k400/k400+k710_b16_f8x224/run.sh)  | [config.yaml](./exp/k400/k400+k710_b16_f8x224/config.yaml)  |
| UniFormerV2-L/14     | CLIP-400M+K710 | 8x3x4  | 88.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_l14_8x224.pyth) | [run.sh](./exp/k400/k400+k710_l14_f8x224/run.sh)  | [config.yaml](./exp/k400/k400+k710_l14_f8x224/config.yaml)  |
| UniFormerV2-L/14     | CLIP-400M+K710 | 16x3x4 | 89.1  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/k400/k400+k710_l14_f16x224/run.sh) | [config.yaml](./exp/k400/k400+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 32x3x2 | 89.3  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/k400/k400+k710_l14_f32x224/run.sh) | [config.yaml](./exp/k400/k400+k710_l14_f32x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x2 | 89.7  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_l14_32x336.pyth) | [run.sh](./exp/k400/k400+k710_l14_f32x336/run.sh) | [config.yaml](./exp/k400/k400+k710_l14_f32x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 64x3x2 | 90.0  | -                                                            | [run.sh](./exp/k400/k400+k710_l14_f64x336/run.sh) | [config.yaml](./exp/k400/k400+k710_l14_f64x336/config.yaml) |

| Frozen Model         | Pretraining    | #Frame | Top-1 | Model                                                        | Shell                                                    | Config                                                       |
| -------------------- | -------------- | ------ | ----- | ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| UniFormerV2-L/14@336 | CLIP-400M      | 8x1x3  | 86.7  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/frozen_k400_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k400/frozen_k400_l14_f8x336/run.sh)       | [config.yaml](./exp/k400/frozen_k400_l14_f8x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 8x1x3  | 87.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/frozen_k400_k710_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k400/frozen_k400+k710_l14_f8x336/run.sh)  | [config.yaml](./exp/k400/frozen_k400+k710_l14_f8x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x4 | 88.9  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/frozen_k400_k710_uniformerv2_l14_32x336.pyth) | [run.sh](./exp/k400/frozen_k400+k710_l14_f32x336/run.sh) | [config.yaml](./exp/k400/frozen_k400+k710_l14_f32x336/config.yaml) |

## K600

| Model                | Pretraining    | #Frame | Top-1 | Model                                                        | Shell                                             | Config                                                      |
| -------------------- | -------------- | ------ | ----- | ------------------------------------------------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| UniFormerV2-B/16     | CLIP-400M      | 8x3x4  | 85.0  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k600/k600_b16_f8x224/run.sh)       | [config.yaml](./exp/k600/k600_b16_f8x224/config.yaml)       |
| UniFormerV2-B/16     | CLIP-400M+K710 | 8x3x4  | 86.1  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k600/k600+k710_b16_f8x224/run.sh)  | [config.yaml](./exp/k600/k600+k710_b16_f8x224/config.yaml)  |
| UniFormerV2-L/14     | CLIP-400M+K710 | 8x3x4  | 89.0  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_l14_8x224.pyth) | [run.sh](./exp/k600/k600+k710_l14_f8x224/run.sh)  | [config.yaml](./exp/k600/k600+k710_l14_f8x224/config.yaml)  |
| UniFormerV2-L/14     | CLIP-400M+K710 | 16x3x4 | 89.4  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/k600/k600+k710_l14_f16x224/run.sh) | [config.yaml](./exp/k600/k600+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 32x3x2 | 89.5  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/k600/k600+k710_l14_f16x224/run.sh) | [config.yaml](./exp/k600/k600+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x2 | 89.9  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/k600_k710_uniformerv2_l14_32x336.pyth) | [run.sh](./exp/k600/k600+k710_l14_f32x336/run.sh) | [config.yaml](./exp/k600/k600+k710_l14_f32x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 64x3x2 | 90.1  | -                                                            | [run.sh](./exp/k600/k600+k710_l14_f64x336/run.sh) | [config.yaml](./exp/k600/k600+k710_l14_f64x336/config.yaml) |

| Frozen Model         | Pretraining    | #Frame | Top-1 | Model                                                        | Shell                                                    | Config                                                       |
| -------------------- | -------------- | ------ | ----- | ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| UniFormerV2-L/14@336 | CLIP-400M      | 8x1x3  | 87.4  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/frozen_k600_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k600/frozen_k600_l14_f8x336/run.sh)       | [config.yaml](./exp/k600/frozen_k600_l14_f8x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 8x1x3  | 88.2  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/frozen_k600_k710_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k600/frozen_k600+k710_l14_f8x336/run.sh)  | [config.yaml](./exp/k600/frozen_k600+k710_l14_f8x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x4 | 89.2  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k600/frozen_k600_k710_uniformerv2_l14_32x336.pyth) | [run.sh](./exp/k600/frozen_k600+k710_l14_f32x336/run.sh) | [config.yaml](./exp/k600/frozen_k600+k710_l14_f32x336/config.yaml) |

## K700

| Model                | Pretraining    | #Frame | Top-1 | Model                                                        | Shell                                             | Config                                                      |
| -------------------- | -------------- | ------ | ----- | ------------------------------------------------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| UniFormerV2-B/16     | CLIP-400M      | 8x3x4  | 75.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/k700_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k700/k700_b16_f8x224/run.sh)       | [config.yaml](./exp/k700/k700_b16_f8x224/config.yaml)       |
| UniFormerV2-B/16     | CLIP-400M+K710 | 8x3x4  | 76.3  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/k700_k710_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/k700/k700+k710_b16_f8x224/run.sh)  | [config.yaml](./exp/k700/k700+k710_b16_f8x224/config.yaml)  |
| UniFormerV2-L/14     | CLIP-400M+K710 | 8x3x4  | 80.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/k700_k710_uniformerv2_l14_8x224.pyth) | [run.sh](./exp/k700/k700+k710_l14_f8x224/run.sh)  | [config.yaml](./exp/k700/k700+k710_l14_f8x224/config.yaml)  |
| UniFormerV2-L/14     | CLIP-400M+K710 | 16x3x4 | 81.2  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/k700_k710_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/k700/k700+k710_l14_f16x224/run.sh) | [config.yaml](./exp/k700/k700+k710_l14_f16x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710 | 32x3x2 | 81.5  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/k700_k710_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/k700/k700+k710_l14_f32x224/run.sh) | [config.yaml](./exp/k700/k700+k710_l14_f32x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x2 | 82.1  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/k700_k710_uniformerv2_l14_32x336.pyth) | [run.sh](./exp/k700/k700+k710_l14_f32x336/run.sh) | [config.yaml](./exp/k700/k700+k710_l14_f32x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 64x3x2 | 82.7  | -                                                            | [run.sh](./exp/k700/k700+k710_l14_f64x336/run.sh) | [config.yaml](./exp/k700/k700+k710_l14_f64x336/config.yaml) |

| Frozen Model         | Pretraining    | #Frame | Top-1 | Model                                                        | Shell                                                    | Config                                                       |
| -------------------- | -------------- | ------ | ----- | ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| UniFormerV2-L/14@336 | CLIP-400M      | 8x1x3  | 79.6  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/frozen_k700_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k700/frozen_k700_l14_f8x336/run.sh)       | [config.yaml](./exp/k700/frozen_k700_l14_f8x336/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 8x1x3  | 79.7  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/frozen_k700_k710_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/k700/frozen_k700+k710_l14_f8x336/run.sh)  | [config.yaml](./exp/k700/frozen_k700+k710_l14_f8x336/config.yaml)) |
| UniFormerV2-L/14@336 | CLIP-400M+K710 | 32x3x4 | 80.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k700/frozen_k700_k710_uniformerv2_l14_32x336.pyth) | [run.sh](./exp/k700/frozen_k700+k710_l14_f32x336/run.sh) | [config.yaml](./exp/k700/frozen_k700+k710_l14_f32x336/config.yaml)) |

## Moments in Time V1

| Model                | Pretraining         | #Frame | Top-1 | Model                                                        | Shell                                     | Config                                              |
| -------------------- | ------------------- | ------ | ----- | ------------------------------------------------------------ | ----------------------------------------- | --------------------------------------------------- |
| UniFormerV2-B/16     | CLIP-400M+K710+K400 | 8x3x4  | 42.6  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/mitv1/mit_uniformerv2_b16_8x224.pyth) | [run.sh](./exp/mit/mit_b16_f8x224/run.sh) | [config.yaml](./exp/mit/mit_b16_f8x224/config.yaml) |
| UniFormerV2-L/14     | CLIP-400M+K710+K400 | 8x3x4  | 47.0  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/mitv1/mit_uniformerv2_l14_8x224.pyth) | [run.sh](./exp/mit/mit_l14_f8x224/run.sh) | [config.yaml](./exp/mit/mit_l14_f8x224/config.yaml) |
| UniFormerV2-L/14@336 | CLIP-400M+K710+K400 | 8x3x4  | 47.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/mitv1/mit_uniformerv2_l14_8x336.pyth) | [run.sh](./exp/mit/mit_l14_f8x336/run.sh) | [config.yaml](./exp/mit/mit_l14_f8x336/config.yaml) |

## Something-Something V1

| Model            | Pretraining | #Frame | Top-1 | Model                                                        | Shell                                         | Config                                                |
| ---------------- | ----------- | ------ | ----- | ------------------------------------------------------------ | --------------------------------------------- | ----------------------------------------------------- |
| UniFormerV2-B/16 | CLIP-400M   | 16x3x1 | 56.8  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv1/sthv1_uniformerv2_b16_16x224.pyth) | [run.sh](./exp/sthv1/ssv1_b16_f16x224/run.sh) | [config.yaml](exp/sthv1/ssv1_b16_f16x224/config.yaml) |
| UniFormerV2-B/16 | CLIP-400M   | 32x3x1 | 59.4  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv1/sthv1_uniformerv2_b16_32x224.pyth) | [run.sh](./exp/sthv1/ssv1_b16_f32x224/run.sh) | [config.yaml](exp/sthv1/ssv1_b16_f32x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M   | 16x3x1 | 60.5  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv1/sthv1_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/sthv1/ssv1_l14_f16x224/run.sh) | [config.yaml](exp/sthv1/ssv1_l14_f16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M   | 32x3x1 | 62.7  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv1/sthv1_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/sthv1/ssv1_l14_f32x224/run.sh) | [config.yaml](exp/sthv1/ssv1_l14_f32x224/config.yaml) |

## Something-Something V2

| Model            | Pretraining | #Frame | Top-1 | Model                                                        | Shell                                         | Config                                                |
| ---------------- | ----------- | ------ | ----- | ------------------------------------------------------------ | --------------------------------------------- | ----------------------------------------------------- |
| UniFormerV2-B/16 | CLIP-400M   | 16x3x1 | 69.5  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv2/sthv2_uniformerv2_b16_16x224.pyth) | [run.sh](./exp/sthv2/ssv2_b16_f16x224/run.sh) | [config.yaml](exp/sthv2/ssv2_b16_f16x224/config.yaml) |
| UniFormerV2-B/16 | CLIP-400M   | 32x3x1 | 70.7  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv2/sthv2_uniformerv2_b16_32x224.pyth) | [run.sh](./exp/sthv2/ssv2_b16_f32x224/run.sh) | [config.yaml](exp/sthv2/ssv2_b16_f32x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M   | 16x3x1 | 72.1  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv2/sthv2_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/sthv2/ssv2_l14_f16x224/run.sh) | [config.yaml](exp/sthv2/ssv2_l14_f16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M   | 32x3x1 | 73.0  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/sthv2/sthv2_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/sthv2/ssv2_l14_f32x224/run.sh) | [config.yaml](exp/sthv2/ssv2_l14_f32x224/config.yaml) |

## ActivityNet

| Model            | Pretraining         | #Frame  | Top-1 | Model                                                        | Shell                                       | Config                                                |
| ---------------- | ------------------- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------- |
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 16x3x10 | 94.3  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/anet/anet_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/anet/anet_l14_16x224/run.sh) | [config.yaml](./exp/anet/anet_l14_16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 32x3x10 | 94.7  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/anet/anet_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/anet/anet_l14_32x224/run.sh) | [config.yaml](./exp/anet/anet_l14_32x224/config.yaml) |

## HACS

| Model            | Pretraining         | #Frame  | Top-1 | Model                                                        | Shell                                       | Config                                                |
| ---------------- | ------------------- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------- |
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 16x3x10 | 95.5  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/hacs/hacs_uniformerv2_l14_16x224.pyth) | [run.sh](./exp/hacs/hacs_l14_16x224/run.sh) | [config.yaml](./exp/hacs/hacs_l14_16x224/config.yaml) |
| UniFormerV2-L/14 | CLIP-400M+K710+K400 | 32x3x10 | 95.4  | [ckpt](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/hacs/hacs_uniformerv2_l14_32x224.pyth) | [run.sh](./exp/hacs/hacs_l14_32x224/run.sh) | [config.yaml](./exp/hacs/hacs_l14_32x224/config.yaml) |

