## Installation

Please follow the installation instructions in [INSTALL](INSTALL.md).

## Datasets

You can find the dataset instructions in [DATASET](DATASET.md). We have provide all the metadata files of our data.


## Note
- All the `config.yaml` in our `exp` are **NOT** the training config actually used, since some hyperparameters are **changed** in the `run.sh` or `test.sh`.
- For more config details, you can read the comments in `slowfast/config/defaults.py`.
- We adopt **sparse sampling** for all the datasets.
- For those **scene-related** datasets (e.g., Kinetics), we **ONLY** add global UniBlocks.
- For those **temporal-related** datasets (e.g., Sth-Sth), we adopt **ALL** the designs, including local UniBlocks, global UniBlocks and temporal downsampling.
- If you meet problem when running the backward process, please see [issue#4](https://github.com/OpenGVLab/UniFormerV2/issues/4).
```yaml
N_LAYERS: 4  # number of global UniBlocks
MLP_DROPOUT: [0.5, 0.5, 0.5, 0.5]  # dropout for each global UniBlocks
CLS_DROPOUT: 0.5  # dropout for the final classification layer
RETURN_LIST: [8, 9, 10, 11]  # layer index for inserting global UniBlocks
NO_LMHRA: True  # whether adding local MHRA in the local UniBlocks
TEMPORAL_DOWNSAMPLE: False  # whether using temporal downsampling in the patch embedding
FROZEN: False  # whether freeze backbone
```


## Training

Our models are based on pretrained ViTs, and we use [CLIP](https://github.com/openai/CLIP) pretrained models by default:
- Follow `extract_clip` to extract visual encoder from CLIP.
- Change `MODEL_PATH` in `slowfast/models/uniformerv2_model.py`.

For training, you can simply run the training scripts in `exp` as follows:
```shell
bash ./exp/k400/k400_b16_f8x224/run.sh
```


## Testing
For testing, you can simply run the training scripts in `exp` as follows:

```shell
bash ./exp/k400/k400_b16_f8x224/test.sh
```

Make sure `TRAIN.ENABLE=False`. You can set the number of crops and clips (in`test.sh`) as follows:


   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 4
   TEST.NUM_SPATIAL_CROPS 3
   ```

You can also set the checkpoint path as follows:

```shell
TEST.CHECKPOINT_FILE_PATH your_model_path
```
