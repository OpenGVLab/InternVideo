# VLN-CE Downstream
This is an official implementation of the visual-language navigation task in [InternVideo](https://arxiv.org/abs/2212.03191).

## Running the code

We currently provide evaluation of our pretrained model.

### Conda environment preperation
1. Please follow https://github.com/jacobkrantz/VLN-CE to install Habitat Simulator and Habitat-lab. We use Python 3.6 in our experiments.
2. Follow https://github.com/openai/CLIP to install CLIP.

### Data/Model preperation
1. Follow https://github.com/jacobkrantz/VLN-CE to download Matterport3D environment to `data/scene_datasets`. Data ahould have the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`.
2. Download preporcessed VLN-CE dataset from  https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/vln/dataset.zip to `data/datasets`. Data should have the form `data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}` and `data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}`.
3. Download pretrained models from  https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/vln/pretrained.zip to `pretrained`. It should have 6 folders: `pretrained/pretrained_models`, `pretrained/VideoMAE`, `pretrained/wp_pred`, `pretrained/ddppo-models`, `pretrained/Prevalent`, `pretrained/wp_pred`.

### Running the code
Simply run `bash eval_**.sh` to start evaluating the agent.