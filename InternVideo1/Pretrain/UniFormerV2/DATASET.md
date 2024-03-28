# Dataset Preparation

We provide the labels of our dataset [here](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL?usp=sharing), including:
- [Kinetics-400/600/700](https://www.deepmind.com/open-source/kinetics)
- [Moments in Time V1](http://moments.csail.mit.edu/)
- [Something-Something V1&V2](https://developer.qualcomm.com/software/ai-datasets/something-something)
- [ActivityNet](http://activity-net.org/)
- [HACS](http://hacs.csail.mit.edu/)
- Our Kinetics-710

For videos, please download them from the dataset providers. You can simply download the metadata files and put them in `data_list`. Note that we use `decord` to decode all the datasets on the fly except Sth-Sth.
> Since some videos in Kinetics may no longer be available, it will lead to small performance gap.

## ActivityNet and HACS

For ActivityNet and HACS, we adopt extra pre-processing. The code can be found in our meta files.

- Training: We split the video according to the `start` and `end`, and we only use those video clips with actions.
- Validation: Since there is only one action in a single video, we directly predict the class via sparse sampling from the total video.

## Kinetics-710
For Kientics-710, we merge the training set of Kinetics-400/600/700, and then delete the repeated videos according to Youtube IDs. Note we also remove testing videos from different Kinetics datasets leaked in our combined training set for correctness. As a result, the total number of training videos is reduced from 1.14M to 0.65M. 
Additionally, we merge the action categories in these three Kinetics datasets, which leads to 710 classes in total. Hence, we call this video benchmark Kinetics-710. More detailed descriptions can be found in our Appendix E. 

In our experiments, we empirically show the effectiveness of our Kinetics-710. For post-pretraining, we simply use 8 input frames and adopt the same hyperparameters as training on the individual Kinetics dataset. After that, no matter how many frames are input (16, 32, or even 64), we only need 5-epoch finetuning for more than 1% top-1 accuracy improvement on Kinetics-400/600/700.
> When finetuning the K710-pretrained models, we load the weights of classification layers and map the weight according to the label list. **We have provide the label map in the meta files.**

| Model       | Pretrain | #Frame | K400 | K600 | K700 |
| ----------- | -------- | ------ | ---- | ---- | ---- |
| UniFormerV2-B | CLIP-400M  | 8x3x4  | 84.4 | 85.0 | 75.8 |
| UniFormerV2-B | CLIP-400M+K710  | 8x3x4  | **85.6 (+1.2)** | **86.1 (+1.1)** | **76.3 (+0.5)** |
| UniFormerV2-L | CLIP-400M  | 8x3x4  | 87.7 | 88.0 | 80.3 |
| UniFormerV2-L | CLIP-400M+K710  | 8x3x4  | **88.8 (1.1)** | **89.0 (+1.0)** | **80.8 (+0.5)** |