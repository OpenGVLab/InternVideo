# Temporal-Action-Localization
The codebase  of  one downstream task (Temporal Action Localization ) based on our InternVideo.

 [**InternVideo: General Video Foundation Models via Generative and Discriminative Learning**](https://arxiv.org/abs/2212.03191) 

We use ViT-H from our InternVideo as backbones for feature extraction. In our experiment, ViT-H models are pretrained from Hybrid datasets. 
As shown in Table, our InternVideo outperforms best than all the preview methods on these four TAL datasets. 
Note that, our InternVideo achieves huge improvements in temporal action localization, especially in fine-grained TAL datasets such as THUMOS-14 and FineAction.

|Backbone | Head | THUMOS-14 | ActivityNet-v1.3 | HACS | FineAction | 
|:----:|:-----:|:----------------:|:-------:|:-------:|:-------:|
|I3D | ActionFormer | 66.80 |-| - |13.24|
|SlowFast | TCANet | - | - | 38.71 | - |
|TSP | ActionFormer | - | 36.60 | - | - |
|**InternVideo** | ActionFormer | **71.58** | **39.00** | **41.32** | **17.57** |




## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.
* The annotations for the used data can be downloaded from [here](https://pjlab-my.sharepoint.cn/:u:/g/personal/wangyi_pjlab_org_cn/EbMrA1ltSdBPttMho9pe0cYBshPWec7IKmYFTNMGvdmT1A?e=jsmerl). Unzip and put them in the folder `./data`.


## To Reproduce Our Results of InternVideo
**Download VideoMAE Features and Annotations**
* Download *thumos_feature* [this BaiduYun link](https://pan.baidu.com/s/1IVcnPWsZyF6rHEPkSzH5Bw). code：qzb1 
<!-- * The file includes VideoMAE features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores. -->
  

* Download *anet_feature* [this BaiduYun link](https://pan.baidu.com/s/1Vjvnesm7WCGHwjrqQWNVDg). code：0elv 
<!-- * The file includes VideoMAE features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores. -->



* Download *hacs_feature* [this BaiduYun link](https://pan.baidu.com/s/1j6GABMj0tY1OUxU2xzyB6g). code：qcnz
<!-- * The file includes VideoMAE features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores. -->



* Download *fineaction_feature* [this BaiduYun link](https://pan.baidu.com/s/1P5QQMuxcPiE2tn4ojQW7pA). code：v45q 
<!-- * The file includes VideoMAE features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores. -->


**Download UniformerV2 Features (soon)**

**Details**: 
The THUMOS-14 features are extracted from Video_MAE models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `4 frames`. This gives one feature vector per `4/30 ~= 0.1333` seconds.

The ANet & HACS & FineAction features are extracted from Video_MAE models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `16 frames`. This gives one feature vector per `16/30 ~= 0.5333` seconds.

<!-- The HACS features are extracted from Video_MAE models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `16 frames`. This gives one feature vector per `16/30 ~= 0.5333` seconds.

The FineAction features are extracted from Video_MAE models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `16 frames`. This gives one feature vector per `16/30 ~= 0.5333` seconds. -->



**Training and Evaluation**
* Train the ActionFormer with InternVideo features. 
* This will create a experiment folder under *./ckpt* that stores training config, logs, and checkpoints.
```shell
bash th14_run.sh
bash anet_run.sh
bash hacs_run.sh
bash fa_run.sh
```




##  Contact 

Any question about our FineAction, you can email me. (yi.liu1@siat.ac.cn)