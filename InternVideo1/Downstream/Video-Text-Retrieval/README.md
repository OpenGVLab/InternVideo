# Video-Text-Retrieval

## Introduction  

A model achieves the state-of-the-art video-text retrieval performance on two settings, six datasets, and ten metrics.

### Our Results
**Zero-Shot Video Retrieval**
|Dataset| Setting | R@1↑ | R@5↑ | R@10↑ | MedR↓ | MeanR↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| MSRVTT | video-to-text | 37.5 | 63.3 | 71.3 | 3.0 | 24.2 |
| | text-to-video | 40 | 65.3 | 74.1 | 2.0 | 23.9 |
| MSVD | video-to-text | 67.6 | 90.6 | 94.6 | 1.0 | 2.9 |
| | text-to-video | 43.4|69.9|79.1|2.0|17.0|
| LSMDC | video-to-text | 13.2|27.8|34.9|33.0|113.6|
| | text-to-video | 17.6|32.4|40.2|23.0|101.7|
| ActivityNet | video-to-text | 31.4|59.4|73.1|3.0|15.6|
| | text-to-video | 30.7 | 57.4 | 70.2| 4.0 | 23.0|
| DiDeMo | video-to-text | 33.5 | 60.3|71.1|3.0|21.5|
| | text-to-video | 31.5 | 57.6 | 68.2 | 3.0 | 35.7 |
| VATEX | video-to-text | 69.5 | 95|98.1|1.0|2.1|
| | text-to-video | 49.5|79.7|87|2.0|9.7|

<!--
|Dataset| Setting | R@1 | R@5 | R@10 | MedR | MeanR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| MSRVTT | video-to-text | 224x224 | 83.5 | 30M | 5G |
| MSVD | ImageNet-1K | 224x224 | 84.2 | 50M | 8G |
| LSMDC | ImageNet-1K | 224x224 | 84.9 | 97M | 16G |
| ActivityNet | ImageNet-22K | 384x384 | 87.7 | 223M | 108G |
| DiDeMo | ImageNet-22K | 384x384 | 88.0 | 335M | 163G |
| VATEX | ImageNet-22K | 384x384 | 88.0 | 335M | 163G |
-->

**Video Retrieval with Full Finetuning**
|Dataset| Setting | R@1↑ | R@5↑ | R@10↑ | MedR↓ | MeanR↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| MSRVTT | video-to-text | 57.9|79.2|86.4|1.0|7.5|
| | text-to-video | 55.2|79.6|87.5|1.0|10.7|
| MSVD | video-to-text | 76.3|96.8|98.7	|1.0|2.1|
| | text-to-video | 58.4|84.5|90.4|1.0|8.2|
| LSMDC | video-to-text | 34.9|54.6|63.1|4.0|32.9|
| | text-to-video | 34.0|53.7|62.9|4.0|38.7|
| ActivityNet | video-to-text | 62.8|86.2|93.3|1.0|3.5|
| | text-to-video | 62.2|85.9|93.2|1.0|3.9|
| DiDeMo | video-to-text | 59.1|81.8|89.0|1.0|7.2|
| | text-to-video | 57.9|82.4|88.9|1.0|9.2 |
| VATEX | video-to-text | 86.0|99.2|99.6|1.0|1.3|
| | text-to-video | 72.0|95.1|97.8|1.0|2.4|

## Main Dependencies  

- CUDA Version 11.1   
- PyTorch 1.8.1  
- torchvision 0.9.0  
- python 3.6.9  

## Usage  

### Data Preparation  

Download Original Dataset: [MSR-VTT](http://ms-multimedia-challenge.com/2017/dataset), [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/), [LSMDC](https://sites.google.com/site/describingmovies/download), [DiDeMo](https://github.com/LisaAnne/LocalizingMoments), [ActivityNet](http://activity-net.org/download.html), [VATEX](https://eric-xw.github.io/vatex-website/about.html).  

All annotation files can be downloaded from [here](https://pjlab-my.sharepoint.cn/:u:/g/personal/wangyi_pjlab_org_cn/EREJFyTbpwFPppzv3tBlHp4BMUHu2wveRamzqDPF2AdhQQ?e=VmmP4p). Unzip and put them in the *data/* folder.

### Pre-processing (optional)  

`python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]`  

This script will compress the video to 3fps with width 224 (or height 224). Modify the variables for your customization.  

### Pre-trained weights 

The pre-trained ViT weights of [CLIP](https://openai.com/blog/clip/) can be found here: [ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt), [ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), [ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt).

Our fine-tuned retrieval checkpoint on each dataset will be released soon.

### Zeroshot evaluation

All zero-shot scripts are provided in the *zeroshot_scripts/* folder. Be free to try different hyper-parameters.  
```sh
./zeroshot_scripts/eval_msrvtt.sh
```

### Fine-Tune

All fine-tune scripts are provided in the *finetune_scripts/* folder, and the **train_${dataset_name}.sh** files are used to fine-tune our InternVideo model. Be free to try different hyper-parameters.  
```sh
./finetune_scripts/train_msrvtt.sh
```

### Evaluate the finetuned model

All the test scripts for evaluating the finetuned checkpoints are provided in the *eval_finetuned_scripts/* folder. The scripts are slightly different from the zero-shot evaluation scripts. 
```sh
./eval_finetuned_scripts/eval_finetuned_msrvtt.sh
```


## License  

This project is released under the MIT license.  

## Acknowledgments  

Our codebase is based on [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip).

