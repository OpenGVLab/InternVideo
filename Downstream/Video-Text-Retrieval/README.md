# Video-Text-Retrieval

## Introduction  

A model achieves the state-of-the-art video-text retrieval performance on two settings, six datasets, and ten metrics.

## Main Dependencies  

- CUDA Version 11.1   
- PyTorch 1.8.1  
- torchvision 0.9.0  
- python 3.6.9  

## Usage  

### Data Preparation  

Download Original Dataset: [MSR-VTT](http://ms-multimedia-challenge.com/2017/dataset), [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/), [LSMDC](https://sites.google.com/site/describingmovies/download), [DiDeMo](https://github.com/LisaAnne/LocalizingMoments), [ActivityNet](http://activity-net.org/download.html), [VATEX](https://eric-xw.github.io/vatex-website/about.html).  

All annotation files are provided in the *data/* folder.  

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

### Performance 
#### MSRVTT Finetune
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|55.2|79.6|87.5|57.9|79.2|86.4|
#### MSRVTT ZeroShot
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|40.7|67.3|76.0|39.6|63.7|73.1|

#### MSVD Finetune
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|58.4|84.5|90.4|76.3|96.8|98.7|
#### MSVD ZeroShot
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|43.4|69.9|79.1|67.6|90.6|94.6|

#### LSMDC Finetune
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|34.0|53.7|62.9|34.9|54.6|63.1|
#### LSMDC ZeroShot
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|17.6|32.4|40.2|13.2|27.8|34.9|

#### ActivityNet Finetune
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|62.2|85.9|93.2|62.8|86.2|93.3|
#### ActivityNet ZeroShot
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|30.7|57.4|70.2|31.4|59.4|73.1|


#### DiDeMO Finetune
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|57.9|82.4|88.9|59.1|81.8|89.0|
#### DiDeMo ZeroShot
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|31.5|57.6|68.2|33.5|60.3|71.1|

#### VATEX Finetune
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|71.1|94.7|97.6|87.2|99.2|100.0|
#### VATEX ZeroShot
|R@1(t2v)|R@5(t2v)|R@10(t2v)|R@1(v2t)|R@5(v2t)|R@10(v2t)|
|----|----|----|----|----|----|
|49.5|79.7|87.0|69.5|95.0|98.1|





## License  

This project is released under the MIT license.  

## Acknowledgments  

Our codebase is based on [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip).
