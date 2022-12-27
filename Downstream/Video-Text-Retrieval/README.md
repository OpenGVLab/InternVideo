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


## License  

This project is released under the MIT license.  

## Acknowledgments  

Our codebase is based on [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip).

