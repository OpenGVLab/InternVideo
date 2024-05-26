# The grounding evaluation for InternVideo2
 
> CG-DETR : Calibrating the Query-Dependency of Video Representation via Correlation-guided Attention for Video Temporal Grounding
> WonJun Moon, SangEek Hyun, SuBeen Lee, Jae-Pil Heo 
> Sungkyunkwan University

##### [Arxiv](https://arxiv.org/abs/2311.08835)


## 📑 Features of Datasets
<b>QVHighlights</b> : Download official feature files for QVHighlights dataset from [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB). 
```
tar -xf path/to/moment_detr_features.tar.gz
```
<b>Charades-STA</b> : <b> [Charades-STA](https://drive.google.com/file/d/1B2721QC799qbbGLGSa7DkXJjdRefvZf-/view?usp=sharing )</b> 33.18GB. (Including SF+C and VGG features) 

After downloading, prepare the data directory as below.

<b>Features of Datasets Extracted by InternVideo2</b> : [Features](https://huggingface.co/cg1177 ) .

```txt
.
├── CGDETR
│   ├── cg_detr
│   └── data
│   └── results
│   └── run_on_video
│   └── standalone_eval
│   └── utils
├── features
    └── qvhighlight
    └── charades

```


## 🚀 Training
We provide training scripts for all datasets in `cg_detr/scripts/` directory.

### Download the packages we used for training.
```
pip install -r requirements.txt 
```
### QVHighlights Training
Training can be executed by running the shell below:
```
bash cg_detr/scripts/train.sh  
```
Best validation accuracy is yielded at the last epoch. 

### Charades-STA
For training, run the shell below:
```
bash cg_detr/scripts/charades_sta/train.sh
```

## ☑️ LICENSE
The annotation files and many parts of the implementations are borrowed from [Moment-DETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.
 
