# Dataset Preparation


# Stage2——Video-language Alignment


## Pretraining

The public portion of the pre-trained dataset we use is as follows：
- [CC3M images](https://github.com/google-research-datasets/conceptual-captions)
- [CC12M images](https://github.com/google-research-datasets/conceptual-12m)
- [SBU images](https://www.cs.rice.edu/~vo9/sbucaptions/)
- [VG images](https://visualgenome.org/api/v0/api_home.html)
- [COCO images](https://cocodataset.org/#download)
- [WebVid videos](https://github.com/m-bain/webvid)
- [InternVid videos](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)

## Evaluation

For evaluation, we follow [VINDLU](https://github.com/klauscc/VindLU/) to prepare the datasets, but we **DO NOT** compress the videos and images.  We use the original data and load the JSON files. And We use the same **JSON** files provided by [VINDLU](https://drive.google.com/drive/folders/12bC7WotvwyTG4pVvYeU4iZzmBLP1-6d9). 


### Video-Text Retrieval

- [MSRVTT videos](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip)
- [MSVD videos](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
- [ActivityNet videos](http://activity-net.org/download.html)
- [DiDeMo videos](https://github.com/LisaAnne/LocalizingMoments)


# Stage3——VideoChat

## Pretraining

- [VideoChat-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)


## Evaluation
### MVBench

Please refer to [MVBench](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)

