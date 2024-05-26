## QVHighlights Dataset

Our annotation files include 3 splits: `train`, `val` and `test`. Each file is in [JSON Line](https://jsonlines.org/) format, each row of the files can be loaded as a single `dict` in Python. Below is an example of the annotation:

```
{
    "qid": 8737, 
    "query": "A family is playing basketball together on a green court outside.", 
    "duration": 126, 
    "vid": "bP5KfdFJzC4_660.0_810.0", 
    "relevant_windows": [[0, 16]],
    "relevant_clip_ids": [0, 1, 2, 3, 4, 5, 6, 7], 
    "saliency_scores": [[4, 1, 1], [4, 1, 1], [4, 2, 1], [4, 3, 2], [4, 3, 2], [4, 3, 3], [4, 3, 3], [4, 3, 2]]
}
```
`qid` is a unique identifier of a `query`. This query corresponds to a video identified by its video id `vid`. The `vid` is formatted as `{youtube_id}_{start_time}_{end_time}`. Use this information, one can retrieve the YouTube video from a url `https://www.youtube.com/embed/{youtube_id}?start={start_time}&end={end_time}&version=3`. For example, the video in this example is `https://www.youtube.com/embed/bP5KfdFJzC4?start=660&end=810&version=3`. 
`duration` is an integer indicating the duration of this video.
`relevant_windows` is the list of windows that localize the moments, each window has two numbers, one indicates the start time of the moment, another one indicates the end time. `relevant_clip_ids` is the list of ids to the segmented 2-second clips that fall into the moments specified by `relevant_windows`, starting from 0.
`saliency_scores` contains the saliency scores annotations, each sublist corresponds to a clip in `relevant_clip_ids`. There are 3 elements in each sublist, they are the scores from three different annotators. A score of `4` means `Very Good`, while `0` means `Very Bad`.

Note that the three fields `relevant_clip_ids`, `relevant_windows` and `saliency_scores` for `test` split is not included. Please refer to [../standalone_eval/README.md](../standalone_eval/README.md) for details on evaluating predictions on `test`.

In addition to the annotation files, we also provided the subtitle file for our weakly supervised ASR pre-training: [subs_train.jsonl](./subs_train.jsonl). This file is formatted similarly as our annotation files, but without the `saliency_scores` entry. This file is not needed if you do not plan to pretrain models using it.

