QVHighlights Evaluation and Codalab Submission
==================

### Task Definition
Given a video and a natural language query, our task requires a system to retrieve the most relevant moments in the video, and detect the highlightness of the clips in the video. 

### Evaluation
At project root, run
```
bash standalone_eval/eval_sample.sh 
```
This command will use [eval.py](eval.py) to evaluate the provided prediction file [sample_val_preds.jsonl](sample_val_preds.jsonl), 
the output will be written into `sample_val_preds_metrics.json`. 
The content in this generated file should be similar if not the same as [sample_val_preds_metrics_raw.json](sample_val_preds_metrics_raw.json) file.

### Format

The prediction file [sample_val_preds.jsonl](sample_val_preds.jsonl) is in [JSON Line](https://jsonlines.org/) format, each row of the files can be loaded as a single `dict` in Python. Below is an example of a single line in the prediction file:
```
{
  "qid": 2579,
  "query": "A girl and her mother cooked while talking with each other on facetime.",
  "vid": "NUsG9BgSes0_210.0_360.0",
  "pred_relevant_windows": [
    [0, 70, 0.9986],
    [78, 146, 0.4138],
    [0, 146, 0.0444],
    ...
  ],  
  "pred_saliency_scores": [-0.2452, -0.3779, -0.4746, ...]
}

```



| entry | description |
| --- | ----|
| `qid` | `int`, unique query id |
| `query` | `str`, natural language query, not used by the evaluation script | 
| `vid` | `str`, unique video id | 
| `pred_relevant_windows` | `list(list)`, moment retrieval predictions. Each sublist contains 3 elements, `[start (seconds), end (seconds), score]`| 
| `pred_saliency_scores` | `list(float)`, highlight prediction scores. The higher the better. This list should contain a score for each of the 2-second clip in the videos, and is ordered. | 


### Codalab Submission
To test your model's performance on `test` split, 
please submit both `val` and `test` predictions to our 
[Codalab evaluation server](https://codalab.lisn.upsaclay.fr/competitions/6937). 
The submission file should be a single `.zip ` file (no enclosing folder) 
that contains the two prediction files 
`hl_val_submission.jsonl` and `hl_test_submission.jsonl`, each of the `*submission.jsonl` file 
should be formatted as instructed above. 

