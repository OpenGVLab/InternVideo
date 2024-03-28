from .qa import *

train_file = [
    [
        f"{anno_root_downstream}/msrvtt_qa_train.json",
        f"{data_root}/msrvtt_2fps_224",
        "video",
    ]
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/msrvtt_qa_val.json",
        f"{data_root}/msrvtt_2fps_224",
        "video",
    ],
    test=[
        f"{anno_root_downstream}/msrvtt_qa_test.json",
        f"{data_root}/msrvtt_2fps_224",
        "video",
    ],
)
dataset_name = "msrvtt"

answer_list = f"{anno_root_downstream}/msrvtt_qa_answer_list.json"  # list of answer words

test_types = ["val"]
stop_key = "val"  # used to choose the best ckpt. If None, save the last.
