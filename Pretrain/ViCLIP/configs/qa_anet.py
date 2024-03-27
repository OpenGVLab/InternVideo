from .qa import *

train_file = [
    [
        f"{anno_root_downstream}/anet_qa_train.json",
        f"{data_root}/activity_net_2fps_360",
        "video",
    ]
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/anet_qa_val.json",
        f"{data_root}/activity_net_2fps_360",
        "video",
    ],
    test=[
        f"{anno_root_downstream}/anet_qa_test.json",
        f"{data_root}/activity_net_2fps_360",
        "video",
    ]
)
dataset_name = "anet"

answer_list = f"{anno_root_downstream}/anet_qa_answer_list.json"  # list of answer words

test_types = ["val"]
stop_key = "val"  # used to choose the best ckpt. If None, save the last.
