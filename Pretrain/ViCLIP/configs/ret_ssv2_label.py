from .ret_msrvtt import *

train_file = [
    f"{anno_root_downstream}/ssv2_ret_label_train.json",
    f"{data_root}/ssv2",
    "video",
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/ssv2_ret_label_val_small.json",
        f"{data_root}/ssv2",
        "video",
    ],
)

test_types = ["val"]
stop_key = None  # used to choose the best ckpt. If None, save the last.

has_multi_vision_gt = True

scheduler["epochs"] = 10
optimizer["lr"] = 1e-4

max_txt_l = 25
