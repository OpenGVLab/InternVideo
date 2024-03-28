from .pretrain import *

del available_corpus

train_file = [
    f"{anno_root_downstream}/coco_train.json",
    f"{data_root}/coco",
    "video",
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/coco_val.json",
        f"{data_root}/coco",
        "video",
    ],
    test=[
        f"{anno_root_downstream}/coco_test.json",
        f"{data_root}/coco",
        "video",
    ],
)

test_types = ["val"]
stop_key = "val/"  # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = False

criterion["loss_weight"]["mlm"] = 0.0
scheduler["warmup_epochs"] = 0
optimizer["lr"] = 1e-5


max_txt_l = 22
batch_size = 128
num_frames = 1
num_frames_test = 1

log_freq = 100
