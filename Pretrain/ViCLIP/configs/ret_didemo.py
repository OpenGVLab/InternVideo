from .pretrain import *

del available_corpus

train_file = [
    f"{anno_root_downstream}/didemo_ret_train.json",
    f"{data_root}/didemo_2fps_360_trimed30",
    "video",
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/didemo_ret_val.json",
        f"{data_root}/didemo_2fps_360_trimed30",
        "video",
    ],
    test=[
        f"{anno_root_downstream}/didemo_ret_test.json",
        f"{data_root}/didemo_2fps_360_trimed30",
        "video",
    ],
)

test_types = ["val"]
stop_key = "val/"  # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = True

criterion["loss_weight"]["mlm"] = 0.0
scheduler["warmup_epochs"] = 0
optimizer["lr"] = 1e-5


max_txt_l = 64
batch_size = 32
num_frames = 12

log_freq = 10
