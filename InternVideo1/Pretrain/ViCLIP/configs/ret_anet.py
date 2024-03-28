from .pretrain import *

del available_corpus

train_file = [
    f"{anno_root_downstream}/anet_ret_train.json",
    f"{data_root}/activity_net_2fps_360",
    "video",
]
test_file = dict(
    test=[
        f"{anno_root_downstream}/anet_ret_val_1.json",
        f"{data_root}/activity_net_2fps_360",
        "video",
    ],
)

test_types = ["test"]
stop_key = "test/"  # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = True

max_txt_l = 64
batch_size = 32
num_frames = 12

optimizer["lr"] = 1e-5
log_freq = 100
