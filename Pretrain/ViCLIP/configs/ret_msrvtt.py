from .pretrain import *

del available_corpus

train_file = [
    f"{anno_root_downstream}/msrvtt_ret_train7k.json",
    f"{data_root}/msrvtt_2fps_224",
    "video",
]
test_file = dict(
    test=[
        f"{anno_root_downstream}/msrvtt_ret_test1k.json",
        f"{data_root}/msrvtt_2fps_224",
        "video",
    ],
)

test_types = ["test"]
stop_key =  None # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = False

criterion["loss_weight"]["mlm"] = 0.0
scheduler["warmup_epochs"] = 0
scheduler["epochs"] = 5
optimizer["lr"] = 1e-5

max_txt_l = 32
batch_size = 32
num_frames = 12

log_freq = 100
