from .pretrain import *

del available_corpus

train_file = [
    f"{anno_root_downstream}/tvqa_train_with_answer.json",
    f"{data_root}/tvqa_trimmed_3fps",
    "video",
]
test_file = dict(
    val=[
        f"{anno_root_downstream}/tvqa_val_with_answer.json",
        f"{data_root}/tvqa_trimmed_3fps",
        "video",
    ],
    test=[
        f"{anno_root_downstream}/tvqa_test_public_with_answer.json",
        f"{data_root}/tvqa_trimmed_3fps",
        "video",
    ],
)

test_types = ["val"]
stop_key = "val"  # used to choose the best ckpt. If None, save the last.
is_paragraph_retrieval = False

criterion["loss_weight"]["mlm"] = 0.0
optimizer["lr"] = 1e-5
scheduler["warmup_epochs"] = 0.5
scheduler["epochs"] = 10

max_txt_l = 150
batch_size = 32
num_frames = 12

log_freq = 100
