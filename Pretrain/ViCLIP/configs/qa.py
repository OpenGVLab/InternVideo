from .pretrain import *

del available_corpus

criterion["loss_weight"]["mlm"] = 0.0
scheduler["warmup_epochs"] = 0.5

max_txt_l = 32
batch_size = 32
num_frames = 12

optimizer["lr"] = 1e-5
log_freq = 100

# =========additional args for VQA ============
eos = "[SEP]"
max_q_len = 25
max_a_len = 5
# =========end ================================

