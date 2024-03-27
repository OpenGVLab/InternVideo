from configs.data import *
from configs.model import *

# ========================= data ==========================
train_corpus = "summarized_230602_resized_200m"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(msrvtt_1k_test=available_corpus["msrvtt_1k_test"])
test_types = ["msrvtt_1k_test"]
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 512
batch_size_test = 64
max_txt_l = 32

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="VindLU_VideoCLIP",
    vision_encoder=dict(
        # backbone
        name="vit_l14",
        pretrained=True,
        d_model=1024,
        kernel_size=1,
        center=True,
        drop_path_rate=0.1,
        masking_prob=0.9,
        checkpoint_num=24,
    ),
    text_encoder=dict(
        pretrained="bert-base-uncased",  # This is for vindlu default tokenizer, this is never used
        name="vit_l14",
        d_model=768,
        vocab_size=49408,
    ),
    requires_raw_text=True,
    embed_dim=768,
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_text=True,
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
        # mlm=1.0, 
        # vtm=1.0, 
        # mvm=0.0,
        # mac=1.0,
    ),  # 0: disabled.
)

optimizer = dict(
    opt="adamW",
    lr=4e-4,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.2,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=0.5)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    entity="likunchang",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="vindlu_videoclip",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 10
seed = 42

save_latest = False
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=False,
    stage=2,
)
