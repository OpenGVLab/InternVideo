from .data import *
from .model import *

# ========================= data ==========================
train_corpus = "webvid_cc3m"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(msrvtt_1k_test=available_corpus["msrvtt_1k_test"])
test_types = ["msrvtt_1k_test"]
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 64
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
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
vision_enc = "beit"
text_enc = "bert"
model = dict(
    vision_encoder="${VisionEncoders[${vision_enc}]}",
    text_encoder="${TextEncoders[${text_enc}]}",
    temporal_modeling=dict(
        num_frames="${num_frames}",
        temporal_model_block="timesformer",
        temporal_model_position="last",
        temporal_model_config=dict(input_dim="${model.vision_encoder.d_model}"),
        use_temporal_position_embedding=True,
    ),
    vit_add_ln=True,
    multimodal=dict(enable=True),
    embed_dim=256,
    temp=0.07,
)

criterion = dict(
    loss_weight=dict(vtc=1.0, mlm=1.0, vtm=1.0, mvm=0.0),  # 0: disabled.
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
)

optimizer = dict(
    opt="adamW",
    lr=1e-4,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=1)

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
    project="vindlu",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?
