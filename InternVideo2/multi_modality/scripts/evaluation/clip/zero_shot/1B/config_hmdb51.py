from configs.data import *
from configs.model import *

# ========================= data ==========================
train_corpus = "webvid_debug"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(act_val=available_corpus["hmdb51_act_val"])
test_types = ["act_val"]
num_workers = 12

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 256
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
model = dict(
    model_cls="InternVideo2_CLIP",
    vision_encoder=dict(
        name="internvideo2_1B",
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.3,
        head_drop_path_rate=0.,
        embed_dim=1408,
        num_heads=16,
        mlp_ratio=48/11,
        init_values=0.1,
        qk_normalization=True,
        depth=40,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        fused_mlp_heuristic=1,
        drop_cls_token=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
    ),
    text_encoder=dict(
        use_flash_attn=True,
        transformer_width=4096,
        llama_path="your_model_path/chinese_alpaca_lora_7b",
        use_lora=True,
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=True,
    open_vision_clip_projector=True,
    freeze_text=True,
    open_text_projection=False,
    open_text_lora=False,
    tokenizer_path="your_model_path/chinese_alpaca_lora_7b",
    vision_ckpt_path="your_model_path/InternVideo2_Stage2_1B.pth",
    load_vision_ckpt_from_internvideo2_stage2=True,
    text_ckpt_path="your_model_path/internvl/internvl_c_13b_224px.pth",
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
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

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.6)

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="likunchang",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="InternVideo2_CLIP",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=False,
    stage=0,
)