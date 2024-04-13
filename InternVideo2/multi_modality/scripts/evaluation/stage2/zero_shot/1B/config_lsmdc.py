from configs.data import *
from configs.model import *

# ========================= data ==========================

train_file = available_corpus["lsmdc_ret_test_1000"]
test_file = dict(lsmdc_ret_test_1000=available_corpus["lsmdc_ret_test_1000"])

test_types = ["lsmdc_ret_test_1000"]

num_workers = 6

best_key = ["lsmdc_ret_test_1000_match", "t2v_r1"]

# ========================= input ==========================
origin_num_frames = 4
num_frames = 8
num_frames_test = 8
batch_size = 8 # 8 * 32
batch_size_test = 4
max_txt_l = 40

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
    model_cls="InternVideo2_Stage2",
    vision_encoder=dict(
        # backbone
        name="pretrain_internvideo2_1b_patch14_224",
        img_size=224, 
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14, 
        d_model=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained='your_model_path/1B_pt.pth',
        use_checkpoint=False,
        checkpoint_num=40,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        # clip teacher
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        only_mask=True,
        sep_image_video_pos_embed=True
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=False
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
        mlm=1.0, 
        vtm=1.0, 
        mvm=0.0,
        uta=0.0,
    ),  # 0: disabled.
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.]
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.05,
    max_grad_norm=3.,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-5),
)

scheduler = dict(sched="cosine", epochs=20, min_lr_multi=0.01, warmup_epochs=1)

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

gradient_checkpointing = True # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="opengvlab",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="InternVideo2-ft",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "ret"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42
zero_shot = True

save_latest = False
auto_resume = True
jump_evaluate = False
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=True,
    stage=1,
)
