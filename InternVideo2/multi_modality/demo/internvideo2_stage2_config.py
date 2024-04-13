from configs.data import *
from configs.model import *
# ========================= data ==========================
# NOTE The train_file will not be used during the evaluation

num_workers = 6

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 8
batch_size_test = 4
size_t = 224
max_txt_l = 40

origin_num_frames = 4

use_half_precision = False
use_bf16 = False

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
        pretrained='your_model_path/1B_stage2_pt.pth',
        use_checkpoint=True,
        checkpoint_num=40,
        use_flash_attn=use_half_precision,
        use_fused_rmsnorm=use_half_precision,
        use_fused_mlp=use_half_precision,
        # clip teacher
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=False
)

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

gradient_checkpointing = True # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# ========================= optimizer ==========================
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = False
auto_resume = True
jump_evaluate = False
pretrained_path = ""

deepspeed = dict(
    enable=True,
    stage=1,
)
