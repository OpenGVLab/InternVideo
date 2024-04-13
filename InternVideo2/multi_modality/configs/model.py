pretrained_paths = dict(
    BEATs_PATH="/mnt/petrelfs/wangchenting/working/VAST/pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt",
    UMT_S1_B_PATH="/mnt/lustre/share/videointern/annotations/pretained_model/clipmae_vit_b16_k710_e200.pth",
    UMT_S1_L_PATH="/mnt/lustre/share/videointern/annotations/pretained_model/clipmae_vit_l16_k710_e200.pth",
    UMT_S1_g_PATH='/mnt/petrelfs/share_data/likunchang/model/um_teacher/umt2/vit_g14_1.1M_CLIP+MAE_300e_pt_k710_ft.pth',
    InternVL_6B_PATH = "/mnt/petrelfs/share_data/wangwenhai/internvl/6b_vit_exp126_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/1499/mp_rank_00_model_states.pt"
)


VisionEncoders = dict()


TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config="configs/config_bert.json",
    d_model=768,
    fusion_layer=9,
)
TextEncoders["bert_large"] = dict(
    name="bert_large",
    pretrained="bert-large-uncased",
    config="configs/config_bert_large.json",
    d_model=1024,
    fusion_layer=19,
)
TextEncoders["med_bert"] = dict(
    name="med_bert_base",
    pretrained="bert-base-uncased",
    config="configs/med_config.json",
    d_model=768,
)

TextEncoders["med_bert_large"] = dict(
    name="med_bert_large",
    pretrained="bert-base-uncased", # not a bug, it just follows BLIP.
    config="configs/med_large_config.json",
    d_model=768
)