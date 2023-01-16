from sacred import Experiment

ex = Experiment("CoTrain", save_git_info=False)


def _loss_names(d):
    ret = {
        # pretrain
        "vtm": 0,
        "mlm": 0,
        "mpp": 0,
        "vtc": 0,
        "vcop": 0,
        "dino": 0,
        # downstream
        "vqa": 0,
        "openend_vqa": 0,
        "mc_vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "multiple_choice": 0,
        'vcr_q2a': 0,
        'zs_classify': 0,
        'contrastive': 0,
        'cap': 0,
        'mim': 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "CoTrain"
    seed = 0
    video_datasets = ["wevid", "howto100m", "yttemporal"]
    image_datasets = ["cc3m", "cc12m"]
    val_datasets = []
    loss_names = _loss_names({"vtm": 1, "mlm": 1})
    val_loss_names = _loss_names({})
    batch_size = 4096  # 128 x 32
    # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    linear_evaluation = False

    draw_false_image = 1
    # video setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 224  # 384/224
    patch_size = 16  # 16/32
    max_image_len = -1
    draw_false_video = 1
    video_only = False
    num_frames = 3  # input video frames

    # Text Setting
    vqav2_label_size = 3129
    msrvttqa_label_size = 1501
    max_text_len = 40  # original: 40, 200: for long sentences/paragraph
    tokenizer = "pretrained/bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    draw_options_text = 0
    # Transformer Setting
    vit = "vit_base_patch16_224"  # "vit_base_patch32_384" / "vit_base_patch16_224"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    shared_embedding_dim = 512  #  add for contrastive learning 512/256
    # model_temporal_frames = 4  #  add for model define， may not consistent with input data

    save_checkpoints_interval = 1  # save each 5 epochs

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    backend = 'a100'  # gpu: a100/v100/others

    # Downstream Setting
    get_recall_metric = False
    get_ind_recall_metric = False
    retrieval_views = 3  # how many views for retrieval

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 16  # 0 will not lead to unstable memory usage but slow training ?
    precision = 16
    model_dir = None

    # clip related settings
    clip = ""
    clip_type = "ori"  # In ["evl", "ori"]
    clip_freeze = False
    clip_freeze_text = False
    clip_dpr = 0.0
    prompt_type = "all"
    clip_lr_mult = 1
    clip_no_pretrain = False
    clip_grad_unfreeze_int = 0  # <= 0 for nothing
    clip_evl_dropout = 0.5
    mim_prob = 0.90
    clip_mlm_decoder_n_layers = 4
    clip_mim_decoder_n_layers = 4
    clip_mim_decoder_width = 512
    clip_cap_decoder_n_layers = 4
    clip_init_zero = True
    clip_qa_type = "vtc"  # vtc for contrastive, cap for caption head, both for both
    clip_mc_type = "vtc"  # vtc for contrastive, cap for caption head, both for both
    # weight = clip_weight * clip_wiseft_coef + load_path * (1 - clip_wiseft_coef), <= 0 for not using
    clip_wiseft_coef = -1.0
    clip_mmt = False
    clip_alt_data = False
    image_data_mult = 1
    clip_cls_dropout = 0.5
    save_last = True
    save_top_k = 1
    clip_use_checkpoint = False
    clip_checkpoint_num = [0, 0, 0]
    clip_momentum_ckpt = 1
    clip_momentum_interval = 1


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/CoTrain/result"
    num_gpus = 8
    num_nodes = 1

# ================================ begin: pretrain ======================
@ex.named_config
def task_mlm_vtm_cotrain():
    exp_name = "mlm_vtm"
    video_datasets = ["webvid"]  # "howto100m",
    image_datasets = ["cc3m"]
    loss_names = _loss_names({"vtm": 1, "mlm": 1})
    batch_size = 2048
    max_epoch = 30
    max_image_len = -1
    val_check_interval = 1.0
    save_checkpoints_interval = 3  # save each 5 epochs

@ex.named_config
def task_mlm_vtm_cotrain_seven():
    exp_name = "mlm_vtm"
    video_datasets = ["webvid", 'yttemporal', "howto100m"]  # 'yttemporal', "howto100m",
    image_datasets = ["cc3m", "cc12m", "vg", 'coco']  # ,  "vg", 'coco'
    loss_names = _loss_names({"vtm": 1, "mlm": 1})
    batch_size = 2048
    max_epoch = 30
    max_image_len = -1
    val_check_interval = 1.0
    save_checkpoints_interval = 1  # save each 5 epochs

@ex.named_config
def task_mlm_vtm_vcop_cotrain():
    exp_name = "mlm_vtm"
    video_datasets = ["howto100m", "webvid"]
    image_datasets = ["cc3m"]
    loss_names = _loss_names({"vtm": 1, "mlm": 1, "vcop": 1})
    batch_size = 2048
    max_epoch = 30
    max_image_len = -1
    val_check_interval = 1.0
    save_checkpoints_interval = 5  # save each 5 epochs

@ex.named_config
def task_mlm_vtm_dino_cotrain():
    exp_name = "mlm_vtm_dino_1f"
    video_datasets = ["webvid"]  # "howto100m",
    image_datasets = ["cc3m"]
    loss_names = _loss_names({"vtm": 1, "mlm": 1, "dino": 1})  # already include dino
    train_transform_keys = ["pixelbert_randaug"]
    val_transform_keys = ["pixelbert_randaug"]
    batch_size = 1024
    max_epoch = 100
    max_image_len = -1
    val_check_interval = 1.0
    save_checkpoints_interval = 1  # save each 5 epochs
# ================================ end: pretrain ======================


# ================================ begin: finetune ======================
# ========== begin: multiple choice ================
# = for lsmdc multiple choice
@ex.named_config
def task_finetune_lsmdcchoice():
    exp_name = "finetune_lsmdc_choice"
    video_datasets = ["lsmdc_choice"]
    image_datasets = []
    loss_names = _loss_names({"multiple_choice": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 5  # 5 choices
    learning_rate = 1e-5
    val_check_interval = 0.5
    lr_mult = 10

# = for msrvtt multiple choice
@ex.named_config
def task_finetune_msrvttchoice():
    exp_name = "finetune_msrvtt_choice"
    video_datasets = ["msrvtt_choice"]
    image_datasets = []
    loss_names = _loss_names({"multiple_choice": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 5  # 5 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10
# ========== end: multiple choice ================
# ind itc
# ========== begin: retrieval ================
@ex.named_config
def task_finetune_vtc_irtr_msrvtt():
    exp_name = "finetune_vtc_irtr_msrvtt"
    video_datasets = ["msrvtt"]
    image_datasets = []
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vtc": 1})
    batch_size = 1024
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1  # 0.1/0.3
    retrieval_views = 1  # use 5 views
    get_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 15
    learning_rate = 6e-4  # 1/3e-4
# ========== end: retrieval ================
# ========== begin: vqa ================
# for msvd qa
@ex.named_config
def task_finetune_msvdqa():
    exp_name = "finetune_msvd_qa"
    video_datasets = ["msvdqa"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})  # msvd have same number of answers with msrvtt
    batch_size = 512
    msrvttqa_label_size = 1001  # vqa voculbary length 1000 + 1 background
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# = add by  for msrvtt qa
@ex.named_config
def task_finetune_msrvttqa():
    exp_name = "finetune_msrvtt_qa"
    video_datasets = ["msrvttqa"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})
    batch_size = 512
    msrvttqa_label_size = 1501  # 1501 / 4540
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1  # 0.1
    draw_false_image = 1
    draw_false_text = 1
    learning_rate = 1e-4  # 1e-4 normal
    val_check_interval = 1.0
    lr_mult = 10



# = for tgif qa on frameqa
@ex.named_config
def task_finetune_tgifqa():
    exp_name = "finetune_tgif_qa"
    video_datasets = ["tgif"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})
    batch_size = 512
    msrvttqa_label_size = 1541  # vqa voculbary length 1540 + 1 background
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# = for tgif qa on action/trans
@ex.named_config
def task_finetune_tgif_action_trans():
    exp_name = "finetune_tgif_action_trans"
    video_datasets = ["tgifqa"]
    image_datasets = []
    loss_names = _loss_names({"mc_vqa": 1})
    batch_size = 512
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    draw_options_text = 5  # 5 choices
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# ========== end: vqa ================


# Task5: ===================== action recognition =====================
@ex.named_config
def task_finetune_action_recognition_hmdb51():
    exp_name = "finetune_action_recognition_hmdb51"
    video_datasets = ["hmdb51"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})  # have
    msrvttqa_label_size = 52  # 51 + 1
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_action_recognition_k400():
    exp_name = "finetune_action_recognition_k400"
    video_datasets = ["k400"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})  # have
    msrvttqa_label_size = 401  # 400 + 1
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 15
    learning_rate = 3e-4
    val_check_interval = 1.0
# end: ===================== action recognition =====================

# ================================ end: finetune ======================
@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def step400k():
    max_epoch = 400
    max_steps = 400000


@ex.named_config
def epoch1():
    max_epoch = 1
    max_steps = None


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12

# ============================= begin: clip_kc pretrain ===================
# = for msrvtt multiple choice
@ex.named_config
def clip_kc_finetune_msrvttchoice():
    exp_name = "clip_kc_finetune_msrvtt_choice"
    video_datasets = ["msrvtt_choice"]
    image_datasets = []
    loss_names = _loss_names({"multiple_choice": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 5 # 5 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "kc"


@ex.named_config
def clip_kc_contrastive_howto_cc3m_choice():
    exp_name = "clip_kc_contrastive_howto_cc3m_choice"
    video_datasets = ["howto100m"]
    image_datasets = ["cc3m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1})
    batch_size = 1024
    max_epoch = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "kc"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc"]
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_kc_contrastive_2plus3_choice():
    exp_name = "clip_kc_contrastive_2plus3_choice"
    video_datasets = ["webvid", "howto100m"]
    image_datasets = ["cc3m", "cc12m", "yfcc15m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1})
    batch_size = 1024
    max_epoch = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "kc"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_kc_contrastive_3plus4_choice():
    exp_name = "clip_kc_contrastive_3plus4_choice"
    video_datasets = ["webvid", "howto100m", "webvid10m"]
    image_datasets = ["cc3m", "cc12m", "yfcc15m", "laion400m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1})
    batch_size = 1024
    max_epoch = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "kc"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_kc_contrastive_cap_3plus4_choice():
    exp_name = "clip_kc_contrastive_3plus4_choice"
    video_datasets = ["webvid", "howto100m", "webvid10m"]
    image_datasets = ["cc3m", "cc12m", "yfcc15m", "laion400m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1, "cap": 1})
    batch_size = 1024
    max_epoch = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "kc"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_kc_new_B16_vtc_cap_3plusM_choice():
    exp_name = "clip_kc_new_L14_vtc_cap_3plusM_choice"
    video_datasets = ["webvid", "howto100m", "webvid10m"]
    image_datasets = ["mix100m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1, "cap": 1})
    per_gpu_batchsize = 32
    num_frames = 8
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-4
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    decay_power = "cosine"
    clip_lr_mult = 0.1
    weight_decay = 0.2
    clip_evl_dropout = 0.0
    clip_cap_decoder_n_layers = 6
    warmup_steps = 4000
    clip_alt_data = True
    image_data_mult = 6
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_kc_new_L14_vtc_cap_3plusM_choice():
    exp_name = "clip_kc_new_L14_vtc_cap_3plusM_choice"
    video_datasets = ["webvid", "howto100m", "webvid10m"]
    image_datasets = ["mix100m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1, "cap": 1})
    per_gpu_batchsize = 14
    num_frames = 8
    max_epoch = 10
    max_text_len = 77
    learning_rate = 8e-5
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-L-14.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    decay_power = "cosine"
    clip_lr_mult = 0.1
    weight_decay = 0.2
    clip_evl_dropout = 0.0
    clip_cap_decoder_n_layers = 6
    warmup_steps = 4000
    clip_alt_data = True
    image_data_mult = 6
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_kc_new_L14_336_vtc_cap_4plusM_choice():
    exp_name = "clip_kc_new_L14_336_vtc_cap_4plusM_choice"
    video_datasets = ["webvid", "howto100m", "webvid10m", "youtube"]
    image_datasets = ["mix100m"]
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    loss_names = _loss_names({"contrastive": 1, "cap": 1})
    image_size = 336
    per_gpu_batchsize = 24
    clip_use_checkpoint = True
    clip_checkpoint_num = [23, 100, 100]
    num_frames = 8
    max_epoch = 2
    max_steps = None
    max_text_len = 77
    learning_rate = 4e-6
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-L-14-336px.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_text = 5
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    decay_power = "cosine"
    weight_decay = 0.2
    clip_evl_dropout = 0.0
    clip_cap_decoder_n_layers = 6
    warmup_steps = 2000
    clip_alt_data = True
    image_data_mult = 6
    val_loss_names = _loss_names({"multiple_choice": 1})
# ============================== end: clip_kc pretrain ====================


@ex.named_config
def clip_finetune_msrvttqa():
    exp_name = "clip_finetune_msrvtt_qa"
    video_datasets = ["msrvttqa"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})
    batch_size = 512
    msrvttqa_label_size = 1501  # 1501 / 4540
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1  # 0.1
    draw_false_image = 1
    draw_false_text = 1
    learning_rate = 1e-4  # 1e-4 normal
    val_check_interval = 1.0
    lr_mult = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "ori"


@ex.named_config
def clip_finetune_tgifqa():
    exp_name = "clip_finetune_tgif_qa"
    video_datasets = ["tgif"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})
    batch_size = 512
    msrvttqa_label_size = 1541  # vqa voculbary length 1540 + 1 background
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "ori"


@ex.named_config
def clip_finetune_msvdqa():
    exp_name = "clip_finetune_msvd_qa"
    video_datasets = ["msvdqa"]
    image_datasets = []
    loss_names = _loss_names({"openend_vqa": 1})  # msvd have same number of answers with msrvtt
    batch_size = 512
    msrvttqa_label_size = 1001  # vqa voculbary length 1000 + 1 background
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "ori"


@ex.named_config
def clip_finetune_zs_k400():
    exp_name = "clip_finetune_zs_k400"
    video_datasets = ["k400_video"]
    image_datasets = []
    loss_names = _loss_names({"zs_classify": 1})
    batch_size = 256
    test_only = True
    max_text_len = 77
    clip = "/mnt/lustre/share_data/likunchang.vendor/code/EVL/ViT-B-16.pt"
    clip_type = "ori"


# ============================== end: clip_kc pretrain ====================
@ex.named_config
def clip_vtc_choice():
    exp_name = "clip_vtc_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1})

    per_gpu_batchsize = 24
    num_frames = 16
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-4
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "ori"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    val_loss_names = _loss_names({"multiple_choice": 1})


@ex.named_config
def clip_vtc_mim_choice():
    exp_name = "clip_vtc_mim_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1, "mim": 1})

    per_gpu_batchsize = 128
    num_frames = 16
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-4
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "ori"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    val_loss_names = _loss_names({"multiple_choice": 1})
    mim_prob = 0.90
    clip_mim_decoder_n_layers = 1


@ex.named_config
def clip_vtc_mim_mlm_choice():
    exp_name = "clip_vtc_mim_mlm_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1, "mim": 1, "mlm": 1})

    per_gpu_batchsize = 128
    num_frames = 16
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-4
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "ori"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    val_loss_names = _loss_names({"multiple_choice": 1})
    mim_prob = 0.90
    clip_mim_decoder_n_layers = 1


@ex.named_config
def clip_vtc_mlm_choice():
    exp_name = "clip_vtc_mlm_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1, "mlm": 1})

    per_gpu_batchsize = 128
    num_frames = 16
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-4
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "ori"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    val_loss_names = _loss_names({"multiple_choice": 1})


# = for msrvtt multiple choice
@ex.named_config
def clip_finetune_msrvttchoice():
    exp_name = "clip_finetune_msrvtt_choice"
    video_datasets = ["webvid10m"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    image_datasets = []
    loss_names = _loss_names({"multiple_choice": 1})
    num_frames = 16
    batch_size = 512
    max_epoch = 10
    warmup_steps = 0.1
    draw_false_text = 5 # 5 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    max_text_len = 77
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "ori"

# ============================== end: clip_kc new nc pretrain ====================
@ex.named_config
def clip_kc_nc_vtc_choice():
    exp_name = "clip_kc_nc_vtc_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1})

    num_frames = 8
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-5
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    clip_freeze_text = True
    val_loss_names = _loss_names({"multiple_choice": 1})
    per_gpu_batchsize = 32
    batch_size = 256


@ex.named_config
def clip_kc_nc_vtc_mim_nd_choice():
    exp_name = "clip_kc_nc_vtc_mim_nd_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1, "mim": 1})

    num_frames = 8
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-5
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    clip_freeze_text = True
    mim_prob = 0.90
    val_loss_names = _loss_names({"multiple_choice": 1})
    per_gpu_batchsize = 32
    batch_size = 256
    clip_mim_decoder_n_layers = 0


@ex.named_config
def clip_kc_nc_vtc_mlm_choice():
    exp_name = "clip_kc_nc_vtc_mlm_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1, "mlm": 1})

    num_frames = 8
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-5
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    clip_freeze_text = True
    val_loss_names = _loss_names({"multiple_choice": 1})
    per_gpu_batchsize = 32
    batch_size = 256
    clip_mim_decoder_n_layers = 0


@ex.named_config
def clip_kc_nc_vtc_mim_nd_mlm_choice():
    exp_name = "clip_kc_nc_vtc_mim_nd_mlm_choice"
    video_datasets = ["webvid10m"]
    image_datasets = []
    train_transform_keys = ["open_clip"]
    val_transform_keys = ["open_clip"]
    val_datasets = ["msrvtt_choice", "lsmdc_choice"]
    loss_names = _loss_names({"contrastive": 1, "mlm": 1, "mim": 1})

    num_frames = 8
    max_epoch = 10
    max_text_len = 77
    learning_rate = 1e-5
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "kc_new"
    vocab_size = 49408
    draw_false_video = 0
    draw_false_text = 5
    decay_power = "cosine"
    weight_decay = 0.2
    warmup_steps = 4000
    clip_freeze_text = True
    val_loss_names = _loss_names({"multiple_choice": 1})
    per_gpu_batchsize = 128
    batch_size = 1024
    clip_mim_decoder_n_layers = 0


# = for msrvtt multiple choice
@ex.named_config
def clip_kc_nc_finetune_msrvttchoice():
    exp_name = "clip_kc_nc_finetune_msrvttchoice"
    video_datasets=["msrvtt_choice", "lsmdc_choice"]
    image_datasets = []
    loss_names = _loss_names({"multiple_choice": 1})
    num_frames = 8
    batch_size = 512
    max_epoch = 10
    warmup_steps = 0.1
    draw_false_text = 5 # 5 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    max_text_len = 77
    clip = "/mnt/petrelfs/share_data/liyizhuo/pretrained/clip_pretrained_models/ViT-B-16.pt"
    clip_type = "kc_new"
    test_only = True