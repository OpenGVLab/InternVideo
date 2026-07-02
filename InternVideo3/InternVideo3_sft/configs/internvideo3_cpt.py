"""Debug config for InternVideo3 SFT - single GPU, small batch size."""
from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
    FSDPConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import InternVideoTokenizeFnConfig
from xtuner.v1.model import InternVideo3Dense8BConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

CONFIG_DIR = os.path.dirname(__file__)


def _default_meta_data_path():
    meta_files = [
        os.path.join(CONFIG_DIR, filename)
        for filename in os.listdir(CONFIG_DIR)
        if filename.endswith(".json")
    ]
    if len(meta_files) != 1:
        raise RuntimeError(
            "Expected exactly one meta JSON in config directory, "
            "or set META_DATA_PATH explicitly."
        )
    return meta_files[0]

# ==================== Path Configuration ====================
# Replace the defaults below with paths for your environment, or set the
# corresponding environment variables before running.
ceph_config = os.environ.get(
    "CEPH_CONFIG", "/path/to/petreloss.conf"
)
meta_data_path = os.environ.get(
    "META_DATA_PATH", _default_meta_data_path()
)
work_dir = os.environ.get(
    "WORK_DIR", "/path/to/save/checkpoints/internvideo3_cpt"
)
tokenizer_cache_dir = os.environ.get(
    "TOKENIZER_CACHE_DIR", "/path/to/tokenizer/cache"
)
load_from = os.environ.get(
    "LOAD_FROM", "/path/to/pretrained/model"
)
processor_path = os.environ.get(
    "PROCESSOR_PATH", "/path/to/tokenizer_or_processor"
)

# ==================== Training Hyperparameters ====================
# These defaults follow the CPT recipe in the reference xtuner config.
sample_max_length = 32768 * 2
pack_max_length = 32768 * 2
num_workers = 8
global_batch_size = 128
total_epoch = 1
hf_interval = 1000
hf_max_keep = 2
checkpoint_interval = 1000
checkpoint_maxkeep = 1
lr = 2e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
loss_reduction = "square"
max_prompt_length = sample_max_length
add_vision_id = True
enable_3d_rope = True

# Frame and visual token settings. Values in the meta JSON can override these
# defaults per dataset. The values below follow InternVideoTokenizeFunction
# defaults, written explicitly here for easier tuning.
min_pixels = 4 * 32 * 32
max_pixels = 16384 * 4 * 32 * 32
video_min_total_pixels = 4 * 4 * 32 * 32
video_max_total_pixels = 32768 * 4 * 32 * 32
video_min_frames = 4
video_max_frames = 768
rand_video_max_frames = 24
fps = 2

# ==================== Model Configuration ====================
model_cfg = InternVideo3Dense8BConfig(
    freeze_vision=False,
    freeze_projector=False,
    freeze_language=False,
    language_model_hf_prefix="model.language_model.",
)

# ==================== Dataset Configuration ====================
oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {
        "dataset": DatasetConfig(
            name=name,
            anno_path=_data["annotation"],
            media_root=_data.get("media_root", ""),
            sample_ratio=_data.get("sample_ratio", 1.0),
            class_name="VLMJsonlDataset",
            enable_sequential_sampler=True,
            cache_tag="cache_tags_v1",
            cache_dir=tokenizer_cache_dir,
        ),
        "tokenize_fn": InternVideoTokenizeFnConfig(
            processor_path=processor_path,
            oss_loader_cfg=oss_loader_cfg,
            max_length=max_prompt_length,
            min_pixels=_data.get("min_pixels", min_pixels),
            max_pixels=_data.get("max_pixels", max_pixels),
            video_min_total_pixels=_data.get(
                "video_min_total_pixels", video_min_total_pixels
            ),
            video_max_total_pixels=_data.get(
                "video_max_total_pixels", video_max_total_pixels
            ),
            video_min_frames=_data.get("video_min_frames", video_min_frames),
            video_max_frames=_data.get("video_max_frames", video_max_frames),
            rand_video_max_frames=_data.get(
                "rand_video_max_frames", rand_video_max_frames
            ),
            fps=_data.get("fps", fps),
            enable_3d_rope=_data.get("enable_3d_rope", enable_3d_rope),
            add_vision_id=_data.get("add_vision_id", add_vision_id),
        ),
    }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level="soft",
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)

# ==================== Optimizer & Scheduler ====================
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)

# ==================== FSDP Configuration ====================
fsdp_cfg = FSDPConfig(
    tp_size=1,
    recompute_ratio=recompute_ratio,
    torch_compile=False,
    cpu_offload=True,
    checkpoint_preserve_rng_state=False,
)

# ==================== Trainer Configuration ====================
trainer = TrainerConfig(
    load_from=load_from,
    auto_resume=True,
    tokenizer_path=load_from,
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    sp_size=1,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    hf_max_keep=hf_max_keep,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    work_dir=work_dir,
)
