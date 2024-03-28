from .vit import build_vit
from .vit_clean import build_vit_clean

from .clip import clip_b16, clip_l14, clip_l14_336
from .clip_vision import clip_vision_b16, clip_vision_l14, clip_vision_l14_336
from .clip_text import clip_text_b16, clip_text_l14, clip_text_l14_336


def build_clip(config):
    model_cls = config.vision_encoder.clip_teacher
    model = eval(model_cls)(
        input_resolution = config.vision_encoder.clip_img_size,
        clip_return_layer=config.vision_encoder.clip_return_layer,
        clip_return_interval=config.vision_encoder.clip_return_interval,
    )
    return model


def build_text_clip(clip_teacher):
    model = eval(clip_teacher)()
    return model