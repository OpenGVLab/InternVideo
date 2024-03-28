from CoTrain.transforms.image.pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
    open_clip_transform,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "open_clip": open_clip_transform,
}


def keys_to_transforms(keys: list, size=224, mode="train"):
    return [_transforms[key](size=size, mode=mode) for key in keys]
