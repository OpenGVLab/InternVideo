from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment


def pixelbert_transform(size=800, mode="train"):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def pixelbert_transform_randaug(size=800, mode="train"):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def open_clip_transform(size=224, mode="train"):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=input_mean, std=input_std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=input_mean, std=input_std),
            ]
        )
