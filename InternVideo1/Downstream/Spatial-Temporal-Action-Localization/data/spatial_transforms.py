"""
Adapted code from:
    @inproceedings{hara3dcnns,
      author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
      title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={6546--6555},
      year={2018},
    }.
"""

import random
import numbers
import collections

import numpy as np
import torch
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import imgaug.augmenters as iaa
import math

from .RandomAugmentBBox import RandomAugmentBBox


class Compose(object):
    """Compose several transforms together.
    Args:
        transforms (list of ``Transform`` objects): List of transforms to compose.
    Example:
        >>> Compose([
        >>>     CenterCrop(10),
        >>>     ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bounding_boxes=None, **kwargs):
        """
        Args:
            img (PIL.Image): Image to be transformed.
        Returns:
            PIL.Image: Transformed image.
        """
        if bounding_boxes is not None:
            for t in self.transforms:
                image, bounding_boxes = t(image, bounding_boxes, **kwargs)
            return image, bounding_boxes
        else:
            for t in self.transforms:
                image = t(image)
            return image

    def randomize_parameters(self, **kwargs):
        for t in self.transforms:
            t.randomize_parameters(**kwargs)

    # def set_k_for_bbox_affine_transform(self, **kwargs):
    #     self.transforms[]

    def __repr__(self):
        return self.__class__.__name__ + '(' + repr(self.transforms) + ')'


class RandomAugment(RandomAugmentBBox):
    def __repr__(self):
        return '{self.__class__.__name__}(aug_type={self.aug_type}, magnitude={self.magnitude})'.format(self=self)


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Convert a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range 
    [0.0, 255.0 / norm_value].
    Args:
        norm_value (float, optional): Normalization constant.
    """

    def __init__(self, norm_value=255.):
        self.norm_value = norm_value

    def __call__(self, pic, bboxes=None, **kwargs):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            if bboxes is not None:
                return img.float().div(self.norm_value), bboxes
            else:
                return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            if bboxes is not None:
                return torch.from_numpy(nppic), bboxes
            else:
                return torch.from_numpy(nppic)
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            if bboxes is not None:
                return img.float().div(self.norm_value), bboxes
            else:
                return img.float().div(self.norm_value)
        else:
            if bboxes is not None:
                return img, bboxes
            else:
                return img

    def randomize_parameters(self, **kwargs):
        pass

    def __repr__(self):
        return '{self.__class__.__name__}(norm_value={self.norm_value})'.format(self=self)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std.
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, bboxes=None, **kwargs):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        if bboxes is not None:
            return tensor, bboxes
        else:
            return tensor

    def randomize_parameters(self, **kwargs):
        pass

    def __repr__(self):
        return '{self.__class__.__name__}(mean={self.mean}, std={self.std})'.format(self=self)


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
        max_ratio (float, optional): If not None, denotes maximum allowed aspect
            ratio after rescaling the input.
    """

    # TO-DO: implement max_ratio with imgaug?
    def __init__(self, resize, interpolation='linear', max_ratio=2.42, fixed_size=False):
        assert isinstance(resize,
                          int) or (isinstance(resize, collections.Iterable) and
                                   len(resize) == 2)
        self.resize = resize
        self.interpolation = interpolation
        self.max_size = int(max_ratio*resize)
        if fixed_size:
            self._resize = iaa.Resize(self.resize, interpolation=interpolation)
        else:
            self._resize = iaa.Resize({"shorter-side": self.resize, "longer-side": "keep-aspect-ratio"}, interpolation=interpolation)
        self._resize_max = iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": self.max_size},  interpolation=interpolation)

    def __call__(self, image, bounding_boxes=None, **kwargs):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        img_aug, boudingbox_aug = self._resize_det(image=image, bounding_boxes=bounding_boxes)

        newh, neww, _ = img_aug.shape

        if max(newh, neww) > self.max_size:
            img_aug, boudingbox_aug = self._resize_det_max(image=image, bounding_boxes=bounding_boxes)

        if bounding_boxes is not None:
            return img_aug, boudingbox_aug
        else:
            return img_aug

    def randomize_parameters(self, **kwargs):
        self._resize_det = self._resize.to_deterministic()
        self._resize_det_max = self._resize_max.to_deterministic()

    def __repr__(self):
        return '{self.__class__.__name__}(resize={self.resize}, interpolation={self.interpolation}, max_size={self.max_size})'.format(self=self)


# TO-DO: re-implement with imgaug
class CenterCrop(object):
    """Crop the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        return [{'transform': 'CenterCrop', 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size})'.format(self=self)


# TO-DO: re-implement with imgaug
class CornerCrop(object):
    """Crop the given PIL.Image at some corner or the center.
    Args:
        size (int): Desired output size of the square crop.
        crop_position (str, optional): Designate the position to be cropped. 
            If is None, a random position will be selected from five choices.
    """

    def __init__(self, size, crop_position=None):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self, param=None):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]
        return [{'transform': 'CornerCrop', 'crop_position': self.crop_position,
                 'size': self.size}]

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, crop_position={self.crop_position})'.format(self=self)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a given probability.
    Args:
        p (float, optional): Probability of flipping.
    """

    def __init__(self, p=0.5):
        self.prob = p
        self._flip = iaa.Fliplr(self.prob)

    def __call__(self, image, bounding_boxes=None, **kwargs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        img_aug, bboxes_aug = self._flip_det(image=image, bounding_boxes=bounding_boxes)
        if bounding_boxes is not None:
            return img_aug, bboxes_aug
        else:
            return img_aug

    def randomize_parameters(self, **kwargs):
        self._flip_det = self._flip.to_deterministic()

    def __repr__(self):
        return '{self.__class__.__name__}(prob={self.prob})'.format(self=self)


# TO-DO: re-implement with imgaug
class ScaleJitteringRandomCrop(object):
    """Randomly rescale the given PIL.Image and then take a random crop.
    Args:
        min_scale (int): Minimum scale for random rescaling.
        max_scale (int): Maximum scale for random rescaling.
        size (int): Desired output size of the square crop.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    """

    def __init__(self, min_size, max_size, size, interpolation='linear'):
        self.min_size = min_size
        self.max_size = max_size
        self.size = size
        self.interpolation = interpolation
        self._resize = iaa.Resize({"shorter-side": (self.min_size, self.max_size), "longer-side": "keep-aspect-ratio"}, interpolation=interpolation)
        self._crop = iaa.CropToFixedSize(width=self.size, height=self.size)

    def __call__(self, image, bounding_boxes=None, **kwargs):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        img_aug, boudingbox_aug = self._resize_det(image=image, bounding_boxes=bounding_boxes)
        img_aug, boudingbox_aug = self._crop_det(image=img_aug, bounding_boxes=boudingbox_aug)

        if bounding_boxes is not None:
            return img_aug, boudingbox_aug
        else:
            return img_aug

    def randomize_parameters(self, **kwargs):
        self._resize_det = self._resize.to_deterministic()
        self._crop_det = self._crop.to_deterministic()

    def __repr__(self):
        return '{self.__class__.__name__}(min_size={self.min_size}, max_size={self.max_size}, size={self.size}, interpolation={self.interpolation})'.format(self=self)


class CropBBoxRescale(object):
    """Crop region within the bbox and resize to fixed size.
    Args:
        min_scale (int): Minimum scale for random rescaling.
        max_scale (int): Maximum scale for random rescaling.
        size (int): Desired output size of the square crop.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    """

    def __init__(self, resize, bbox_scales=None, interpolation='linear'):
        if bbox_scales is None:
            bbox_scales = [1, 1.5]
        self.size = resize
        self.interpolation = interpolation
        self._resize = iaa.Resize(resize, interpolation=interpolation)
        self.scale_list = bbox_scales

    def __call__(self, image, bounding_boxes=None, **kwargs):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        h, w, _ = image.shape

        bbox = bounding_boxes[0]
        bbox = scale_box(bbox, self.scale)
        bbox = clip_box_to_image(bbox, h, w)
        x1, y1, x2, y2 = round_down(bbox[0]), round_down(bbox[1]), round_down(bbox[2]), round_down(bbox[3])
        # Crop region inside the bbox
        img_cropped = image[y1:y2, x1:x2,:]
        # Resize cropped image to fixed size
        img_aug = self._resize_det(image=img_cropped)

        # if bounding_boxes is not None:
        #     return img_aug, boudingbox_aug
        # else:

        return img_aug, bounding_boxes

    def randomize_parameters(self, **kwargs):
        self.scale = random.uniform(self.scale_list[0], self.scale_list[1])
        self._resize_det = self._resize.to_deterministic()

    def __repr__(self):
        return '{self.__class__.__name__}(size={self.size}, interpolation={self.interpolation})'.format(self=self)


def round_down(x):
    return int(math.floor(x))


def clip_box_to_image(box, height, width):
    """Clip an box to an image with the given height and width."""
    box[[0, 2]] = np.minimum(width - 1., np.maximum(0., box[[0, 2]]))
    box[[1, 3]] = np.minimum(height - 1., np.maximum(0., box[[1, 3]]))
    return box


def scale_box(box, scale):
    """
    box: (4,)  [w1, h1, w2, h2]
    """
    w1, h1, w2, h2 = box.x1, box.y1, box.x2, box.y2
    center_w = (w1 + w2) / 2
    center_h = (h1 + h2) / 2

    half_w = center_w - w1
    half_h = center_h - h1

    half_w = scale * half_w
    half_h = scale * half_h

    new_box = np.array([center_w-half_w, center_h-half_h,
                        center_w+half_w, center_h+half_h])

    return new_box