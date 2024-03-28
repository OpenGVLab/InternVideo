import torch
import random
import numpy as np
import cv2

cv2.setNumThreads(0)


class Compose(object):
    # Compose different kinds of video transforms into one.
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, videos, target):
        transform_randoms = {}
        for t in self.transforms:
            videos, target, transform_randoms = t((videos, target, transform_randoms))
        return videos, target, transform_randoms

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TemporalCrop(object):
    def __init__(self, frame_num, sample_rate, temporal_jitter=0):
        self.frame_num = frame_num
        self.sample_rate = sample_rate
        self.temporal_jitter = temporal_jitter

    def __call__(self, results):
        clip, target, transform_randoms = results
        # crop the input frames from raw clip
        raw_frame_num = clip.shape[0]

        # determine frame start based on frame_num, sample_rate and the jitter shift.
        frame_start = (raw_frame_num - self.frame_num * self.sample_rate) // 2 + (
                self.sample_rate - 1) // 2 + self.temporal_jitter
        idx = np.arange(frame_start, frame_start + self.frame_num * self.sample_rate, self.sample_rate)
        idx = np.clip(idx, 0, raw_frame_num - 1)

        clip = clip[idx]
        return clip, target, transform_randoms


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        # Calculate output size according to min_size, max_size.
        h, w = image_size
        size = self.min_size

        # # max_size######
        # max_size = self.max_size
        # if max_size is not None:
        #     min_original_size = float(min((w, h)))
        #     max_original_size = float(max((w, h)))
        #     if max_original_size / min_original_size * size > max_size:
        #         size = int(round(max_size * min_original_size / max_original_size))
        # # max_size######

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, results):
        # Input clip should be [TxHxWxC](uint8) ndarray.
        clip, target, transform_randoms = results
        if isinstance(clip, list):
            clip = np.array(clip)
        size = self.get_size(clip.shape[1:3])
        clip_new = np.zeros((clip.shape[0], size[1], size[0], clip.shape[3]), dtype=np.uint8)
        for i in range(clip.shape[0]):
            cv2.resize(clip[i], size, clip_new[i])
        if target is not None:
            target = target.resize(size)
        # Store the size for object box transforms.
        transform_randoms["Resize"] = size
        return clip_new, target, transform_randoms


class RandomClip(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.size = 224  # (w, h)

    def __call__(self, results):
        # Input clip should be [TxHxWxC](uint8) ndarray.
        # size = self.get_size(clip.shape[1:3])
        clip, target, transform_randoms = results
        if self.is_train:
            size = self.size
            clip_new = np.zeros((clip.shape[0], size, size, clip.shape[3]), dtype=np.uint8)

            image_height, image_width = clip.shape[1], clip.shape[2]
            self.tl_x = random.random()
            self.tl_y = random.random()
            x1 = int(self.tl_x * (image_width - size))
            y1 = int(self.tl_y * (image_height - size))
            x2 = x1 + size
            y2 = y1 + size

            for i in range(clip.shape[0]):
                # cv2.resize(clip[i], size, clip_new[i])
                assert clip_new[i].shape == clip[i, y1:y2, x1:x2, :].shape, \
                    print('x1={}, y1={}, x2={}, y2={}, ori_size={}'.format(x1, y1, x2, y2, clip.shape))
                clip_new[i] = clip[i, y1:y2, x1:x2, :]
            if target is not None:
                # target = target.resize(size)
                target = target.crop([x1, y1, x2, y2])
        else:
            clip_new = clip
        # Store the size for object box transforms.
        # transform_randoms["Resize"] = size
        return clip_new, target, transform_randoms


class ColorJitter(object):
    def __init__(self, hue_shift, sat_shift, val_shift):
        # color jitter in hsv space. H: 0~360, circular S: 0.0~1.0 V: 0.0~1.0
        self.hue_bound = int(round(hue_shift / 2))
        self.sat_bound = int(round(sat_shift * 255))
        self.val_bound = int(round(val_shift * 255))

    def __call__(self, results):
        # Convert: RGB->HSV
        clip, target, transform_randoms = results
        clip_hsv = np.zeros_like(clip)
        for i in range(clip.shape[0]):
            cv2.cvtColor(clip[i], cv2.COLOR_RGB2HSV, clip_hsv[i])
        clip_hsv = clip_hsv.astype(np.int32)

        # Jittering.
        hue_s = random.randint(-self.hue_bound, self.hue_bound)
        clip_hsv[..., 0] = (clip_hsv[..., 0] + hue_s + 180) % 180

        sat_s = random.randint(-self.sat_bound, self.sat_bound)
        clip_hsv[..., 1] = np.clip(clip_hsv[..., 1] + sat_s, 0, 255)

        val_s = random.randint(-self.val_bound, self.val_bound)
        clip_hsv[..., 2] = np.clip(clip_hsv[..., 2] + val_s, 0, 255)

        clip_hsv = clip_hsv.astype(np.uint8)

        # Convert: HSV->RGB
        clip = np.zeros_like(clip)
        for i in range(clip.shape[0]):
            cv2.cvtColor(clip_hsv[i], cv2.COLOR_HSV2RGB, clip[i])

        return clip, target, transform_randoms


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        # Input clip should be [TxHxWxC] ndarray.(uint8)
        clip, target, transform_randoms = results
        flip_random = random.random()
        if flip_random < self.prob:
            clip = np.flip(clip, 2)
            if target is not None:
                target = target.transpose(0)

        # Store the random variable for object box transforms
        transform_randoms["Flip"] = flip_random
        return clip, target, transform_randoms


class ToTensor(object):
    def __call__(self, results):
        # Input clip should be [TxHxWxC] ndarray.
        # Convert to [CxTxHxW] tensor.
        clip, target, transform_randoms = results
        return torch.from_numpy(clip.transpose((3, 0, 1, 2)).astype(np.float32)), target, transform_randoms


class Normalize(object):
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def video_normalize(self, tensor, mean, std):
        # Copied from torchvision.transforms.functional.normalize but remove the type check of tensor.
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def __call__(self, results):
        clip, target, transform_randoms = results
        if self.to_bgr:
            clip = clip[[2, 1, 0]]
        # normalize: (x-mean)/std
        clip = self.video_normalize(clip, self.mean, self.std)
        return clip, target, transform_randoms


class SlowFastCrop(object):
    # Class used to split frames for slow pathway and fast pathway.
    def __init__(self, tau, alpha, slow_jitter=False):
        self.tau = tau
        self.alpha = alpha
        self.slow_jitter = slow_jitter

    def __call__(self, results):
        clip, target, transform_randoms = results
        if self.slow_jitter:
            # if jitter, random choose a start
            slow_start = random.randint(0, self.tau - 1)
        else:
            # if no jitter, select the middle
            slow_start = (self.tau - 1) // 2
        slow_clip = clip[:, slow_start::self.tau, :, :]

        fast_stride = self.tau // self.alpha
        fast_start = (fast_stride - 1) // 2
        fast_clip = clip[:, fast_start::fast_stride, :, :]

        return [slow_clip, fast_clip], target, transform_randoms


class Identity(object):
    # Return what is received. Do nothing.
    def __init__(self):
        pass

    def __call__(self, results):
        return results
