# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Object detection strong augmentation utilities."""
# pylint: disable=g-explicit-length-test

import data.augmentations as augmentations
# from FasterRCNN.utils import np_box_ops

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmenters.geometric import Affine
import numpy as np
import torch
from PIL import Image
import cv2
# from tensorpack.utils import logger


RANDOM_COLOR_POLICY_OPS = (
    'Identity',
    'AutoContrast',
    'Equalize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'Posterize',
)

# for image size == 800, 0.1 is 80.
CUTOUT = iaa.Cutout(nb_iterations=(1, 5), size=[0, 0.2], squared=True)

DEGREE = 30
AFFINE_TRANSFORM = iaa.Sequential(
    [
        iaa.OneOf([
            Affine(  # TranslateX
                translate_percent={'x': (-0.1, 0.1)},
                order=[0, 1],
                cval=125,
            ),
            Affine(  # TranslateY
                translate_percent={'y': (-0.1, 0.1)},
                order=[0, 1],
                cval=125,
            ),
            Affine(  # Rotate
                rotate=(-DEGREE, DEGREE),
                order=[0, 1],
                cval=125,
            ),
            Affine(  # ShearX and ShareY
                shear=(-DEGREE, DEGREE),
                order=[0, 1],
                cval=125,
            ),
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True)

# AFFINE_TRANSFORM = iaa.Sequential(
#     [
#         iaa.OneOf([
#             Affine(  # TranslateX
#                 translate_percent={'x': (-0.1, 0.1)},
#                 order=[0, 1],
#                 cval=125,
#             )
#         ])
#     ],
#     # do all of the above augmentations in random order
#     random_order=True)

AFFINE_TRANSFORM_WEAK = iaa.Sequential(
    [
        iaa.OneOf([
            Affine(
                translate_percent={'x': (-0.05, 0.05)},
                order=[0, 1],
                cval=125,
            ),
            Affine(
                translate_percent={'y': (-0.05, 0.05)},
                order=[0, 1],
                cval=125,
            ),
            Affine(
                rotate=(-10, 10),
                order=[0, 1],
                cval=125,
            ),
            Affine(
                shear=(-10, 10),
                order=[0, 1],
                cval=125,
            ),
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True)

COLOR = iaa.Sequential(
    [
        iaa.OneOf(  # apply one color transformation
            [
                iaa.Add((0, 0)),  # identity transform
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),
                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                # Improve or worsen the contrast of images.
                iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
            ])
    ],
    random_order=True)


def bb_to_array(bbs):
    coords = []
    for bb in range(len(bbs)):
        coord = list(bbs[bb]._split_into_xyxy())
        coords.append([i.squeeze().numpy() for i in coord])
    coords = np.array(coords)
    return coords

def siglebb_to_array(bbs):
    coords = []
    for bb in bbs:
        coords.append([bb.x1, bb.y1, bb.x2, bb.y2])
    # coords = np.array(coords)
    return coords

def allbb_to_array(bbs):
    coords = []
    for bb in bbs:
        coords.append(array_to_bb(bb_to_array(bb)))
    return coords

def array_to_bb(coords):
    # convert to bbs
    bbs = []
    for b in coords:
        bbs.append(ia.BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
    return bbs


class RandomAugmentBBox(object):
    """Augmentation class."""

    def __init__(self,
                 is_train = True,
                 aug_type='strong',
                 magnitude=10,
                 weighted_inbox_selection=False):
        self.affine_aug_op = AFFINE_TRANSFORM
        # for inbox affine, we use small degree
        self.inbox_affine_aug_op = AFFINE_TRANSFORM_WEAK
        self.jitter_aug_op = COLOR
        self.cutout_op = CUTOUT
        self.magnitude = magnitude
        self.aug_type = aug_type
        self.is_train = is_train
        self.weighted_inbox_selection = weighted_inbox_selection

        # self.augment_fn is a list of list (i.g. [[],...]), each item is a list of
        # augmentation will be randomly picked up. Th total items determine the
        # number of layers of augmentation to apply
        # Note: put cutout_augment at last if used.
        if aug_type == 'strong':
            # followd by cutout
            self.augment_fn = [[self.color_augment],
                               [self.bbox_affine_transform, self.affine_transform],
                               [self.cutout_augment]]
            # self.augment_fn = [[self.bbox_affine_transform]]
        elif aug_type == 'default':
            self.augment_fn = [[self.color_augment],
                               [self.affine_transform]]
        elif aug_type == 'default_w_cutout':
            self.augment_fn = [[self.color_augment],
                               [self.affine_transform],
                               [self.cutout_augment]]
        elif aug_type == 'default_wo_affine':
            self.augment_fn = [[self.color_augment],
                               [self.affine_transform],
                               [self.cutout_augment]]
        else:
            raise NotImplementedError('aug_type {} does not exist'.format(aug_type))
        self.randomize_parameters()
        # logger.info('-' * 100)
        # logger.info('Augmentation type {}: {}'.format(aug_type, self.augment_fn))
        # logger.info('-' * 100)

    def normaize(self, x):
        x = x / 255.0
        x = x / 0.5 - 1.0
        return x

    def unnormaize(self, x):
        x = (x + 1.0) * 0.5 * 255.0
        return x

    def numpy_apply_policies(self, arglist):
        x, policies = arglist
        re = []
        for y, policy in zip(x, policies):
            # apply_policy has input to have images [-1, 1]
            y_a = augmentations.apply_policy(policy, self.normaize(y))
            y_a = np.reshape(y_a, y.shape)
            y_a = self.unnormaize(y_a)
            re.append(y_a)
        return np.stack(re).astype('f')

    def bbox_affine_transform(self, results, **kwargs):
        """In-box affine transformation.,这里是将图片中的一些box部分图片裁剪下来,仅仅对裁剪下来的部分进行变换""" 
        images, bounding_boxes,transform_randoms = results
        real_box_n = [len(i) for i in bounding_boxes]
        if isinstance(images, np.ndarray):
            images = np.split(images, len(images), axis=0)
            images = [i.squeeze() for i in images]
        shape = images[0].shape
        copybounding_boxes = [bounding_boxes for _ in range(len(images))]
        for im, boxes in zip(images, copybounding_boxes):
            boxes = bb_to_array(boxes)
            # large area has better probability to be sampled
            if self.weighted_inbox_selection:
                area = np_box_ops.area(boxes[:real_box_n])
                k = np.random.choice([i for i in range(real_box_n)],
                                     1,
                                     p=area / area.sum())[0]
            else:
                k = self.k_det

            if len(boxes) > 0:
                # import pdb
                # pdb.set_trace()
                box = boxes[k]
                im_crop = im[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                im_paste = self.inbox_affine_aug_op_det(images=[im_crop])[0]
                # in-memory operation
                im[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = im_paste
        assert shape == images[0].shape
        images = np.array(images)
        return images, bounding_boxes, transform_randoms

    def affine_transform(self, results):
        """Global affine transformation."""
        images, bounding_boxes,transform_randoms = results
        if isinstance(images, np.ndarray):
            images = np.split(images, len(images), axis=0)
            images = [i.squeeze() for i in images]
        shape = images[0].shape
        bounding_boxes = [bounding_boxes for _ in range(len(images))]
        ori_bbx = allbb_to_array(bounding_boxes)
        images_aug, bbs_aug = self.affine_aug_op_det(
            images=images, bounding_boxes=ori_bbx)
        for i in range(len(bbs_aug)):
            new_array = siglebb_to_array(bbs_aug[i])
            new_array = torch.as_tensor(new_array, dtype=torch.float32).reshape(-1, 4)
            assert new_array.shape == bounding_boxes[i].bbox.shape
            bounding_boxes[i].update_box(new_array,"xyxy")
            break
        assert shape == images_aug[0].shape
        images_aug = np.array(images_aug)
        return images_aug, bounding_boxes[0], transform_randoms

    def jitter_augment(self, results):
        """Color jitters."""
        images, bounding_boxes,transform_randoms = results
        images_aug = self.jitter_aug_op(images=images)
        return images_aug, bounding_boxes, transform_randoms

    def cutout_augment(self, results):
        """Cutout augmentation."""
        images, bounding_boxes,transform_randoms = results
        images_aug = self.cutout_op_det(images=images)
        return images_aug, bounding_boxes, transform_randoms

    def color_augment(self, results, p=1.0, **kwargs):
        """RandAug color augmentation."""
        """颜色增强,不会改变box,images:List[np.ndarray]"""
        del kwargs
        images, bounding_boxes,transform_randoms = results
        policy = lambda: [(op, p, self.magnitude_det)  # pylint: disable=g-long-lambda
                          for op in self.color_policies_det]
        images_aug = []
        shape = images[0].shape
        for x in range(len(images)):
            images_aug.append(
                self.numpy_apply_policies((images[x:x + 1], [policy()]))[0])
        assert shape == images_aug[0].shape
        if bounding_boxes is None:
            return images_aug
        images_aug = np.array(images_aug)
        return images_aug, bounding_boxes, transform_randoms

    def set_k_for_bbox_affine_transform(self, **kwargs):
        real_box_n = kwargs['n_real_box']
        self.k_det = np.random.choice([i for i in range(real_box_n)], 1)[0]

    def randomize_parameters(self, **kwargs):
        self.cutout_op_det = self.cutout_op.to_deterministic()
        self.affine_aug_op_det = self.affine_aug_op.to_deterministic()
        self.inbox_affine_aug_op_det = self.inbox_affine_aug_op.to_deterministic()
        self.magnitude_det = np.random.randint(1, self.magnitude)
        self.color_policies_det = np.random.choice(RANDOM_COLOR_POLICY_OPS, 1)
        # if 'n_real_box' in kwargs:
        real_box_n = kwargs['n_real_box'] if 'n_real_box' in kwargs else 1
        self.k_det = np.random.choice([i for i in range(real_box_n)], 1)[0]
        if len(self.augment_fn) > 0 and self.augment_fn[-1][0].__name__ == 'cutout_augment':
            naug = len(self.augment_fn)
            self.oder1_det = np.random.permutation(np.arange(naug - 1))
        else:
            self.order2_det = np.random.permutation(np.arange(len(self.augment_fn)))
        self.tmp = np.random.randint(0, 2)

    def __call__(self, results, **kwargs):
        if self.is_train:
            images, bounding_boxes,transform_randoms = results
            # random order
            if len(self.augment_fn) > 0 and self.augment_fn[-1][0].__name__ == 'cutout_augment':
                # put cutout in the last always
                naug = len(self.augment_fn)
                order = self.oder1_det
                order = np.concatenate([order, [naug - 1]], 0)
            else:
                order = self.order2_det

            # pylint: disable=invalid-name
            T = None
            for i in order:
                fns = self.augment_fn[i]
                # fn = fns[np.random.randint(0, len(fns))]
                if len(fns) > 1:
                    fn = fns[self.tmp]
                else:
                    fn = fns[0]
                images, bounding_boxes, _T = fn(
                    results=(images, bounding_boxes, transform_randoms), **kwargs)
                if _T is not None:
                    T = _T
            return images, bounding_boxes, T
        else:
            return results

if __name__ == '__main__':
    import sys
    sys.path.append('/mnt/cache/xingsen/VideoMAE_ava_ft3/')
    img = np.random.randint(0, 255, (2, 100, 100, 3), dtype=np.uint8)
    img = np.split(img,img.shape[0],axis=0)
    img = [np.array(Image.open('/mnt/cache/xingsen/VideoMAE_ava_ft3/data/1.jpg'))]
    for i in range(len(img)):
        img[i] = img[i].squeeze()
    print(type(img[0]))
    from alphaction.structures.bounding_box import BoxList

    im_w, im_h = 100, 100
    n = 1
    xy = np.array([16,11]).reshape(1,2)#np.zeros([n,2])#np.uint8(np.random.random([n, 2]) * 50 + 30)
    w = np.uint8(np.ones([n, 1]) * 53)
    h = np.uint8(np.ones([n, 1]) * 62)
    # cv2.rectangle(img[0], (16,11), (69,73), (0,255,0), 4)
    # cv2.imwrite('test.jpg',img[0])
    boxes = np.concatenate([xy, w, h], axis=1)
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
    boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")
    xmin, ymin, xmax, ymax = boxes[0]._split_into_xyxy()
    aug = RandomAugmentBBox()
    aug.randomize_parameters()
    images_aug, bbs_aug, _ = aug((img, [boxes], None))
    print(images_aug,bbs_aug)
    # x1,y1,x2,y2 = bbs_aug[0][0].x1,bbs_aug[0][0].y1,bbs_aug[0][0].x2,bbs_aug[0][0].y2
    # if x1 < 0:
    #     x1 = 0
    # if y1 < 0:
    #     y1 = 0
    # cv2.rectangle(images_aug[0], (x1,y1), (x2,y2), (0,255,0), 4)
    # cv2.imwrite('test2.jpg',images_aug[0])