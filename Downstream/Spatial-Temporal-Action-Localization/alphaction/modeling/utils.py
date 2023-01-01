"""
Miscellaneous utility functions
"""

import torch
from alphaction.structures.bounding_box import BoxList


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def pad_sequence(sequence, targ_size, padding_value=0):
    tensor_size = sequence[0].size()
    trailing_dims = tensor_size[1:]
    out_dims = (len(sequence), targ_size) + trailing_dims

    out_tensor = sequence[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequence):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor

    return out_tensor

def prepare_pooled_feature(x_pooled, boxes, detach=True):
    image_shapes = [box.size for box in boxes]
    boxes_per_image = [len(box) for box in boxes]
    box_tensors = [a.bbox for a in boxes]

    if detach:
        x_pooled = x_pooled.detach()
    pooled_feature = x_pooled.split(boxes_per_image, dim=0)

    boxes_result = []
    for feature_per_image, boxes_per_image, image_shape in zip(
            pooled_feature, box_tensors, image_shapes
    ):
        boxlist = BoxList(boxes_per_image, image_shape, mode="xyxy")
        boxlist.add_field("pooled_feature", feature_per_image)
        boxes_result.append(boxlist)
    return boxes_result