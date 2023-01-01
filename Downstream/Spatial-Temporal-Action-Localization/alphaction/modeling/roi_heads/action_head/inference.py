import torch
from torch import nn
import torch.nn.functional as F

from alphaction.structures.bounding_box import BoxList


class PostProcessor(nn.Module):
    def __init__(self, pose_action_num):
        super(PostProcessor, self).__init__()
        self.pose_action_num = pose_action_num

    def forward(self, x, boxes):
        # boxes should be (#detections,4)
        # prob should be calculated in different way.
        class_logits, = x
        pose_action_prob = F.softmax(class_logits[:,:self.pose_action_num],-1)
        interaction_action_prob = torch.sigmoid(class_logits[:,self.pose_action_num:])

        action_prob = torch.cat((pose_action_prob,interaction_action_prob),1)

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        box_tensors = [a.bbox for a in boxes]

        action_prob = action_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_image, image_shape in zip(
                action_prob, box_tensors, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_image, prob, image_shape)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist


def make_roi_action_post_processor(cfg):
    softmax_num = cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES
    postprocessor = PostProcessor(softmax_num)
    return postprocessor
