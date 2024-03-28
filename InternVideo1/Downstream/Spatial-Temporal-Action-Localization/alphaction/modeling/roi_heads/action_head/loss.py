import torch
from alphaction.layers import SigmoidFocalLoss, SoftmaxFocalLoss
from alphaction.modeling.utils import cat


class ActionLossComputation(object):
    def __init__(self, cfg):
        self.proposal_per_clip = cfg.MODEL.ROI_ACTION_HEAD.PROPOSAL_PER_CLIP
        self.num_pose = cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES
        self.num_object = cfg.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES
        self.num_person = cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES

        self.weight_dict = dict(
            loss_pose_action = cfg.MODEL.ROI_ACTION_HEAD.POSE_LOSS_WEIGHT,
            loss_object_interaction = cfg.MODEL.ROI_ACTION_HEAD.OBJECT_LOSS_WEIGHT,
            loss_person_interaction = cfg.MODEL.ROI_ACTION_HEAD.PERSON_LOSS_WEIGHT,
        )

        gamma = cfg.MODEL.ROI_ACTION_HEAD.FOCAL_LOSS.GAMMA
        alpha = cfg.MODEL.ROI_ACTION_HEAD.FOCAL_LOSS.ALPHA
        self.sigmoid_focal_loss = SigmoidFocalLoss(gamma, alpha, reduction="none")
        self.softmax_focal_loss = SoftmaxFocalLoss(gamma, alpha, reduction="sum")

    def sample_box(self, boxes):
        proposals = []
        num_proposals = self.proposal_per_clip
        for boxes_per_image in boxes:
            num_boxes = len(boxes_per_image)

            if num_boxes > num_proposals:
                choice_inds = torch.randperm(num_boxes)[:num_proposals]
                proposals_per_image = boxes_per_image[choice_inds]
            else:
                proposals_per_image = boxes_per_image
            proposals_per_image = proposals_per_image.random_aug(0.2, 0.1, 0.1, 0.05)
            proposals.append(proposals_per_image)
        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, avg_box_num):
        class_logits = cat(class_logits, dim=0)
        assert class_logits.shape[1] == (self.num_pose + self.num_object + self.num_person), \
            "The shape of tensor class logits doesn't match total number of action classes."

        if not hasattr(self, "_proposals"):
            raise RuntimeError("sample_box needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        assert class_logits.shape[1] == labels.shape[1], \
            "The shape of tensor class logits doesn't match the label tensor."

        loss_dict = {}

        if self.num_pose > 0:
            pose_label = labels[:, :self.num_pose].argmax(dim=1)
            pose_logits = class_logits[:, :self.num_pose]
            pose_loss = self.softmax_focal_loss(pose_logits, pose_label) / avg_box_num
            loss_dict["loss_pose_action"] = pose_loss

        interaction_label = labels[:, self.num_pose:].to(dtype=torch.float32)
        object_label = interaction_label[:, :self.num_object]
        person_label = interaction_label[:, self.num_object:]

        interaction_logits = class_logits[:, self.num_pose:]
        object_logits = interaction_logits[:, :self.num_object]
        person_logits = interaction_logits[:, self.num_object:]

        if self.num_object > 0:
            object_loss = self.sigmoid_focal_loss(object_logits, object_label).mean(dim=1).sum() / avg_box_num
            loss_dict["loss_object_interaction"] = object_loss
        if self.num_person > 0:
            person_loss = self.sigmoid_focal_loss(person_logits, person_label).mean(dim=1).sum() / avg_box_num
            loss_dict["loss_person_interaction"] = person_loss

        return loss_dict, self.weight_dict


def make_roi_action_loss_evaluator(cfg):
    loss_evaluator = ActionLossComputation(cfg)

    return loss_evaluator