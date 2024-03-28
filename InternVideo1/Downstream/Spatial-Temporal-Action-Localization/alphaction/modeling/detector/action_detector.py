from torch import nn

from ..backbone import build_backbone
from ..roi_heads.roi_heads_3d import build_3d_roi_heads


class ActionDetector(nn.Module):
    def __init__(self, cfg):
        super(ActionDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_3d_roi_heads(cfg, self.backbone.dim_out)

    def forward(self, slow_video, fast_video, boxes, objects=None, extras={}, part_forward=-1):
        # part_forward is used to split this model into two parts.
        # if part_forward<0, just use it as a single model
        # if part_forward=0, use this model to extract pooled feature(person and object, no memory features).
        # if part_forward=1, use the ia structure to aggregate interactions and give final result.
        # implemented in roi_heads

        if part_forward==1:
            slow_features = fast_features = None
        else:
            slow_features, fast_features = self.backbone(slow_video, fast_video)

        result, detector_losses, loss_weight, detector_metrics = self.roi_heads(slow_features, fast_features, boxes, objects, extras, part_forward)

        if self.training:
            return detector_losses, loss_weight, detector_metrics, result

        return result

    def c2_weight_mapping(self):
        if not hasattr(self, "c2_mapping"):
            weight_map = {}
            for name, m_child in self.named_children():
                if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                    child_map = m_child.c2_weight_mapping()
                    for key, val in child_map.items():
                        new_key = name + '.' + key
                        weight_map[new_key] = val
            self.c2_mapping = weight_map
        return self.c2_mapping

def build_detection_model(cfg):
    return ActionDetector(cfg)