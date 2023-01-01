from torch import nn
from alphaction.modeling import registry


@registry.ROI_ACTION_PREDICTORS.register("FCPredictor")
class FCPredictor(nn.Module):
    def __init__(self, config, dim_in):
        super(FCPredictor, self).__init__()

        num_classes = config.MODEL.ROI_ACTION_HEAD.NUM_CLASSES

        dropout_rate = config.MODEL.ROI_ACTION_HEAD.DROPOUT_RATE
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=True)

        self.cls_score = nn.Linear(dim_in, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        scores = self.cls_score(x)

        return scores

    def c2_weight_mapping(self):
        return {"cls_score.weight": "pred_w",
                "cls_score.bias": "pred_b"}


def make_roi_action_predictor(cfg, dim_in):
    func = registry.ROI_ACTION_PREDICTORS[cfg.MODEL.ROI_ACTION_HEAD.PREDICTOR]
    return func(cfg, dim_in)
