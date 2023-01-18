import torch
import torch.nn as nn
import numpy as np
import vlnce_baselines.waypoint_pred.utils as utils

from .transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

class BinaryDistPredictor_TRM(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=12, device=None):
        super(BinaryDistPredictor_TRM, self).__init__()

        self.device = device

        self.num_angles = 120
        self.num_imgs = 12
        self.n_classes = 12  # num of distances
        self.TRM_LAYER = 2
        self.TRM_NEIGHBOR = 1
        self.HEATMAP_OFFSET = 5

        # self.visual_fc_rgb = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(np.prod([2048,7,7]), hidden_dim),
        #     nn.ReLU(True),
        # )
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),
            nn.ReLU(True),
        )
        self.visual_merge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(True),
        )

        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = self.TRM_LAYER
        self.waypoint_TRM = WaypointBert(config=config)

        layer_norm_eps = config.layer_norm_eps
        self.mergefeats_LayerNorm = BertLayerNorm(
            hidden_dim,
            eps=layer_norm_eps
        )
        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=self.TRM_NEIGHBOR).to(self.device)

        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),
        )

    def forward(self, rgb_feats, depth_feats):
        bsi = rgb_feats.size(0) // self.num_imgs

        # rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
        #     bsi, self.num_imgs, -1)
        depth_x = self.visual_fc_depth(depth_feats).reshape(
            bsi, self.num_imgs, -1)
        # vis_x = self.visual_merge(
        #     torch.cat((rgb_x, depth_x), dim=-1)
        # )
        vis_x = depth_x

        attention_mask = self.mask.repeat(bsi,1,1,1)
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        )

        vis_logits = self.vis_classifier(vis_rel_x)
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes)

        # heatmap offset (each image is pointing at the middle)
        vis_logits = torch.cat(
            (vis_logits[:,self.HEATMAP_OFFSET:,:], vis_logits[:,:self.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
