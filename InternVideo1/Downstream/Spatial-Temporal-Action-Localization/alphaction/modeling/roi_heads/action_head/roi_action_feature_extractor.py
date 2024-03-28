import torch
from torch import nn
from torch.nn import functional as F

from alphaction.modeling import registry
from alphaction.modeling.poolers import make_3d_pooler
from alphaction.modeling.roi_heads.action_head.IA_structure import make_ia_structure
from alphaction.modeling.utils import cat, pad_sequence, prepare_pooled_feature
from alphaction.utils.IA_helper import has_object


@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD

        self.pooler = make_3d_pooler(head_cfg)

        resolution = head_cfg.POOLER_RESOLUTION

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))

        if config.MODEL.IA_STRUCTURE.ACTIVE:
            self.max_feature_len_per_sec = config.MODEL.IA_STRUCTURE.MAX_PER_SEC

            self.ia_structure = make_ia_structure(config, dim_in)

        representation_size = head_cfg.MLP_HEAD_DIM

        fc1_dim_in = dim_in
        if config.MODEL.IA_STRUCTURE.ACTIVE and (config.MODEL.IA_STRUCTURE.FUSION == "concat"):
            fc1_dim_in += config.MODEL.IA_STRUCTURE.DIM_OUT

        self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        self.dim_out = representation_size

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_x = slow_x.mean(dim=2, keepdim=True)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_x = fast_x.mean(dim=2, keepdim=True)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def max_pooling_zero_safe(self, x):
        if x.size(0) == 0:
            _, c, t, h, w = x.size()
            res = self.config.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION
            x = torch.zeros((0, c, 1, h - res + 1, w - res + 1), device=x.device)
        else:
            x = self.max_pooler(x)
        return x

    def forward(self, slow_features, fast_features, proposals, objects=None, extras={}, part_forward=-1):
        ia_active = hasattr(self, "ia_structure")
        if part_forward == 1:
            person_pooled = cat([box.get_field("pooled_feature") for box in proposals])
            if objects is None:
                object_pooled = None
            else:
                object_pooled = cat([box.get_field("pooled_feature") for box in objects])
        else:
            x = self.roi_pooling(slow_features, fast_features, proposals)

            person_pooled = self.max_pooler(x)

            if has_object(self.config.MODEL.IA_STRUCTURE):
                object_pooled = self.roi_pooling(slow_features, fast_features, objects)
                object_pooled = self.max_pooling_zero_safe(object_pooled)
            else:
                object_pooled = None

        if part_forward == 0:
            return None, person_pooled, object_pooled

        x_after = person_pooled

        if ia_active:
            tsfmr = self.ia_structure
            mem_len = self.config.MODEL.IA_STRUCTURE.LENGTH
            mem_rate = self.config.MODEL.IA_STRUCTURE.MEMORY_RATE
            use_penalty = self.config.MODEL.IA_STRUCTURE.PENALTY
            memory_person, memory_person_boxes = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                                                       self.max_feature_len_per_sec, tsfmr.dim_others,
                                                                       person_pooled, proposals, use_penalty)

            ia_feature = self.ia_structure(person_pooled, proposals, object_pooled, objects, memory_person, )
            x_after = self.fusion(x_after, ia_feature, self.config.MODEL.IA_STRUCTURE.FUSION)

        x_after = x_after.view(x_after.size(0), -1)

        x_after = F.relu(self.fc1(x_after))
        x_after = F.relu(self.fc2(x_after))

        return x_after, person_pooled, object_pooled

    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, fixed_dim, current_x, current_box, use_penalty):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        if use_penalty and self.training:
            cur_loss = extras["cur_loss"]
        else:
            cur_loss = 0.0
        current_feat = prepare_pooled_feature(current_x, current_box, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = feature_pool[movie_id]
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                   for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                  for mem_ind in after_inds]
            mem_box_current = [self.sample_mem_feature(new_feat, max_boxes), ]
            mem_box_list = mem_box_list_before + mem_box_current + mem_box_list_after
            mem_feature_list += [box_list.get_field("pooled_feature")
                                 if box_list is not None
                                 else torch.zeros(0, fixed_dim, 1, 1, 1, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]

        seq_length = sum(mem_len) + 1
        person_per_seq = seq_length * max_boxes
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, fixed_dim, 1, 1, 1)
        mem_feature = mem_feature.to(device)
        mem_pos = pad_sequence(mem_pos_list, max_boxes)
        mem_pos = mem_pos.view(-1, person_per_seq, 4)
        mem_pos = mem_pos.to(device)

        return mem_feature, mem_pos

    def check_fetch_mem_feature(self, movie_cache, mem_ind, max_num, cur_loss, use_penalty):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        box_list = self.sample_mem_feature(box_list, max_num)
        if use_penalty and self.training:
            loss_tag = box_list.delete_field("loss_tag")
            penalty = loss_tag / cur_loss if loss_tag < cur_loss else cur_loss / loss_tag
            features = box_list.get_field("pooled_feature") * penalty
            box_list.add_field("pooled_feature", features)
        return box_list

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError


def make_roi_action_feature_extractor(cfg, dim_in):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, dim_in)
