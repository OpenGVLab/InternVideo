import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.utils.common import CustomFixedCategorical

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.image_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    length2mask, angle_feature, dir_angle_feature)
import math


@baseline_registry.register_policy
class PolicyViewSelectionCMA(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class CMANet(Net):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    CMA state encoder
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions
    ):
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.backbone_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.backbone_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.backbone_type in [
            "TorchVisionResNet152", "TorchVisionResNet50"
        ], "RGB_ENCODER.backbone_type must be TorchVisionResNet152 or TorchVisionResNet50"
        if model_config.RGB_ENCODER.backbone_type == "TorchVisionResNet50":
            self.rgb_encoder = TorchVisionResNet50(
                observation_space,
                model_config.RGB_ENCODER.output_size,
                device,
                spatial_output=model_config.spatial_output,
            )

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        # merging visual inputs
        self.rgb_linear = nn.Sequential(
            nn.Linear(
                2048,
                model_config.RGB_ENCODER.output_size,          # 256
            ),
            nn.ReLU(True),
        )
        if self.depth_encoder.spatial_output:
            None
        else:
            self.depth_linear = nn.Sequential(
                nn.Linear(
                    128,        # 128
                    model_config.DEPTH_ENCODER.output_size,    # 128
                ),
                nn.ReLU(True),
            )

        self.vismerge_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.output_size + model_config.RGB_ENCODER.output_size + model_config.VISUAL_DIM.directional,
                model_config.VISUAL_DIM.vis_hidden,
            ),
            nn.ReLU(True),
        )

        self.enc_prev_act = nn.Sequential(
            nn.Linear(model_config.VISUAL_DIM.directional, model_config.VISUAL_DIM.directional),
            nn.Tanh(),
        )

        # Init the RNN state decoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=model_config.VISUAL_DIM.vis_hidden + model_config.VISUAL_DIM.directional,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self.prev_state_vis_attn = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.VISUAL_DIM.vis_hidden,
            output_tilde=False
        )

        self.text_vis_attn = SoftDotAttention(
            self.instruction_encoder.output_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.VISUAL_DIM.vis_hidden,
            output_tilde=False
        )

        self.state_text_attn = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size,
            self.instruction_encoder.output_size,
            self.instruction_encoder.output_size,
            output_tilde=False
        )

        self.state_vis_logits = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size+model_config.VISUAL_DIM.vis_hidden+self.instruction_encoder.output_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.STATE_ENCODER.hidden_size,
            output_tilde=False
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.space_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(start_dim=2),)

        # self.critic = nn.Sequential(
        #     nn.Linear(model_config.STATE_ENCODER.hidden_size, model_config.STATE_ENCODER.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(model_config.STATE_ENCODER.hidden_size, 1),
        # )
        # self.drop = nn.Dropout(p=0.50)
        # self.drop_env = nn.Dropout(p=0.40)

        self.train()
        self.rgb_encoder.cnn.eval()
        self.depth_encoder.eval()
        # self.waypoint_predictor.eval()

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, mode=None,
                waypoint_predictor=None,
                observations=None, 
                instruction=None, text_mask=None,
                rnn_states=None,
                cand_rgb=None, cand_depth=None,
                cand_direction=None, cand_mask=None,
                headings=None, masks=None,
                post_states=None, in_train=True):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        if mode == 'language':
            ctx, all_lang_masks = self.instruction_encoder(observations)

            return ctx, all_lang_masks

        elif mode == 'waypoint':
            # batch_size = observations['instruction'].size(0)
            batch_size = observations['rgb'].shape[0]
            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            # reverse the order of input images to clockwise
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:  # You might need to double check the keys order
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1

            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch

            depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            rgb_embedding = self.rgb_encoder(obs_view12)      # torch.Size([bs, 2048, 7, 7])

            ''' waypoint prediction ----------------------------- '''
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 2048, 7, 7)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4)
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            depth_feats = torch.cat((
                depth_embed_reshape[:,0:1,:], 
                torch.flip(depth_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            # way_feats = torch.cat((
            #     way_feats[:,0:1,:], 
            #     torch.flip(way_feats[:,1:,:], [1]),
            # ), dim=1)

            # from heatmap to points
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(
                batch_size, NUM_ANGLES, NUM_CLASSES,
            )
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,
                sigma=(7.0,5.0))
            # predicted waypoints before sampling
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            if isinstance(candidate_lengths, int):
                candidate_lengths = [candidate_lengths]
            max_candidate = max(candidate_lengths)  # including stop
            cand_mask = length2mask(candidate_lengths, device=self.device)

            if in_train:
                # Waypoint augmentation
                # parts of heatmap for sampling (fix offset first)
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-waypoint_predictor.HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-waypoint_predictor.HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                batch_way_log_prob = []
                for j in range(batch_size):
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # # candidate waypoint states
                    # way_feats_regional = way_feats[j][img_idxes]
                    # heatmap regions for sampling
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act//12+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
                    batch_way_log_prob.append(
                        probs_c.log_prob(way_heats_act))
            else:
                # batch_way_log_prob = None
                None

            cand_rgb = torch.zeros(
                (batch_size, max_candidate, 2048, 7, 7),
                dtype=torch.float32, device=self.device)
            cand_depth = torch.zeros(
                (batch_size, max_candidate, 128, 4, 4),
                dtype=torch.float32, device=self.device)
            batch_angles = []
            batch_distances = []
            batch_img_idxes = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # distance indexes for candidates
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # 2pi- becoz counter-clockwise is the positive direction
                angle_rad = 2*math.pi-angle_idxes.float()/120*2*math.pi
                batch_angles.append(angle_rad.tolist())
                batch_distances.append(
                    ((distance_idxes + 1)*0.25).tolist())
                # counter-clockwise image indexes
                img_idxes = 12 - ((angle_idxes.cpu().numpy()+5) // 10)
                img_idxes[img_idxes==12] = 0
                batch_img_idxes.append(img_idxes)
                for k in range(len(img_idxes)):
                    cand_rgb[j][k] = rgb_feats[j][img_idxes[k]]
                    cand_depth[j][k] = depth_feats[j][img_idxes[k]] 
            cand_direction = dir_angle_feature(batch_angles).to(self.device)

            if in_train:
                return cand_rgb, cand_depth, cand_direction, cand_mask, candidate_lengths, batch_angles, batch_distances #, batch_way_log_prob
            else:
                return cand_rgb, cand_depth, cand_direction, cand_mask, candidate_lengths, batch_angles, batch_distances

        elif mode == 'navigation':
            cand_rgb_feats_pool = self.space_pool(cand_rgb)
            # cand_rgb_feats_pool = self.drop_env(cand_rgb_feats_pool)
            rgb_in = self.rgb_linear(cand_rgb_feats_pool)
            cand_depth_feats_pool = self.space_pool(cand_depth)
            # cand_depth_feats_pool = self.drop_env(cand_depth_feats_pool)
            depth_in = self.depth_linear(cand_depth_feats_pool)

            vis_in = self.vismerge_linear(
                torch.cat((rgb_in, depth_in, cand_direction), dim=2),)

            ''' aggregate visual features by agent's previous state -------------- '''
            prev_state = rnn_states[:, 0:self.state_encoder.num_recurrent_layers].squeeze(1)
            vis_prev_state, _ = self.prev_state_vis_attn(
                prev_state, vis_in, cand_mask)

            ''' first state encoder for new visual features '''
            prev_actions = angle_feature(headings, device=self.device)
            prev_actions = self.enc_prev_act(prev_actions)
            # prev_actions = self.drop(prev_actions)
            
            state_in = torch.cat([vis_prev_state, prev_actions], dim=1)
            rnn_states_out = rnn_states.detach().clone()
            (
                state,
                rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
            ) = self.state_encoder(
                state_in,
                rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
                masks,
            )

            ''' language attention using state '''
            text_state, _ = self.state_text_attn(
                state, instruction, text_mask)

            ''' visual attention using attended language '''
            vis_text_feats, _ = self.text_vis_attn(
                text_state, vis_in, cand_mask)

            x = torch.cat((state, vis_text_feats, text_state), dim=1)
            _, logits = self.state_vis_logits(
                        x, vis_in, cand_mask, output_prob=False)

            return logits, rnn_states_out

        elif mode == 'waypoint_actor':
            
            None

        elif mode == 'critic':
            return self.critic(post_states)


class SoftDotAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, output_tilde=False):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_q = nn.Linear(q_dim, hidden_dim, bias=True)
        self.linear_kv = nn.Linear(kv_dim, hidden_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

        self.output_tilde = output_tilde
        if output_tilde:
            self.linear_out = nn.Linear(q_dim + hidden_dim, hidden_dim, bias=False)
            self.tanh = nn.Tanh()

    def forward(self, q, kv, mask=None, output_prob=True):
        '''Propagate h through the network.
        q: (query) batch x dim
        kv: (keys and values) batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        x_q = self.linear_q(q).unsqueeze(2)  # batch x dim x 1
        x_kv = self.linear_kv(kv)

        # Get attention
        attn = torch.bmm(x_kv, x_q).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_x_kv = torch.bmm(attn3, x_kv).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if self.output_tilde:
            h_tilde = torch.cat((weighted_x_kv, q), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_x_kv, attn
