from copy import deepcopy
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

from vlnce_baselines.models.hamt.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.image_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.encoders.video_encoder import VideoRGBEcnoder
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)
import math

@baseline_registry.register_policy
class PolicyViewSelectionHAMT(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            HAMT(
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
        config.MODEL.use_critic = (config.IL.feedback == 'sample')
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class HAMT(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the HAMT model ...')
        self.vln_bert = get_vlnbert_models(config=model_config)
        if model_config.task_type == 'r2r':
            self.rgb_projection = nn.Linear(model_config.RGB_ENCODER.output_size, 768)
        elif model_config.task_type == 'rxr':
            self.rgb_projection = nn.Linear(model_config.RGB_ENCODER.output_size, 512)
        # self.rgb_projection = nn.Linear(2048, 768) # for vit 768 compability
        self.drop_env = nn.Dropout(p=0.4)
        if model_config.use_critic:
            self.critic = Critic(drop_ratio=0.5)

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
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.backbone_type in [
        #     "TorchVisionResNet152", "TorchVisionResNet50"
        # ], "RGB_ENCODER.backbone_type must be TorchVisionResNet152 or TorchVisionResNet50"
        if model_config.RGB_ENCODER.backbone_type == "TorchVisionResNet50":
            self.rgb_encoder = TorchVisionResNet50(
                observation_space,
                model_config.RGB_ENCODER.output_size,
                device,
                spatial_output=model_config.spatial_output,
            )
        elif model_config.RGB_ENCODER.backbone_type == "CLIP":
            self.rgb_encoder = CLIPEncoder(self.device)
        elif model_config.RGB_ENCODER.backbone_type.startswith("VideoIntern"):
            self.rgb_encoder = VideoRGBEcnoder(
                observation_space,
                model_config.RGB_ENCODER.output_size,
                model_config.RGB_ENCODER.backbone_type,
                self.device
            )
            self.clip_encoder = CLIPEncoder(self.device)
            if "Base" in model_config.RGB_ENCODER.backbone_type:
                self.rgb_embedding_projection = nn.Linear(512+768, 768)
            elif "Large" in model_config.RGB_ENCODER.backbone_type:
                self.rgb_embedding_projection = nn.Linear(512+1024, 768)


        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

        if model_config.progress_monitor:
            self.state2 = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(768,512),
                    nn.Tanh()
                )
            self.progress_monitor =  nn.Sequential(
                    nn.Linear(model_config.max_len + 512, 1),
                    nn.Sigmoid()
                )

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, mode=None, 
            waypoint_predictor=None, observations=None, in_train=True,
            txt_ids=None, txt_masks=None, txt_embeds=None,
            hist_rgb_fts=None, hist_depth_fts=None, hist_ang_fts=None, embeddings = None,
            hist_pano_rgb_fts=None, hist_pano_depth_fts=None, hist_pano_ang_fts=None,
            hist_embeds=None, hist_lens=None, ob_step=None,
            ob_rgb_fts=None, ob_dep_fts=None, ob_ang_fts=None, ob_dis_fts=None,
            ob_nav_types=None, ob_masks=None, return_states=False, critic_states=None,
            h_t=None,language_attention=None):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

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

            rgb_batch = rgb_batch/255

                        
            obs_view12['depth'] = depth_batch
            obs_view12['video_rgbs'] = observations['video_rgbs']
            depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            video_embedding = self.rgb_encoder(obs_view12)      # torch.Size([bs, 2048, 7, 7])
            clip_embedding = self.clip_encoder({'rgb':rgb_batch})
            rgb_embedding_cated = torch.cat([video_embedding,clip_embedding],1)
            rgb_embedding = self.rgb_embedding_projection(rgb_embedding_cated)

            ''' waypoint prediction ----------------------------- '''
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 768, 1, 1)
            # rgb_embed_reshape = rgb_embedding.reshape(
            #     batch_size, NUM_IMGS, 2048, 7, 7)
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

            # candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            # if isinstance(candidate_lengths, int):
            #     candidate_lengths = [candidate_lengths]
            # max_candidate = max(candidate_lengths)  # including stop
            # cand_mask = length2mask(candidate_lengths, device=self.device)

            if in_train:
                # Waypoint augmentation
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                # batch_way_log_prob = []
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
                    # batch_way_log_prob.append(
                    #     probs_c.log_prob(way_heats_act))
            else:
                # batch_way_log_prob = None
                None
            
            rgb_feats = self.space_pool_rgb(rgb_feats).cpu()
            depth_feats = self.space_pool_depth(depth_feats).cpu()

            # for cand
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_dis_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # for angle & distance
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi             # 顺时针
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # 逆时针
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
                cand_dis_fts.append((((distance_idxes + 1)*0.25/3).repeat(4,1).T).cpu())
                # for img idxes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # 逆时针
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 2048
            pano_depth = depth_feats                        # B x 12 x 128
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_dis_fts = torch.zeros_like(pano_angle_fts) # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts 顺时针
            # cand_angles 逆时针
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]
                'cand_depth': cand_depth,           # [K x 128]
                'cand_angle_fts': cand_angle_fts,   # [K x 4]
                'cand_dis_fts': cand_dis_fts,       # [K x 4]
                'cand_img_idxes': cand_img_idxes,   # [K]
                'cand_angles': cand_angles,         # [K]
                'cand_distances': cand_distances,   # [K]

                'pano_rgb': pano_rgb,               # B x 12 x 2048
                'pano_depth': pano_depth,           # B x 12 x 128
                'pano_angle_fts': pano_angle_fts,   # 12 x 4
                'pano_dis_fts': pano_dis_fts,       # 12 x 4
                'pano_img_idxes': pano_img_idxes,   # 12 
            }
            
            return outputs

        elif mode == 'navigation':
            hist_embeds = torch.stack(hist_embeds, 1)
            hist_masks = length2mask(hist_lens, size=hist_embeds.size(1)).logical_not()
            
            ob_rgb_fts = self.drop_env(ob_rgb_fts)
            ob_rgb_fts = self.rgb_projection(ob_rgb_fts)
            
            act_logits, txt_embeds, hist_embeds, ob_embeds, lang_attention_score = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_rgb_fts=ob_rgb_fts, ob_dep_fts=ob_dep_fts, ob_ang_fts=ob_ang_fts, ob_dis_fts=ob_dis_fts,
                ob_nav_types=ob_nav_types, ob_masks=ob_masks)

            if return_states:
                # if self.args.no_lang_ca:
                #     states = hist_embeds[:, 0]
                # else:
                #     states = txt_embeds[:, 0] * hist_embeds[:, 0]   # [CLS]
                states = txt_embeds[:, 0] * hist_embeds[:, 0]
                return act_logits, states, lang_attention_score

            return (act_logits, )
        
        elif mode == 'history':
            if hist_rgb_fts is not None:
                hist_rgb_fts = self.drop_env(hist_rgb_fts)
                hist_rgb_fts = self.rgb_projection(hist_rgb_fts)
            if hist_pano_rgb_fts is not None:
                hist_pano_rgb_fts = self.drop_env(hist_pano_rgb_fts)
                hist_pano_rgb_fts = self.rgb_projection(hist_pano_rgb_fts)
            if ob_step is not None:
                ob_step_ids = torch.LongTensor([ob_step]).cuda()
            else:
                ob_step_ids = None
            hist_embeds = self.vln_bert(mode, hist_rgb_fts=hist_rgb_fts, 
                hist_ang_fts=hist_ang_fts, hist_depth_fts=hist_depth_fts, ob_step_ids=ob_step_ids,
                hist_pano_rgb_fts=hist_pano_rgb_fts, hist_pano_depth_fts=hist_pano_depth_fts,
                hist_pano_ang_fts=hist_pano_ang_fts)
            return hist_embeds
        
        elif mode == 'critic':
            return self.critic(critic_states)

        elif mode == 'progress':
            pm_in = torch.cat((self.state2(h_t),language_attention.sum(1)),1)
            progresses = self.progress_monitor(pm_in)
            return progresses

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
