import gc
import os
import io
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.utils import reduce_loss

from habitat.utils.visualizations.utils import images_to_video

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

@baseline_registry.register_trainer(name="HAMT")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl

        #check ceph things
        if config.CEPH_IO:
            from petrel_client.client import Client
            conf_path = '~/petreloss.conf'
            self.client = Client(conf_path)



    def _make_dirs(self) -> None:
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int,) -> None:
        if not self.config.CEPH_IO:
            torch.save(
                obj={
                    "state_dict": self.policy.state_dict(),
                    "config": self.config,
                    "optim_state": self.optimizer.state_dict(),
                    "iteration": iteration,
                },
                f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
            )
        else:
            save_dict = {
                    "state_dict": self.policy.state_dict(),
                    "config": self.config,
                    "optim_state": self.optimizer.state_dict(),
                    "iteration": iteration,
                    }
            path = os.path.join(self.config.CEPH_URL, f"ckpt.iter{iteration}.pth")
            with io.BytesIO() as buffer:
                torch.save(save_dict, buffer)
                self.client.put(path, buffer.getvalue())

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'NDTW'] # for RL reward
        self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.config.IL.max_traj_len 
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = True # not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS

        if self.config.IL.progress_monitor == True:
            self.config.MODEL.progress_monitor = True
            self.config.MODEL.max_len = self.config.IL.max_text_len
        else:
            self.config.MODEL.progress_monitor = False

        self.config.freeze()



        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _get_iter(self, x):
        x_iter = int(x.split('.')[-2][4:])
        return x_iter

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> int:
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load('pretrained/wp_pred/waypoint_predictor', map_location = torch.device('cpu'))['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)

        if config.IL.resume:
            import glob
            if not self.config.CEPH_IO:
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
            else:
                ckpt_list = [os.path.join(self.config.CEPH_URL,p) for p in self.client.list(self.config.CEPH_URL)]

            ckpt_list.sort(key=self._get_iter)

            if len(ckpt_list) > 0:
                config.defrost()
                config.IL.ckpt_to_load = ckpt_list[-1]
                load_from_ckpt = True
                config.IL.is_requeue = True
                config.freeze()
            else:
                load_from_ckpt = False

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load

            if not self.config.CEPH_IO:
                ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            else:
                with io.BytesIO(self.client.get(ckpt_path)) as buffer:
                    ckpt_dict = torch.load(buffer, map_location="cpu")

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if config.IL.is_requeue:
                start_iter = ckpt_dict["iteration"]
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
		
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })

            outputs = self.envs.call(["get_cand_idx"]*self.envs.num_envs,kargs)
            oracle_cand_idx, progresses = [list(x) for x in zip(*outputs)]
            return oracle_cand_idx, progresses

    def _cand_pano_feature_variable(self, obs):
        batch_size = len(obs['cand_angles'])
        ob_cand_lens = [len(x)+1 for x in obs['cand_angles']]  # +1 is for the end
        ob_lens = []
        ob_rgb_fts, ob_dep_fts, ob_ang_fts, ob_dis_fts, ob_nav_types = [], [], [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i in range(batch_size):
            cand_nav_types = []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            cand_rgb_fts = obs['cand_rgb'][i]
            cand_dep_fts = obs['cand_depth'][i]
            cand_ang_fts = obs['cand_angle_fts'][i]
            cand_dis_fts = obs['cand_dis_fts'][i]
            cand_nav_types += [1] * cand_ang_fts.shape[0]
            # stop
            stop_rgb_fts = torch.zeros([1, 768])
            # stop_rgb_fts = torch.zeros([1, 2048])
            stop_dep_fts = torch.zeros([1, 128])
            stop_ang_fts = torch.zeros([1, 4])
            stop_dis_fts = torch.zeros([1, 4])
            cand_nav_types += [2]
            # pano context
            pano_rgb_fts = obs['pano_rgb'][i][~cand_idxes]
            pano_dep_fts = obs['pano_depth'][i][~cand_idxes]
            pano_ang_fts = obs['pano_angle_fts'][~cand_idxes]
            pano_dis_fts = obs['pano_dis_fts'][~cand_idxes]
            cand_nav_types += [0] * (12-np.sum(cand_idxes))

            cand_pano_rgb = torch.cat([cand_rgb_fts, stop_rgb_fts, pano_rgb_fts], dim=0)
            cand_pano_dep = torch.cat([cand_dep_fts, stop_dep_fts, pano_dep_fts], dim=0)
            cand_pano_ang = torch.cat([cand_ang_fts, stop_ang_fts, pano_ang_fts], dim=0)
            cand_pano_dis = torch.cat([cand_dis_fts, stop_dis_fts, pano_dis_fts], dim=0)
            ob_rgb_fts.append(cand_pano_rgb)
            ob_dep_fts.append(cand_pano_dep)
            ob_ang_fts.append(cand_pano_ang)
            ob_dis_fts.append(cand_pano_dis)
            ob_nav_types.append(cand_nav_types)
            ob_lens.append(len(cand_nav_types))
        
        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(batch_size):
            num_pads = max_len - ob_lens[i]
            ob_rgb_fts[i] = torch.cat([ob_rgb_fts[i], torch.zeros(num_pads, 768)], dim=0)
            # ob_rgb_fts[i] = torch.cat([ob_rgb_fts[i], torch.zeros(num_pads, 2048)], dim=0)
            ob_dep_fts[i] = torch.cat([ob_dep_fts[i], torch.zeros(num_pads, 128)], dim=0)
            ob_ang_fts[i] = torch.cat([ob_ang_fts[i], torch.zeros(num_pads, 4)], dim=0)
            ob_dis_fts[i] = torch.cat([ob_dis_fts[i], torch.zeros(num_pads, 4)], dim=0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0]*num_pads)
        
        ob_rgb_fts = torch.stack(ob_rgb_fts, dim=0).cuda()
        ob_dep_fts = torch.stack(ob_dep_fts, dim=0).cuda()
        ob_ang_fts = torch.stack(ob_ang_fts, dim=0).cuda()
        ob_dis_fts = torch.stack(ob_dis_fts, dim=0).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()

        return ob_rgb_fts, ob_dep_fts, ob_ang_fts, ob_dis_fts, ob_nav_types, ob_lens, ob_cand_lens

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_depth_fts = obs['pano_depth'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_depth_fts = obs['pano_depth'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_depth_fts, hist_pano_rgb_fts, hist_pano_depth_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause, *args):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                for arg in args:
                    arg.pop(idx)
            
            for k, v in batch.items():
                if k != 'video_rgbs':
                    batch[k] = v[state_index]
                # else:
                #     batch[k] = [v[state_i] for state_i in state_index]

        return envs, batch

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")["config"]
            )
        else:
            config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.IL.ckpt_to_load = checkpoint_path
        if config.IL.progress_monitor == True:
            config.MODEL.progress_monitor = True
            config.MODEL.max_len = config.IL.max_text_len
        else:
            config.MODEL.progress_monitor = False
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print("skipping -- evaluation exists.")
                return
        envs = construct_envs(
            config, 
            get_env_class(config.ENV_NAME),
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        state_episodes = {}
        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )
        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None

        while len(state_episodes) < episodes_to_eval:
            envs.resume_all()
            observations = envs.reset()
            instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
            instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                    max_length=instr_max_len, pad_id=instr_pad_id)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            keys = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
            # states = [[envs.call_at(i,"get_agent_state",{})] for i in range(envs.num_envs)]
            history_images = [{k:observations[i][k][None,...] for k in keys} for i in range(envs.num_envs)]
            video_inputs = [{k:observations[i][k][None,...].repeat(16,0) for k in keys} for i in range(envs.num_envs)]
            batch['video_rgbs'] = video_inputs

            envs_to_pause = [i for i, ep in enumerate(envs.current_episodes()) if ep.episode_id in state_episodes]
            envs, batch = self._pause_envs(envs, batch, envs_to_pause, history_images, video_inputs)

            if envs.num_envs == 0:
                break

            # encode instructions
            all_txt_ids = batch['instruction']
            all_txt_masks = (all_txt_ids != instr_pad_id)
            all_txt_embeds = self.policy.net(
                mode='language',
                txt_ids=all_txt_ids,
                txt_masks=all_txt_masks,
            )

            not_done_index = list(range(envs.num_envs))
            hist_lens = np.ones(envs.num_envs, dtype=np.int64)
            hist_embeds = [self.policy.net('history').expand(envs.num_envs, -1)]
            for stepk in range(self.max_len):
                txt_embeds = all_txt_embeds[not_done_index]
                txt_masks = all_txt_masks[not_done_index]
                
                # cand waypoint prediction
                wp_outputs = self.policy.net(
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
                ob_rgb_fts, ob_dep_fts, ob_ang_fts, ob_dis_fts, \
                ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(wp_outputs)
                
                ob_masks = length2mask(ob_lens).logical_not()

                # navigation
                visual_inputs = {
                    'mode': 'navigation',
                    'txt_embeds': txt_embeds,
                    'txt_masks': txt_masks,
                    'hist_embeds': hist_embeds,    # history before t step
                    'hist_lens': hist_lens,
                    'ob_rgb_fts': ob_rgb_fts,
                    'ob_dep_fts': ob_dep_fts,
                    'ob_ang_fts': ob_ang_fts,
                    'ob_dis_fts': ob_dis_fts,
                    'ob_nav_types': ob_nav_types,
                    'ob_masks': ob_masks,
                    'return_states': False,
                }
                t_outputs = self.policy.net(**visual_inputs)
                logits = t_outputs[0]

                # sample action
                a_t = logits.argmax(dim=-1, keepdim=True)
                cpu_a_t = a_t.squeeze(1).cpu().numpy()

                # update history
                if stepk != self.max_len-1:
                    hist_rgb_fts, hist_depth_fts, hist_pano_rgb_fts, hist_pano_depth_fts, hist_pano_ang_fts = self._history_variable(wp_outputs)
                    prev_act_ang_fts = torch.zeros([envs.num_envs, 4]).cuda()
                    for i, next_id in enumerate(cpu_a_t):
                        prev_act_ang_fts[i] = ob_ang_fts[i, next_id]
                    t_hist_inputs = {
                        'mode': 'history',
                        'hist_rgb_fts': hist_rgb_fts,
                        'hist_depth_fts': hist_depth_fts,
                        'hist_ang_fts': prev_act_ang_fts,
                        'hist_pano_rgb_fts': hist_pano_rgb_fts,
                        'hist_pano_depth_fts': hist_pano_depth_fts,
                        'hist_pano_ang_fts': hist_pano_ang_fts,
                        'ob_step': stepk,
                    }
                    t_hist_embeds = self.policy.net(**t_hist_inputs)
                    hist_embeds.append(t_hist_embeds)
                    hist_lens = hist_lens + 1

                # make equiv action
                env_actions = []
                for j in range(envs.num_envs):
                    if cpu_a_t[j].item()==ob_cand_lens[j]-1 or stepk==self.max_len-1:
                        env_actions.append({'action':{'action': 0, 'action_args':{}}})
                    else:
                        t_angle = wp_outputs['cand_angles'][j][cpu_a_t[j]]
                        if self.config.EVAL.ANG30:
                            t_angle = round(t_angle / math.radians(30)) * math.radians(30)
                        t_distance = wp_outputs['cand_distances'][j][cpu_a_t[j]]
                        env_actions.append({'action':{'action': 4, 'action_args':{'angle': t_angle, 'distance': t_distance, 
                                                                                  'niu1niu': not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING}
                                                     }
                                            })
                outputs = envs.step(env_actions)
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                for j, ob in enumerate(observations):
                    if env_actions[j]['action']['action'] == 0:
                        continue
                    else:
                        envs.call_at(
                            j, 'change_current_path',    # to update and record low-level path
                            {'new_path': ob.pop('positions'),   
                             'collisions': ob.pop('collisions')}
                        )

                # calculate metric
                current_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    ep_id = str(current_episodes[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(np.float)
                    if 'current_path' in current_episodes[i].info.keys():
                        positions_ = np.array(current_episodes[i].info['current_path']).astype(np.float)
                        collisions_ = np.array(current_episodes[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(np.float)
                    distance = np.array(info['position']['distance']).astype(np.float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. and env_actions[i]['action']['action'] == 0 else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                    if collisions_.size == 0:
                        metric['collisions'] = 0
                    else:
                        metric['collisions'] = collisions_.mean()
                    gt_length = distance[0]
                    metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])
                    act_con_path = positions_
                    gt_con_path = np.array(dis_to_con(gt_path)).astype(np.float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))
                    metric['ndtw'] = nDTW
                    state_episodes[current_episodes[i].episode_id] = metric

                    if len(state_episodes)%300 == 0:
                        aggregated_states = {}
                        num_episodes = len(state_episodes)
                        for stat_key in next(iter(state_episodes.values())).keys():
                            aggregated_states[stat_key] = (
                                sum(v[stat_key] for v in state_episodes.values()) / num_episodes
                            )
                        print(aggregated_states)

                    if config.use_pbar:
                        pbar.update()

                # pause env
                if sum(dones) > 0:
                    for i in reversed(list(range(envs.num_envs))):
                        if dones[i]:
                            not_done_index.pop(i)
                            envs.pause_at(i)
                            observations.pop(i)
                            video_inputs.pop(i)
                            history_images.pop(i)
                if envs.num_envs == 0:
                    break

                for i in range(len(observations)):
                    states_i = observations[i].pop('states')
                    # states[i] += states_i
                    new_images_i = {k:[] for k in keys}
                    for position, rotation in states_i[:-1]:
                        new_image = envs.call_at(i,'get_pano_rgbs_observations_at', {'source_position':position,'source_rotation':rotation})
                        for k in keys:
                            new_images_i[k].append(new_image[k][None,...])
                    for k in keys:
                        new_images_i[k].append(observations[i][k][None,...])
                        history_images[i][k] = np.vstack((history_images[i][k], np.vstack(new_images_i[k])))
                        if len(history_images[i][k]) < 16:
                            video_inputs[i][k][16-len(history_images[i][k]):] = history_images[i][k]
                        else:
                            video_inputs[i][k] = history_images[i][k][-16:]
                    # print(i,stepk,len(new_images_i[k]))
                    position, rotation = states_i[-1]
                    envs.call_at(i,'set_agent_state', {'position':position,'rotation':rotation})

                hist_lens = hist_lens[np.array(dones)==False]
                for j in range(len(hist_embeds)):
                    hist_embeds[j] = hist_embeds[j][np.array(dones)==False]
                observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)
                batch['video_rgbs'] = video_inputs

        envs.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(state_episodes)
        for stat_key in next(iter(state_episodes.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in state_episodes.values()) / num_episodes
            )
        print(aggregated_states)
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(state_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")

            if config.EVAL.SAVE_RESULTS:
                checkpoint_num = int(checkpoint_index[4:])
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                if config.EVAL.SAVE_RESULTS:
                    writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self) -> None:
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        if self.config.IL.progress_monitor == True:
            self.config.MODEL.progress_monitor = True
            self.config.MODEL.max_len = config.IL.max_text_len
        else:
            self.config.MODEL.progress_monitor = False
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = []
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWINFERENCE'
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        self.config.INFERENCE.EPISODE_COUNT = -1
        self.config.INFERENCE.ANG30 = False
        self.config.freeze()

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")["config"]
            )
        else:
            config = self.config.clone()

        envs = construct_envs(
            config, 
            get_env_class(config.ENV_NAME),
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        episode_predictions = defaultdict(list)
        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}

        if config.INFERENCE.EPISODE_COUNT == -1:
            episodes_to_infer = sum(envs.number_of_episodes)
        else:
            episodes_to_infer = min(
                config.INFERENCE.EPISODE_COUNT, sum(envs.number_of_episodes)
            )
        pbar = tqdm.tqdm(total=episodes_to_infer)

        while len(episode_predictions) < episodes_to_infer:
            envs.resume_all()
            observations = envs.reset()
            instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 300
            instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                    max_length=instr_max_len, pad_id=instr_pad_id)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            keys = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
            # states = [[envs.call_at(i,"get_agent_state",{})] for i in range(envs.num_envs)]
            history_images = [{k:observations[i][k][None,...] for k in keys} for i in range(envs.num_envs)]
            video_inputs = [{k:observations[i][k][None,...].repeat(16,0) for k in keys} for i in range(envs.num_envs)]
            batch['video_rgbs'] = video_inputs

            envs_to_pause = [i for i, ep in enumerate(envs.current_episodes()) if ep.episode_id in episode_predictions]
            envs, batch = self._pause_envs(envs, batch, envs_to_pause, history_images, video_inputs)
            if envs.num_envs == 0:
                break

            # init predicitos start point
            current_episodes = envs.current_episodes()
            for i in range(envs.num_envs):
                episode_predictions[current_episodes[i].episode_id].append(
                    envs.call_at(i, "get_agent_info", {})
                )
                if config.INFERENCE.FORMAT == "rxr":
                    ep_id = current_episodes[i].episode_id
                    k = current_episodes[i].instruction.instruction_id
                    instruction_ids[ep_id] = int(k)

            # encode instructions
            all_txt_ids = batch['instruction']
            all_txt_masks = (all_txt_ids != instr_pad_id)
            all_txt_embeds = self.policy.net(
                mode='language',
                txt_ids=all_txt_ids,
                txt_masks=all_txt_masks,
            )

            not_done_index = list(range(envs.num_envs))
            hist_lens = np.ones(envs.num_envs, dtype=np.int64)
            hist_embeds = [self.policy.net('history').expand(envs.num_envs, -1)]
            for stepk in range(self.max_len):
                txt_embeds = all_txt_embeds[not_done_index]
                txt_masks = all_txt_masks[not_done_index]
                
                # cand waypoint prediction
                wp_outputs = self.policy.net(
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
                ob_rgb_fts, ob_dep_fts, ob_ang_fts, ob_dis_fts, \
                ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(wp_outputs)
                ob_masks = length2mask(ob_lens).logical_not()

                # navigation
                visual_inputs = {
                    'mode': 'navigation',
                    'txt_embeds': txt_embeds,
                    'txt_masks': txt_masks,
                    'hist_embeds': hist_embeds,    # history before t step
                    'hist_lens': hist_lens,
                    'ob_rgb_fts': ob_rgb_fts,
                    'ob_dep_fts': ob_dep_fts,
                    'ob_ang_fts': ob_ang_fts,
                    'ob_dis_fts': ob_dis_fts,
                    'ob_nav_types': ob_nav_types,
                    'ob_masks': ob_masks,
                    'return_states': False,
                }
                t_outputs = self.policy.net(**visual_inputs)
                logits = t_outputs[0]

                # sample action
                a_t = logits.argmax(dim=-1, keepdim=True)
                cpu_a_t = a_t.squeeze(1).cpu().numpy()

                # update history
                if stepk != self.max_len-1:
                    hist_rgb_fts, hist_depth_fts, hist_pano_rgb_fts, hist_pano_depth_fts, hist_pano_ang_fts = self._history_variable(wp_outputs)
                    prev_act_ang_fts = torch.zeros([envs.num_envs, 4]).cuda()
                    for i, next_id in enumerate(cpu_a_t):
                        prev_act_ang_fts[i] = ob_ang_fts[i, next_id]
                    t_hist_inputs = {
                        'mode': 'history',
                        'hist_rgb_fts': hist_rgb_fts,
                        'hist_depth_fts': hist_depth_fts,
                        'hist_ang_fts': prev_act_ang_fts,
                        'hist_pano_rgb_fts': hist_pano_rgb_fts,
                        'hist_pano_depth_fts': hist_pano_depth_fts,
                        'hist_pano_ang_fts': hist_pano_ang_fts,
                        'ob_step': stepk,
                    }
                    t_hist_embeds = self.policy.net(**t_hist_inputs)
                    hist_embeds.append(t_hist_embeds)
                    hist_lens = hist_lens + 1

                # make equiv action
                env_actions = []
                for j in range(envs.num_envs):
                    if cpu_a_t[j].item()==ob_cand_lens[j]-1 or stepk==self.max_len-1:
                        env_actions.append({'action':{'action': 0, 'action_args':{}}})
                    else:
                        t_angle = wp_outputs['cand_angles'][j][cpu_a_t[j]]
                        if self.config.INFERENCE.ANG30:
                            t_angle = round(t_angle / math.radians(30)) * math.radians(30)
                        t_distance = wp_outputs['cand_distances'][j][cpu_a_t[j]]
                        env_actions.append({'action':{'action': 4, 'action_args':{'angle': t_angle, 'distance': t_distance, 
                                                                                  'niu1niu': not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING}
                                                     }
                                            })
                outputs = envs.step(env_actions)
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                for j, ob in enumerate(observations):
                    if env_actions[j]['action']['action'] == 0:
                        continue
                    else:
                        envs.call_at(
                            j, 'update_cur_path', {'new_path': ob.pop('cur_path')}
                        )
                
                # record path
                current_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue
                    ep_id = current_episodes[i].episode_id
                    if 'cur_path' in current_episodes[i].info:
                        episode_predictions[ep_id] += current_episodes[i].info['cur_path']
                    episode_predictions[ep_id][-1]['stop'] = True
                    pbar.update()

                # pause env
                if sum(dones) > 0:
                    for i in reversed(list(range(envs.num_envs))):
                        if dones[i]:
                            not_done_index.pop(i)
                            envs.pause_at(i)
                            observations.pop(i)
                            video_inputs.pop(i)
                            history_images.pop(i)
                if envs.num_envs == 0:
                    break

                for i in range(len(observations)):
                    states_i = observations[i].pop('states')
                    # states[i] += states_i
                    new_images_i = {k:[] for k in keys}
                    for position, rotation in states_i[:-1]:
                        new_image = envs.call_at(i,'get_pano_rgbs_observations_at', {'source_position':position,'source_rotation':rotation})
                        for k in keys:
                            new_images_i[k].append(new_image[k][None,...])
                    for k in keys:
                        new_images_i[k].append(observations[i][k][None,...])
                        history_images[i][k] = np.vstack((history_images[i][k], np.vstack(new_images_i[k])))
                        if len(history_images[i][k]) < 16:
                            video_inputs[i][k][16-len(history_images[i][k]):] = history_images[i][k]
                        else:
                            video_inputs[i][k] = history_images[i][k][-16:]
                    # print(i,stepk,len(new_images_i[k]))
                    position, rotation = states_i[-1]
                    envs.call_at(i,'set_agent_state', {'position':position,'rotation':rotation})

                hist_lens = hist_lens[np.array(dones)==False]
                for j in range(len(hist_embeds)):
                    hist_embeds[j] = hist_embeds[j][np.array(dones)==False]
                observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)
                batch['video_rgbs'] = video_inputs

        envs.close()
        if config.INFERENCE.FORMAT == "r2r":
            with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)
            logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            predictions_out = []

            for k,v in episode_predictions.items():

                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if path[-1] != p["position"]:
                        path.append(p["position"])

                predictions_out.append(
                    {
                        "instruction_id": instruction_ids[k],
                        "path": path,
                    }
                )

            predictions_out.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(predictions_out)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )