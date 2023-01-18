import json
import jsonlines
import os
import sys
import time
import glob
import warnings
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations12
from ..utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class BaseVLNCETrainer(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
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
            torch.load(
                'pretrained/wp_pred/waypoint_predictor',
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=True, broadcast_buffers=False)
            # self.waypoint_predictor = DDP(self.waypoint_predictor.to(self.device), device_ids=[self.device],
            #     output_device=self.device, find_unused_parameters=True, broadcast_buffers=False)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.config.IL.lr,
        )

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"])
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                # self.waypoint_predictor.load_state_dict(ckpt_dict["waypoint_predictor_state_dict"])
                # self.waypoint_predictor = self.waypoint_predictor.module 
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"])
                # self.waypoint_predictor.load_state_dict(ckpt_dict["waypoint_predictor_state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        self.waypoint_predictor.eval()
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6} MB. Trainable: {params_t/1e6} MB")
        logger.info("Finished setting up policy.")

    # def save_checkpoint(self, file_name) -> None:
    #     r"""Save checkpoint with specified name.

    #     Args:
    #         file_name: file name for checkpoint

    #     Returns:
    #         None
    #     """
    #     checkpoint = {
    #         "state_dict": self.policy.state_dict(),
    #         "config": self.config,
    #     }
    #     torch.save(
    #         checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
    #     )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    # def _update_agent(
    #     self,
    #     observations,
    #     prev_actions,
    #     not_done_masks,
    #     corrected_actions,
    #     weights,
    #     step_grad: bool = True,
    #     loss_accumulation_scalar: int = 1,
    # ):
    #     T, N = corrected_actions.size()

    #     recurrent_hidden_states = torch.zeros(
    #         N,
    #         self.num_recurrent_layers,
    #         self.config.MODEL.STATE_ENCODER.hidden_size,
    #         device=self.device,
    #     )
    #     AuxLosses.clear()
    #     # observations['rgb'] = observations['rgb'][0:2]
    #     # observations['depth'] = observations['depth'][0:2]
    #     # observations['rxr_instruction'] = observations['rxr_instruction'][0:2]
    #     # not_done_masks = not_done_masks[0:2]
    #     # prev_actions = prev_actions[0:2]

    #     distribution = self.policy.build_distribution(
    #         observations, recurrent_hidden_states, prev_actions, not_done_masks)

    #     logits = distribution.logits
    #     logits = logits.view(T, N, -1)

    #     action_loss = F.cross_entropy(
    #         logits.permute(0, 2, 1), corrected_actions, reduction="none"
    #     )
    #     action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

    #     aux_mask = (weights > 0).view(-1)
    #     aux_loss = AuxLosses.reduce(aux_mask)

    #     loss = action_loss + aux_loss
    #     loss = loss / loss_accumulation_scalar
    #     loss.backward()

    #     if step_grad:
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #     # if isinstance(aux_loss, torch.Tensor):
    #     #     aux_loss = aux_loss.item()
    #     # return loss.item(), action_loss.item(), aux_loss

    #     return loss, action_loss, aux_loss

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
        # positions=None
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                # positions.pop(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
            # positions
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()
        config.defrost()
        # config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        # config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        # config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        # config.TASK_CONFIG.TASK.SDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
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
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj  # split by rank
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

        observations = envs.reset() 
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        if 'CMA' in self.config.MODEL.policy_name:
            rnn_states = torch.zeros(
                envs.num_envs,
                self.num_recurrent_layers,
                config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
        elif 'VLNBERT' in self.config.MODEL.policy_name:
            h_t = torch.zeros(
                envs.num_envs, 768,
                device=self.device,
            )
            language_features = torch.zeros(
                envs.num_envs, 80, 768,
                 device=self.device,
            )
       
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # number = 0
        total_weight = 0.
        ml_loss = 0.
        bpositions = [[] for _ in range(envs.num_envs)]
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)):
                agent_state_i = envs.call_at(ob_i, "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            with torch.no_grad():
                if 'CMA' in self.config.MODEL.policy_name:
                    # instructions
                    instruction_embedding, all_lang_masks = self.policy.net(
                        mode = "language",
                        observations = batch,
                    )

                    # candidate waypoints prediction
                    cand_rgb, cand_depth, \
                    cand_direction, cand_mask, candidate_lengths, \
                    batch_angles, batch_distances = self.policy.net(
                        mode = "waypoint",
                        waypoint_predictor = self.waypoint_predictor,
                        observations = batch,
                        in_train = False,
                    )
                    # navigation action logits
                    logits, rnn_states = self.policy.net(
                        mode = 'navigation',
                        observations = batch,
                        instruction = instruction_embedding,
                        text_mask = all_lang_masks,
                        rnn_states = rnn_states,
                        headings = headings,
                        cand_rgb = cand_rgb, 
                        cand_depth = cand_depth,
                        cand_direction = cand_direction,
                        cand_mask = cand_mask,
                        masks = not_done_masks,
                    )
                    logits = logits.masked_fill_(cand_mask, -float('inf'))

                elif 'VLNBERT' in self.config.MODEL.policy_name:
                    if 'R2R' in self.config.TASK_CONFIG.DATASET.DATA_PATH:
                        lang_idx_tokens = batch['instruction']
                        padding_idx = 0
                        lang_masks = (lang_idx_tokens != padding_idx)
                        lang_lengths = lang_masks.sum(1)
                        lang_token_type_ids = torch.zeros_like(lang_masks,
                            dtype=torch.long, device=self.device)
                        h_t_flag = h_t.sum(1)==0.0       
                        h_t_init, language_features = self.policy.net(
                            mode='language',
                            lang_idx_tokens=lang_idx_tokens,
                            lang_masks=lang_masks)
                    elif 'RxR' in self.config.TASK_CONFIG.DATASET.DATA_PATH:
                        to_be_masked = ((torch.abs(batch['rxr_instruction']) == 0)*1.).mean(-1)
                        lang_masks = torch.ones_like(to_be_masked) - to_be_masked
                        # lang_lengths = all_lang_masks.sum(1)
                        h_t_flag = h_t.sum(1)==0.0       
                        h_t_init, language_features = self.policy.net(
                            mode='language',
                            observations=batch,
                            lang_masks=lang_masks,
                        )
                    else:
                        raise NotImplementedError
                    h_t[h_t_flag] = h_t_init[h_t_flag]
                    language_features = torch.cat(
                        (h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
                    # candidate waypoints prediction
                    cand_rgb, cand_depth, \
                    cand_direction, cand_mask, candidate_lengths, \
                    batch_angles, batch_distances = self.policy.net(
                        mode = "waypoint",
                        waypoint_predictor = self.waypoint_predictor,
                        observations = batch,
                        in_train = False,
                    )
                    # navigation action logits
                    logits, h_t = self.policy.net(
                        mode = 'navigation',
                        observations=batch,
                        lang_masks=lang_masks,
                        lang_feats=language_features,
                        # lang_token_type_ids=lang_token_type_ids,
                        headings=headings,
                        cand_rgb = cand_rgb, 
                        cand_depth = cand_depth,
                        cand_direction = cand_direction,
                        cand_mask = cand_mask,                    
                        masks = not_done_masks,
                    )
                    logits = logits.masked_fill_(cand_mask, -float('inf'))

                # high-to-low actions in environments
                actions = logits.argmax(dim=-1, keepdim=True)
                env_actions = []
                for j in range(logits.size(0)):
                    if actions[j].item() == candidate_lengths[j]-1:
                        env_actions.append({'action':
                            {'action': 0, 'action_args':{}}})
                    else:
                        env_actions.append({'action':
                            {'action': 4,  # HIGHTOLOW
                            'action_args':{
                                'angle': batch_angles[j][actions[j].item()], 
                                'distance': batch_distances[j][actions[j].item()],
                            }}})

            outputs = envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            for j, ob in enumerate(observations):
                if env_actions[j]['action']['action'] == 0:
                    continue
                else:
                    envs.call_at(j, 
                        'change_current_path',    # to update and record low-level path
                        {'new_path': ob.pop('positions'),   
                        'collisions': ob.pop('collisions')}
                    )

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8, device=self.device)

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                # ep done, calculate metrics
                info = infos[i]
                metric = {}
                metric['steps_taken'] = info['steps_taken']
                ep_id = str(envs.current_episodes()[i].episode_id)
                gt_path = np.array(self.gt_data[ep_id]['locations']).astype(np.float)
                if 'current_path' in envs.current_episodes()[i].info.keys():
                    positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(np.float)
                    collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                    assert collisions_.shape[0] == positions_.shape[0] - 1
                else:
                    positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(np.float)
                distance = np.array(info['position']['distance']).astype(np.float)
                metric['distance_to_goal'] = distance[-1]
                metric['success'] = 1. if distance[-1] <= 3. and env_actions[i]['action']['action'] == 0 else 0.
                metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                try:
                    metric['collisions'] = collisions_.mean()
                except:
                    metric['collisions'] = 0
                    pass
            
                gt_length = distance[0]
                metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                act_con_path = positions_
                gt_con_path = np.array(dis_to_con(gt_path)).astype(np.float)
                dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                metric['ndtw'] = nDTW
                stats_episodes[current_episodes[i].episode_id] = metric

                observations[i] = envs.reset_at(i)[0] # envs[i] change to next episode
                
                if 'CMA' in self.config.MODEL.policy_name:
                    rnn_states[i] *= 0.
                elif 'VLNBERT' in self.config.MODEL.policy_name:
                    h_t[i] *= 0.

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=episodes_to_eval,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "spl": stats_episodes[
                                current_episodes[i].episode_id
                            ]["spl"]
                        },
                        tb_writer=writer,
                        fps=1,
                    )

                    # del stats_episodes[current_episodes[i].episode_id][
                    #     "top_down_map_vlnce"
                    # ]
                    # del stats_episodes[current_episodes[i].episode_id][
                    #     "collisions"
                    # ]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:  # 出现了重复的ep，表示这个模拟器中的episode已经全部过了一遍
                    envs_to_pause.append(i)

            if 'VLNBERT' in self.config.MODEL.policy_name:
                rnn_states = h_t

            headings = torch.tensor(headings)
            (
                envs,
                rnn_states,
                not_done_masks,
                headings,  # prev_actions
                batch,
                rgb_frames,
                # positions
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                headings,  # prev_actions
                batch,
                rgb_frames,
                # positions
            )
            headings = headings.tolist()
            if 'VLNBERT' in self.config.MODEL.policy_name:
                h_t = rnn_states

        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        # print('rank', self.local_rank, 'evaluated',num_episodes, 'episodes')
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )
        # print(self.local_rank, aggregated_stats)
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                # print(self.local_rank, k+':', v.item(), num_episodes, 'before reduce')
                cat_v = gather_list_and_concat(v,self.world_size)
                # print(self.local_rank, k+':', cat_v, num_episodes, 'after_reduce')
                v = (sum(cat_v)/total).item()
                # print(self.local_rank, k+':', v, num_episodes, 'after divide total')
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    def collect_val_traj(self):
        from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            if "{role}" in self.config.IL.RECOLLECT_TRAINER.gt_file:
                gt_data = {}
                for role in RxRVLNCEDatasetV1.annotation_roles:
                    if (
                        ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
                        and role not in self.config.TASK_CONFIG.DATASET.ROLES
                    ):
                        continue

                    with gzip.open(
                        self.config.IL.RECOLLECT_TRAINER.gt_file.format(
                            split=split, role=role
                        ),
                        "rt",
                    ) as f:
                        gt_data.update(json.load(f))
            else:
                with gzip.open(
                    self.config.IL.RECOLLECT_TRAINER.gt_path.format(
                        split=split)
                ) as f:
                    gt_data = json.load(f)
        else:
            with gzip.open(
                self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)
            ) as f:
                gt_data = json.load(f)

        self.gt_data = gt_data

        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]

        return trajectories

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        # split = self.config.TASK_CONFIG.DATASET.SPLIT
        # self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        # self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION', 'STEPS_TAKEN']
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]

        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        # if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
        #     self.config.EVAL.trajectories_file = \
        #         self.config.EVAL.trajectories_file[:-8] + '_w' + \
        #         str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")

            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))

        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        # self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        # self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
        #     -1
        # )
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        #
        # if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
        self.traj = self.collect_val_traj()
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(
                    self.config.EVAL.CKPT_PATH_DIR
                )
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.EVAL.CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                # prev_ckpt_ind = -1 #TODO eval start index
                evaluated = []
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        checkpoint_folder = self.config.EVAL_CKPT_PATH_DIR
                        if not self.config.CEPH_IO:
                            models_paths = [p for p in filter(os.path.isfile, glob.glob(checkpoint_folder + "/*")) if p not in evaluated]
                        else:
                            models_paths = [os.path.join(self.config.CEPH_URL,p) for p in self.client.list(self.config.CEPH_URL) if os.path.join(self.config.CEPH_URL,p) not in evaluated]
                        if len(models_paths) > 0:
                            models_paths.sort(key=self._get_iter)
                            current_ckpt = models_paths[0]
                            prev_ckpt_ind = current_ckpt.split('.')[-2]
                        else:
                            current_ckpt = None
                        time.sleep(2)  # sleep for 2 secs before polling again
                    # time.sleep(10)
                    if self.local_rank < 1:
                        logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    # prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )
                    evaluated.append(current_ckpt)

    def inference(self) -> None:
        r"""Runs inference on a single checkpoint, creating a path predictions file."""
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.IL.ckpt_to_load = self.config.INFERENCE.CKPT_PATH
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = []
        self.config.TASK_CONFIG.TASK.SENSORS = [
            s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        ########### Additional Config ###########
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
        # self.config.ENV_NAME = "VLNCEInferenceEnv" #TODO is this necessary?
        self.config.freeze()

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.freeze()

        eps = self.collect_val_traj()
        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=eps[:10] if sys.gettrace() else None # for debug, ep subset
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

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        if 'CMA' in self.config.MODEL.policy_name:
            rnn_states = torch.zeros(
                envs.num_envs,
                self.num_recurrent_layers,
                config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
        elif 'VLNBERT' in self.config.MODEL.policy_name:
            h_t = torch.zeros(
                envs.num_envs, 768,
                device=self.device,
            )
            language_features = torch.zeros(
                envs.num_envs, 80, 768,
                 device=self.device,
            )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        episode_predictions = defaultdict(list)
        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}

        # populate episode_predictions with the starting state
        current_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(
                envs.call_at(i, "get_agent_info", {})
            )
            if config.INFERENCE.FORMAT == "rxr":
                ep_id = current_episodes[i].episode_id
                k = current_episodes[i].instruction.instruction_id
                instruction_ids[ep_id] = int(k)
        
        with tqdm.tqdm(
            total=sum(envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while envs.num_envs > 0:
                current_episodes = envs.current_episodes()
                positions = []; headings = []
                for i in range(envs.num_envs):
                    agent_state_i = envs.call_at(i,"get_agent_info", {})
                    positions.append(agent_state_i['position'])
                    headings.append(agent_state_i['heading'])
                
                with torch.no_grad():
                    if 'CMA' in self.config.MODEL.policy_name:
                        # instructions
                        instruction_embedding, all_lang_masks = self.policy.net(
                            mode = "language",
                            observations = batch,
                        )

                        # candidate waypoints prediction
                        cand_rgb, cand_depth, \
                        cand_direction, cand_mask, candidate_lengths, \
                        batch_angles, batch_distances = self.policy.net(
                            mode = "waypoint",
                            waypoint_predictor = self.waypoint_predictor,
                            observations = batch,
                            in_train = False,
                        )
                        # navigation action logits
                        logits, rnn_states = self.policy.net(
                            mode = 'navigation',
                            observations = batch,
                            instruction = instruction_embedding,
                            text_mask = all_lang_masks,
                            rnn_states = rnn_states,
                            headings = headings,
                            cand_rgb = cand_rgb, 
                            cand_depth = cand_depth,
                            cand_direction = cand_direction,
                            cand_mask = cand_mask,
                            masks = not_done_masks,
                        )
                        logits = logits.masked_fill_(cand_mask, -float('inf'))

                    # high-to-low actions in environments
                    actions = logits.argmax(dim=-1, keepdim=True)
                    env_actions = []
                    for j in range(logits.size(0)):
                        if actions[j].item() == candidate_lengths[j]-1:
                            env_actions.append({'action':
                                {'action': 0, 'action_args':{}}})
                        else:
                            env_actions.append({'action':
                                {'action': 4,  # HIGHTOLOW
                                'action_args':{
                                    'angle': batch_angles[j][actions[j].item()], 
                                    'distance': batch_distances[j][actions[j].item()],
                                }}})

                outputs = envs.step(env_actions)
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                for i, ob in enumerate(observations):
                    if env_actions[i]['action']['action'] == 0:
                        continue
                    else:
                        envs.call_at(
                            i, 'update_cur_path', {'new_path': ob.pop('cur_path')}
                        ) # to update and record low-level path

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue
                    
                    ep_id = envs.current_episodes()[i].episode_id
                    if 'cur_path' in envs.current_episodes()[i].info:
                        episode_predictions[ep_id] += envs.current_episodes()[i].info['cur_path']
                    episode_predictions[ep_id][-1]['stop'] = True
                    # assert len(episode_predictions[ep_id]) <= 500

                    observations[i] = envs.reset_at(i)[0]
                    if 'CMA' in self.config.MODEL.policy_name:
                        rnn_states[i] *= 0.
                    elif 'VLNBERT' in self.config.MODEL.policy_name:
                        h_t[i] *= 0.
                    # prev_actions[i] = torch.zeros(1, dtype=torch.long)
                    pbar.update()

                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)

                envs_to_pause = []
                next_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue

                    if next_episodes[i].episode_id in episode_predictions:
                        envs_to_pause.append(i)
                    else:
                        episode_predictions[next_episodes[i].episode_id].append(
                            envs.call_at(i, "get_agent_info", {})
                        )
                        if config.INFERENCE.FORMAT == "rxr":
                            ep_id = next_episodes[i].episode_id
                            k = next_episodes[i].instruction.instruction_id
                            instruction_ids[ep_id] = int(k)
                    # number += 1

                headings = torch.tensor(headings)
                (
                    envs,
                    rnn_states,
                    not_done_masks,
                    headings,  # prev_actions
                    batch,
                    rgb_frames,
                    # positions
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    rnn_states,
                    not_done_masks,
                    headings,  # prev_actions
                    batch,
                    # rgb_frames,
                    # positions
                )
                headings = headings.tolist()
                if 'VLNBERT' in self.config.MODEL.policy_name:
                    h_t = rnn_states

        envs.close()

        if config.INFERENCE.FORMAT == "r2r":
            with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
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
