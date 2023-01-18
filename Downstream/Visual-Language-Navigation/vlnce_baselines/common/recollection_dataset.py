import gzip
import json
from collections import defaultdict, deque

import numpy as np
import torch
import tqdm
from gym import Space
from habitat.config.default import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens


class TeacherRecollectionDataset(torch.utils.data.IterableDataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # self._preload = []
        self._preload = deque()
        self.world_size = self.config.GPU_NUMBERS
        self.rank = self.config.local_rank

        assert (
            config.IL.RECOLLECT_TRAINER.preload_size >= config.IL.batch_size
        ), "preload size must be greater than batch size."
        self.envs = None
        self._env_observations = None

        if config.IL.use_iw:
            self.inflec_weights = torch.tensor(
                [1.0, config.IL.inflection_weight_coef]
            )
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        if self.config.IL.RECOLLECT_TRAINER.preload_trajectories_file:
            self.config.defrost()
            self.config.IL.RECOLLECT_TRAINER.trajectories_file = \
                self.config.IL.RECOLLECT_TRAINER.trajectories_file[
                :-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.rank) + '.json.gz'
            self.config.freeze()
            with gzip.open(
                config.IL.RECOLLECT_TRAINER.trajectories_file, "rt"
            ) as f:
                self.trajectories = json.load(f)
        else:
            self.trajectories = self.collect_dataset()

        self.initialize_sims()

    def initialize_sims(self):
        config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.MEASUREMENTS = []
        config.freeze()

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            episodes_allowed=list(self.trajectories.keys()),
        )
        self.length = sum(self.envs.number_of_episodes)
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], self.obs_transforms
        )

        self.env_step = [0 for _ in range(self.envs.num_envs)]
        self._env_observations = [[] for _ in range(self.envs.num_envs)]

        observations = self.envs.reset()
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        for i, ep in enumerate(self.envs.current_episodes()):
            path_step = self.trajectories[str(ep.episode_id)][0]
            self._env_observations[i].append(
                (
                    observations[i],
                    path_step[0],  # prev_action
                    path_step[2],  # oracle_action
                )
            )

    @property
    def batch_size(self):
        return self.config.IL.batch_size

    @property
    def observation_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        assert self._observation_space is not None
        return self._observation_space

    @property
    def action_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        return self.envs.action_spaces[0]

    def close_sims(self):
        self.envs.close()
        del self.envs
        del self._env_observations
        self.envs = None
        self._env_observations = None

    def collect_dataset(self):
        r"""Uses the ground truth trajectories to create a teacher forcing
        datset for a given split. Loads both guide and follower episodes.
        """
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT

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
                self.config.IL.RECOLLECT_TRAINER.gt_path.format(split=split)
            ) as f:
                gt_data = json.load(f)

        t = (
            tqdm.tqdm(gt_data.items(), "GT Collection")
            if self.config.use_pbar
            else gt_data.items()
        )

        for episode_id, trajectory in t:
            if (
                self.config.IL.RECOLLECT_TRAINER.max_traj_len != -1
                and len(trajectory["actions"])
                > self.config.IL.RECOLLECT_TRAINER.max_traj_len
            ) or (
                self.config.IL.RECOLLECT_TRAINER.min_traj_len != -1
                and len(trajectory["actions"])
                < self.config.IL.RECOLLECT_TRAINER.min_traj_len
            ):
                continue

            for i, action in enumerate(trajectory["actions"]):
                prev_action = (
                    trajectories[episode_id][i - 1][1]
                    if i
                    else HabitatSimActions.STOP
                )

                # [prev_action, action, oracle_action]
                trajectories[episode_id].append([prev_action, action, action])

        trajectories = dict(list(trajectories.items())[self.rank::self.world_size])
        self.config.defrost()
        self.config.IL.RECOLLECT_TRAINER.trajectories_file = \
            self.config.IL.RECOLLECT_TRAINER.trajectories_file[:-8]+'_w'+ \
            str(self.world_size)+'_r'+str(self.rank) + '.json.gz'
        self.config.freeze()
        with gzip.open(
            self.config.IL.RECOLLECT_TRAINER.trajectories_file, "wt"
        ) as f:
            f.write(json.dumps(trajectories))
        return trajectories

    def _load_next(self):
        """
        Episode length is currently not considered. We were previously batching episodes
        together with similar lengths. Not sure if we need to bring that back.
        """
        # self.rank = 0
        if len(self._preload):
            # out = self._preload[self.rank]
            # self._preload = self._preload[self.world_size:]
            # return out
            return self._preload.popleft()

        while (
            len(self._preload) < self.config.IL.RECOLLECT_TRAINER.preload_size
        ):
            current_episodes = self.envs.current_episodes()
            prev_eps = current_episodes

            # get the next action for each env
            actions = [
                self.trajectories[str(ep.episode_id)][self.env_step[i]][1]
                for i, ep in enumerate(current_episodes)
            ]

            outputs = self.envs.step(actions)
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            current_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                self.env_step[i] += 1
                if dones[i]:
                    assert len(self._env_observations[i]) == len(
                        self.trajectories[str(prev_eps[i].episode_id)]
                    ), "Collected episode does not match the step count of trajectory"
                    self._preload.append(
                        (
                            [o[0] for o in self._env_observations[i]],
                            [o[1] for o in self._env_observations[i]],
                            [o[2] for o in self._env_observations[i]],
                        )
                    )
                    self._env_observations[i] = []
                    self.env_step[i] = 0

                path_step = self.trajectories[
                    str(current_episodes[i].episode_id)
                ][self.env_step[i]]
                self._env_observations[i].append(
                    (
                        observations[i],
                        path_step[0],  # prev_action
                        path_step[2],  # oracle_action
                    )
                )
                assert (
                    len(self._env_observations[i])
                    <= self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
                ), "Trajectories should be no more than the maximum episode steps."

        # out = self._preload[self.rank]
        # self._preload = self._preload[self.world_size:]
        # return out
        return self._preload.popleft()

    def __next__(self):
        """Takes about 1s to once self._load_next() has finished with a batch
        size of 5. For this reason, we probably don't need to use extra workers.
        """
        x = self._load_next()
        obs, prev_actions, oracle_actions = x

        # transpose obs
        obs_t = defaultdict(list)
        for k in obs[0]:
            for i in range(len(obs)):
                obs_t[k].append(obs[i][k])

            obs_t[k] = np.array(obs_t[k])

        for k, v in obs_t.items():
            obs_t[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs_t,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                worker_info.num_workers == 1
            ), "multiple workers not supported."

        return self
