#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

import habitat_sim

from habitat_sim.simulator import MutableMapping, MutableMapping_T
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space
from collections import OrderedDict

# inherit habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py
@registry.register_simulator(name="Sim-v1")
class Simulator(HabitatSim):
    r"""Simulator wrapper over habitat-sim
    habitat-sim repo: https://github.com/facebookresearch/habitat-sim
    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def step_without_obs(self,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()

        # # step physics by dt
        # step_start_Time = time.time()
        # super().step_world(dt)
        # self._previous_step_time = time.time() - step_start_Time

        multi_observations = {}
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation

        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs

    def step_with_specific_sensors(self,
        sensors,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()

        # # step physics by dt
        # step_start_Time = time.time()
        # super().step_world(dt)
        # self._previous_step_time = time.time() - step_start_Time

        multi_observations = self.get_specific_sensors_observations(sensors = sensors)
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation

        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs

        return multi_observations

    def get_specific_sensors_observations(
        self, sensors, agent_ids: Union[int, List[int]] = 0,
    ) -> Union[
        Dict[str, Union[ndarray, "Tensor"]],
        Dict[int, Dict[str, Union[ndarray, "Tensor"]]],
    ]:
        if isinstance(agent_ids, int):
            agent_ids = [agent_ids]
            return_single = True
        else:
            return_single = False

        for agent_id in agent_ids:
            agent_sensorsuite = sensors[agent_id]
            for _sensor_uuid, sensor in agent_sensorsuite.items():
                sensor.draw_observation()

        # As backport. All Dicts are ordered in Python >= 3.7
        observations: Dict[int, Dict[str, Union[ndarray, "Tensor"]]] = OrderedDict()
        for agent_id in agent_ids:
            agent_observations: Dict[str, Union[ndarray, "Tensor"]] = {}
            for sensor_uuid, sensor in sensors[agent_id].items():
                agent_observations[sensor_uuid] = sensor.get_observation()
            observations[agent_id] = agent_observations
        if return_single:
            return next(iter(observations.values()))
        return observations

    # def render_specific_sensors(self, sensors, mode: str = "rgb") -> Any:
    #     r"""
    #     Args:
    #         mode: sensor whose observation is used for returning the frame,
    #             eg: "rgb", "depth", "semantic"

    #     Returns:
    #         rendered frame according to the mode
    #     """
    #     sim_obs = self.get_specific_sensors_observations(sensors = sensors)
    #     observations = self._sensor_suite.get_observations(sim_obs)

    #     output = observations.get(mode)
    #     assert output is not None, "mode {} sensor is not active".format(mode)
    #     if not isinstance(output, np.ndarray):
    #         # If it is not a numpy array, it is a torch tensor
    #         # The function expects the result to be a numpy array
    #         output = output.to("cpu").numpy()

    #     return output