from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, List, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from habitat.core.simulator import SensorSuite
from habitat.core.registry import registry


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.prev_episode_id = "something different"
        self.keys = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']
        self.pano_rgbs_sensors = [{k:self._env.sim._sensors[k] for k in self.keys}]

        sim_sensors = []
        for sensor_name in ['RGB_SENSOR'] + [k.upper() for k in self.keys if k != 'rgb']:
            sensor_cfg = getattr(self._env.sim.habitat_config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))
        self.sensor_suite = SensorSuite(sim_sensors)

        self.current_scene = self._env.sim._current_scene

    def reset(self) -> Observations:
        observations = self._env.reset()
        if self.current_scene != self._env.sim._current_scene:
            self.pano_rgbs_sensors = [{k:self._env.sim._sensors[k] for k in self.keys}]

            sim_sensors = []
            for sensor_name in ['RGB_SENSOR'] + [k.upper() for k in self.keys if k != 'rgb']:
                sensor_cfg = getattr(self._env.sim.habitat_config, sensor_name)
                sensor_type = registry.get_sensor(sensor_cfg.TYPE)

                assert sensor_type is not None, "invalid sensor type {}".format(
                    sensor_cfg.TYPE
                )
                sim_sensors.append(sensor_type(sensor_cfg))
            self.sensor_suite = SensorSuite(sim_sensors)
        
            self.current_scene = self._env.sim._current_scene

        return observations

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }

    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        return self._env.sim.get_observations_at(
            source_position,
            source_rotation,
            keep_agent_at_new_pose)

    def observations_by_angles(self, angle_list: List[float]):
        r'''for getting observations from desired angles
        requires rad, positive represents anticlockwise'''
        obs = []
        sim = self._env.sim
        init_state = sim.get_agent_state()
        prev_angle = 0
        left_action = HabitatSimActions.TURN_LEFT
        init_amount = sim.get_agent(0).agent_config.action_space[left_action].actuation.amount # turn left
        for angle in angle_list:
            sim.get_agent(0).agent_config.action_space[left_action].actuation.amount = (angle-prev_angle)*180/np.pi
            obs.append(sim.step(left_action))
            prev_angle = angle
        sim.set_agent_state(init_state.position, init_state.rotation)
        sim.get_agent(0).agent_config.action_space[left_action].actuation.amount = init_amount
        return obs

    def current_dist_to_goal(self):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance

    def current_dist_to_refpath(self, path):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        current_pos = init_state.position
        circle_dists = []
        for pos in path:
            circle_dists.append(
                self._env.sim.geodesic_distance(current_pos, pos)
            )
        # circle_dists = np.linalg.norm(np.array(path)-current_pos, axis=1).tolist()
        return circle_dists

    def get_cand_idx(self,ref_path,angles,distances,candidate_length):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0,0.0,0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        cand_dists_to_goal = []
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        for k in range(len(angles)):
            angle_k = angles[k]
            forward_k = distances[k]
            dist_k = self.cand_dist_to_subgoal(angle_k, forward_k, sub_goal_pos)
            # distance to subgoal
            cand_dists_to_goal.append(dist_k)

        # distance to final goal
        curr_dist_to_goal = self.current_dist_to_goal()
        # if within target range (which def as 3.0)
        if curr_dist_to_goal < 1.5:
            oracle_cand_idx = candidate_length - 1
        else:
            oracle_cand_idx = np.argmin(cand_dists_to_goal)

        self.prev_episode_id = episode_id
        # if curr_dist_to_goal == np.inf:
        
        progress100 = self.progress/len(ref_path)
        return oracle_cand_idx, progress100#, sub_goal_pos

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance

    def cand_dist_to_subgoal(self, 
        angle: float, forward: float,
        sub_goal: Any):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        prev_pos = init_state.position
        dis = 0.
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
            pos = sim.get_agent_state().position
            dis += np.linalg.norm(prev_pos - pos)
            prev_pos = pos
        post_state = sim.get_agent_state()

        post_distance = self._env.sim.geodesic_distance(
            post_state.position, sub_goal,
        ) + dis

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
        
    def change_current_path(self, new_path: Any, collisions: Any):
        '''just for recording current path in high to low'''
        if self._env.current_episode.info is None:
            self._env.current_episode.info = {}
        if 'current_path' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['current_path'] = [np.array(self._env.current_episode.start_position)]
        self._env.current_episode.info['current_path'] += new_path
        if 'collisions' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['collisions'] = []
        self._env.current_episode.info['collisions'] += collisions

    # def draw_point(self,point,type,map):
    #     from scripts.draw_map_utils import drawpoint
    #     drawpoint(point,type,map,self._env.sim)

    def update_cur_path(self, new_path: Dict):
        if self._env.current_episode.info is None:
            self._env.current_episode.info = defaultdict(list)
        if 'cur_path' not in self._env.current_episode.info:
            self._env.current_episode.info['cur_path'] = []
        self._env.current_episode.info['cur_path'] += new_path
    
    def stop_cur_path(self): # not used
        assert self._env.current_episode.info is not None
        assert 'cur_path' in self._env.current_episode.info.keys()
        self._env.current_episode.info['cur_path'][-1]['stop'] = True

    def get_pano_rgbs_observations_at(self,
            source_position: List,
            source_rotation: List,):
        self._env.sim.set_agent_state(source_position,source_rotation)
        pano_rgbs = self.sensor_suite.get_observations(self._env.sim.get_specific_sensors_observations(self.pano_rgbs_sensors))

        return pano_rgbs

    def get_agent_state(self):
        agent_state = self._env.sim.get_agent_state()

        return (agent_state.position,agent_state.rotation)
    
    def set_agent_state(self, position, rotation):
        self._env.sim.set_agent_state(position,rotation)
        

@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
