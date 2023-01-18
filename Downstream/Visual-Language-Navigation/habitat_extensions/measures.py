import gzip
import json
import pickle
from typing import Any, List, Union

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war
from habitat.utils.visualizations import maps as habitat_maps

from habitat_extensions import maps

@registry.register_measure
class Position(Measure):
    r"""Path Length (PL)

    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "position"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = {'distance':[], 'position':[]}
        self.update_metric(episode)

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        if len(self._metric['position']) > 0:
            if (current_position == self._metric['position'][-1]).all():
                return
        distance = self._sim.geodesic_distance(
            current_position,
            [goal.position for goal in episode.goals],
            episode,
        )
        self._metric['position'].append(self._sim.get_agent_state().position)
        self._metric['distance'].append(distance)

@registry.register_measure
class PathLength(Measure):
    r"""Path Length (PL)

    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "path_length"

    @staticmethod
    def euclidean_distance(
        position_a: np.ndarray, position_b: np.ndarray
    ) -> float:
        return np.linalg.norm(position_b - position_a, ord=2)

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += self.euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    r"""Oracle Navigation Error (ONE)

    ONE = min(geosdesic_distance(agent_pos, goal))
            over all locations in the agent's path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(episode, task)

    def update_metric(self, episode, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    r"""Oracle Success Rate (OSR)

    OSR = I(ONE <= goal_radius),
    where ONE is Oracle Navigation Error.
    """

    cls_uuid: str = "oracle_success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0
        self.update_metric(episode, task)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < self._config.SUCCESS_DISTANCE)


@registry.register_measure
class OracleSPL(Measure):
    r"""OracleSPL (Oracle Success weighted by Path Length)

    OracleSPL = max(SPL) over all points in the agent path
    """

    cls_uuid: str = "oracle_spl"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)


@registry.register_measure
class StepsTaken(Measure):
    r"""Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    cls_uuid: str = "steps_taken"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


@registry.register_measure
class NDTW(Measure):
    r"""NDTW (Normalized Dynamic Time Warping)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    cls_uuid: str = "ndtw"

    @staticmethod
    def euclidean_distance(
        position_a: Union[List[float], np.ndarray],
        position_b: Union[List[float], np.ndarray],
    ) -> float:
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.dtw_func = fastdtw if config.FDTW else dtw

        if "{role}" in config.GT_PATH:
            self.gt_json = {}
            for role in RxRVLNCEDatasetV1.annotation_roles:
                with gzip.open(
                    config.GT_PATH.format(split=config.SPLIT, role=role), "rt"
                ) as f:
                    self.gt_json.update(json.load(f))
        else:
            with gzip.open(
                config.GT_PATH.format(split=config.SPLIT), "rt"
            ) as f:
                self.gt_json = json.load(f)

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=self.euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance
            / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
        )
        self._metric = nDTW


@registry.register_measure
class SDTW(Measure):
    r"""SDTW (Success Weighted be nDTW)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    cls_uuid: str = "sdtw"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [NDTW.cls_uuid, Success.cls_uuid]
        )
        self.update_metric(episode, task)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        nDTW = task.measurements.measures[NDTW.cls_uuid].get_metric()
        self._metric = ep_success * nDTW


@registry.register_measure
class TopDownMapVLNCE(Measure):
    r"""A top down map that optionally shows VLN-related visual information
    such as MP3D node locations and MP3D agent traversals.
    """

    cls_uuid: str = "top_down_map_vlnce"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        with open(self._config.GRAPHS_FILE, "rb") as f:
            self._conn_graphs = pickle.load(f)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_original_map(self):
        habitat_maps.get_topdown_map_from_sim
        top_down_map = maps.get_top_down_map(
            self._sim,
            self._config.MAP_RESOLUTION,
            self._meters_per_pixel,
        )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._scene_id = episode.scene_id.split("/")[-2]
        self._step_count = 0
        self._metric = None
        self._meters_per_pixel = habitat_maps.calculate_meters_per_pixel(
            self._config.MAP_RESOLUTION, self._sim
        )
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]
        a_x, a_y = habitat_maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                np.array([a_x, a_y]),
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / habitat_maps.calculate_meters_per_pixel(
                    self._config.MAP_RESOLUTION, sim=self._sim
                ),
            )

        if self._config.DRAW_FIXED_WAYPOINTS:
            maps.draw_mp3d_nodes(
                self._top_down_map,
                self._sim,
                episode,
                self._conn_graphs[scene_id],
                self._meters_per_pixel,
            )

        if self._config.DRAW_SHORTEST_PATH:
            shortest_path_points = self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[0].position
            )
            maps.draw_straight_shortest_path_points(
                self._top_down_map,
                self._sim,
                self._config.MAP_RESOLUTION,
                shortest_path_points,
            )

        if self._config.DRAW_REFERENCE_PATH:
            maps.draw_reference_path(
                self._top_down_map,
                self._sim,
                episode,
                self._config.MAP_RESOLUTION,
                self._meters_per_pixel,
            )

        # draw source and target points last to avoid overlap
        if self._config.DRAW_SOURCE_AND_TARGET:
            maps.draw_source_and_target(
                self._top_down_map,
                self._sim,
                episode,
                self._meters_per_pixel,
            )

        # MP3D START NODE
        self._nearest_node = maps.get_nearest_node(
            self._conn_graphs[scene_id], np.take(agent_position, (0, 2))
        )
        nn_position = self._conn_graphs[self._scene_id].nodes[
            self._nearest_node
        ]["position"]
        self.s_x, self.s_y = habitat_maps.to_grid(
            nn_position[2],
            nn_position[0],
            self._top_down_map.shape[0:2],
            self._sim,
        )
        self.update_metric(episode, action=None)

    def update_metric(self, *args: Any, **kwargs: Any):
        self._step_count += 1
        (
            house_map,
            map_agent_pos,
        ) = self.update_map(self._sim.get_agent_state().position)

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_agent_pos,
            "agent_angle": self.get_polar_angle(),
            "bounds": {
                k: v
                for k, v in zip(
                    ["lower", "upper"],
                    self._sim.pathfinder.get_bounds(),
                )
            },
            "meters_per_px": self._meters_per_pixel,
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = habitat_maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            self._sim,
        )
        # Don't draw over the source point
        gradient_color = 15 + min(
            self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
        )
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            maps.drawline(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                gradient_color,
                thickness=int(
                    self._config.MAP_RESOLUTION
                    * 1.4
                    / maps.MAP_THICKNESS_SCALAR
                ),
                style="filled",
            )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                np.array([a_x, a_y]),
                self.get_polar_angle(),
                self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / habitat_maps.calculate_meters_per_pixel(
                    self._config.MAP_RESOLUTION, sim=self._sim
                ),
            )

        point_padding = int(0.2 / self._meters_per_pixel)
        prev_nearest_node = self._nearest_node
        self._nearest_node = maps.update_nearest_node(
            self._conn_graphs[self._scene_id],
            self._nearest_node,
            np.take(agent_position, (0, 2)),
        )
        if (
            self._nearest_node != prev_nearest_node
            and self._config.DRAW_MP3D_AGENT_PATH
        ):
            nn_position = self._conn_graphs[self._scene_id].nodes[
                self._nearest_node
            ]["position"]
            (prev_s_x, prev_s_y) = (self.s_x, self.s_y)
            self.s_x, self.s_y = habitat_maps.to_grid(
                nn_position[2],
                nn_position[0],
                self._top_down_map.shape[0:2],
                self._sim,
            )
            self._top_down_map[
                self.s_x
                - int(2.0 / 3.0 * point_padding) : self.s_x
                + int(2.0 / 3.0 * point_padding)
                + 1,
                self.s_y
                - int(2.0 / 3.0 * point_padding) : self.s_y
                + int(2.0 / 3.0 * point_padding)
                + 1,
            ] = gradient_color

            maps.drawline(
                self._top_down_map,
                (prev_s_y, prev_s_x),
                (self.s_y, self.s_x),
                gradient_color,
                thickness=int(
                    1.0
                    / 2.0
                    * np.round(
                        self._config.MAP_RESOLUTION / maps.MAP_THICKNESS_SCALAR
                    )
                ),
            )

        self._previous_xy_location = (a_y, a_x)
        map_agent_pos = (a_x, a_y)
        return self._top_down_map, map_agent_pos
