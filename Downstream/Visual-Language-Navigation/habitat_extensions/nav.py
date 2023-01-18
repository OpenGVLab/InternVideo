from turtle import heading
from typing import Any

import math
import numpy as np

from habitat.core.embodied_task import (
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector


@registry.register_task_action
class MoveHighToLowAction(SimulatorTaskAction):
    def turn(self, angle):
        ''' angle: 0 ~ 360 degree '''
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount

        states = []

        if 180 < angle <= 360:
                angle -= 360
        if angle >=0:
            turn_actions = [left_action] * int(angle // turn_unit)
        else:
            turn_actions = [right_action] * int(-angle // turn_unit)

        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)
            state = self._sim.get_agent_state()
            states.append((state.position,state.rotation))

        return states

    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        
        angle = math.degrees(angle)
        states = self.turn(angle)
        states.append((init_state.position,rotation))
        self._sim.set_agent_state(init_state.position, rotation)
        
        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
                state = self._sim.get_agent_state()
                states.append((state.position,state.rotation))
            else:
                self._sim.step_without_obs(forward_action)
                state = self._sim.get_agent_state()
                states.append((state.position,state.rotation))

        output['states'] = states
        return output


@registry.register_task_action
class MoveHighToLowActionEval(SimulatorTaskAction):
    def turn(self, angle):
        ''' angle: 0 ~ 360 degree '''
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount

        states = []

        if 180 < angle <= 360:
                angle -= 360
        if angle >=0:
            turn_actions = [left_action] * int(angle // turn_unit)
        else:
            turn_actions = [right_action] * int(-angle // turn_unit)

        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)
            state = self._sim.get_agent_state()
            states.append((state.position,state.rotation))

        return states
    
    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        positions = []
        collisions = []
        forward_action = HabitatSimActions.MOVE_FORWARD

        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount
        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)

        angle = math.degrees(angle)
        states = self.turn(angle)
        states.append((init_state.position,rotation))
        self._sim.set_agent_state(init_state.position, rotation)

        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
                state = self._sim.get_agent_state()
                states.append((state.position,state.rotation))
            else:
                self._sim.step_without_obs(forward_action)
                state = self._sim.get_agent_state()
                states.append((state.position,state.rotation))

            positions.append(self._sim.get_agent_state().position)
            collisions.append(self._sim.previous_step_collided)

        output['positions'] = positions
        output['collisions'] = collisions
        output['states'] = states

        return output


@registry.register_task_action
class MoveHighToLowActionInference(SimulatorTaskAction):

    def turn(self, angle):
        ''' angle: 0 ~ 360 degree '''
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount

        states = []

        if 180 < angle <= 360:
                angle -= 360
        if angle >=0:
            turn_actions = [left_action] * int(angle // turn_unit)
        else:
            turn_actions = [right_action] * int(-angle // turn_unit)

        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)
            state = self._sim.get_agent_state()
            states.append((state.position,state.rotation))

        return states

    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        cur_path = []
        forward_action = HabitatSimActions.MOVE_FORWARD

        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount
        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)

        angle = math.degrees(angle)
        states = self.turn(angle)
        states.append((init_state.position,rotation))
        self._sim.set_agent_state(init_state.position, rotation)

        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
                cur_path.append(self.get_agent_info())
                state = self._sim.get_agent_state()
                states.append((state.position,state.rotation))
            else:
                self._sim.step_without_obs(forward_action)
                cur_path.append(self.get_agent_info())
                state = self._sim.get_agent_state()
                states.append((state.position,state.rotation))

        output['cur_path'] = cur_path
        output['states'] = states
        return output

    
    def get_agent_info(self):
        agent_state = self._sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": False,
        }