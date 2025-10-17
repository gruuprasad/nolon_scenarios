# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.wheeled_robots.controllers import WheelBasePoseController
from isaacsim.core.api.physics_context import PhysicsContext

from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions

import asyncio
import numpy as np
import carb

class HandWaveController(BaseController):
    def __init__(self):
        super().__init__(name="hand_wave_controller")
        self._increment_step = np.pi/16
        self._index_map = {"pan": 6, "lift": 7}
        self._lower_limit = -np.pi/4
        self._upper_limit = np.pi/4

    def forward(self, command, current_joint_position) -> ArticulationAction:
        print("HandWaveController::forward() called: ", command)
        if command not in self._index_map.keys():
            return
        if current_joint_position == self._lower_limit:
            self._increment_step = np.pi/24
        elif current_joint_position == self._upper_limit:
            self._increment_step = -np.pi/24

        current_joint_position += self._increment_step
        print("new joint position = ", current_joint_position)
        # only updating pan_joint for now
        action = ArticulationActions(joint_positions=np.array([[current_joint_position, 0]]),
                                    joint_indices=[self._index_map[command], 7])
        return action

    def is_done(self):
        return True


class HandWaving(BaseTask):
    def __init__(self, name):
        print("HandWaving::__init__():called")
        BaseTask.__init__(self, name=name, offset=None)
        self._ur5_robot = None
        self._ur5_asset_path = "/home/ubuntu/nolon/assets/ur5.usd"
        self.number_of_waves = 3
        self.num_envs = 1

    def set_up_scene(self, scene: Scene) -> None:
        print("HandWaving::set_up_scene():called")
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=self._ur5_asset_path, prim_path="/World/Scene/Ur5")
        self._ur5_robot = scene.add(
            Articulation(
                prim_paths_expr="/World/Scene/Ur5",
                name="my_ur5",
                positions=np.array([[0, 0.0, 0.02]]),
                orientations=np.array([euler_angles_to_quat(np.array([0, 0, -3*np.pi/4]))])
            )
        )


    def get_observations(self) -> dict:
        print("HandWaving::get_observations():called")
        positions = self._ur5_robot.get_joint_positions()[0]
        return {
            "my_ur5": {
                "shoulder_pan_joint": positions[6],
                "shoulder_lift_joint": positions[7]
            }
        }

    def pre_step(self, time_step_index, sim_time):
        BaseTask.pre_step(self, time_step_index, sim_time)
        return

    def post_reset(self):
        return

    def cleanup(self):
        return

    def get_params(self) -> dict:
        params_representation = dict()
        params_representation["number_of_waves"] = {"value": self.number_of_waves, "modifiable": False}
        params_representation["robot_name"] = {"value": self._ur5_robot.name, "modifiable": False}
        return params_representation


class MoveArm(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._my_robot = None

        return

    def setup_scene(self):
        print("MoveArm::setup_scene():called")

        world = self.get_world()
        world.add_task(HandWaving(name="hand_waving"))
        return

    async def setup_post_load(self):
        print("MoveArm::setup_post_load():called")
        self._ur5_task = self._world.get_task(name="hand_waving")

        self._task_params = self._ur5_task.get_params()
        self._my_robot = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._controller = HandWaveController()

        self._world.add_physics_callback("sim_step", self.on_pbysics_step)

        # stiffness and dampness for the joints
        current_kp, current_kd = self._my_robot.get_gains(joint_indices=np.array([6, 7]))
        current_max_effort = self._my_robot.get_max_efforts(joint_indices=np.array([6, 7]))
        # values by default on loading usd.
        print("stiffness = ", current_kp)               #stiffness =  [[290362.97 219451.75]]
        print("dampness = ", current_kd)                #dampness =  [[116.14518  87.7807 ]]
        print("max_effort = ", current_max_effort)      #max_effort =  [[10. 10.]]
        stiffness = np.tile(np.array([500, 800]), (1, 1))
        dampings = np.tile(np.array([50, 100]), (1, 1))
        max_efforts = np.tile(np.array([100, 300]), (1, 1))
        self._my_robot.set_gains(kps=stiffness, kds=dampings, joint_indices=np.array([6, 7]))
        self._my_robot.set_max_efforts(max_efforts, joint_indices=np.array([6, 7]))
        #self._my_robot.set_joints_default_state(
        #    positions=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        #)

        return

    def on_pbysics_step(self, step_size):
        observations = self._world.get_observations()
        shoulder_lift_state = observations[self._task_params["robot_name"]["value"]]["shoulder_lift_joint"]
        shoulder_pan_state = observations[self._task_params["robot_name"]["value"]]["shoulder_pan_joint"]

        action = self._controller.forward("pan", shoulder_pan_state)
        self._my_robot.apply_action(action)

        current_kp, current_kd = self._my_robot.get_gains(joint_indices=np.array([6, 7]))
        current_max_effort = self._my_robot.get_max_efforts(joint_indices=np.array([6, 7]))
        print("stiffness = ", current_kp)
        print("dampness = ", current_kd)
        print("current_max_effort = ", current_max_effort)

        return

    async def setup_pre_reset(self):
        print("MoveArm::setup_pre_reset():called")
        if self._world.physics_callback_exists("sim_step"):
            self._world.remove_physics_callback("sim_step")
        return

    def world_cleanup(self):
        print("MoveArm::cleanup():called")
        self._controller = None
        return
