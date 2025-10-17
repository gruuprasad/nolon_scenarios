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

import numpy as np
import carb

class HandWaveController(BaseController):
    def __init__(self):
        super().__init__(name="hand_wave_controller")

    def forward(self, command: str, current_joint_positions: np.ndarray, joint_indices=None) -> ArticulationAction:
        if command == "move":
            print("move comamdn received")
            pass  #TODO
        elif command == "still":
            print(current_joint_positions)
        else:
            return

        action = ArticulationAction(joint_positions=current_joint_positions, joint_indices=joint_indices)
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

    def set_up_scene(self, scene: Scene) -> None:
        print("HandWaving::set_up_scene():called")
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=self._ur5_asset_path, prim_path="/World/Scene/Ur5")
        self._ur5_robot = scene.add(
            WheeledRobot(
                prim_path="/World/Scene/Ur5",
                name="my_ur5",
                wheel_dof_names=[
                    "fl_wheel_joint",
                    "fr_wheel_joint",
                    "rl_wheel_joint",
                    "rr_wheel_joint",
                ],
                create_robot=False,
                usd_path=self._ur5_asset_path,
                position=np.array([0, 0.0, 0.02]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self._ur5_robot.set_joints_default_state(
            positions=np.array([0, 0, 0, 0, 0, -np.pi / 2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
        )

    def get_observations(self) -> dict:
        print("HandWaving::get_observations():called")
        joints_state = self._ur5_robot.get_joints_state()
        return {
            "my_ur5": {
                "joint_positions": joints_state.positions
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
        self._articulation_controller = None
        self._controller = None

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
        my_ur5 = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._articulation_controller = my_ur5.get_articulation_controller()
        self._controller = HandWaveController()

        self._world.add_physics_callback("sim_step", self.on_pbysics_step)
        await self._world.play_async()
        return

    def on_pbysics_step(self, step_size):
        observations = self._world.get_observations()
        # TODO add action here.
        actions = self._controller.forward("still", current_joint_positions=observations[self._task_params["robot_name"]["value"]]["joint_positions"])
        if self._controller.is_done():
            self._world.pause()
        if actions is not None:
            self._articulation_controller.apply_action(actions)
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
