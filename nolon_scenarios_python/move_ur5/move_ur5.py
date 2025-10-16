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

import numpy as np
import carb

# Note: checkout the required tutorials at https://docs.isaacsim.omniverse.nvidia.com/latest/index.html


class MoveUR5(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._robot_path = "/home/ubuntu/nolon/assets/ur5_base.usd"
        self._arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        add_reference_to_stage(usd_path=self._robot_path, prim_path="/World/Ur5")

        self._ur5 = world.scene.add(
            WheeledRobot(
                prim_path="/World/Ur5",
                name="my_ur5",
                wheel_dof_names=[
                    "fl_wheel_joint",
                    "fr_wheel_joint",
                    "rl_wheel_joint",
                    "rr_wheel_joint",
                ],
                create_robot=False,
                usd_path=self._robot_path,
                position=np.array([0, 0.0, 0.02]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        self._save_count = 0

        self._scene = PhysicsContext()
        self._scene.set_physics_dt(1 / 30.0)

        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._diff_controller = DifferentialController(
            name="simple_control",
            wheel_radius=0.07,
            wheel_base=0.6
        )

        self._diff_controller.reset()
        self._ur5.initialize()

        # Get their indices
        self._arm_joint_indices = [
            self._ur5.get_dof_index(joint_name) for joint_name in self._arm_joint_names
        ]
        print(self._arm_joint_names)
        print(self._arm_joint_indices)
        print(self._ur5.get_joints_default_state().positions)
        self._ur5_base.set_joints_default_state(
            positions=np.array([0, 0, 0, 0, 0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )
        print(self._ur5.get_joint_positions())

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        carb.log_info("setup_post_load():Success")

        return

    def send_robot_actions(self, step_size):

        self._save_count += 1

        wheel_action = None

        # linear X, angular Z commands
        if self._save_count <= 100:
            wheel_action = self._diff_controller.forward(command=[1.0, 0.0])
        elif self._save_count <= 250:
            wheel_action = self._diff_controller.forward(command=[-1.0, 0.0])
        elif self._save_count <= 300:
            wheel_action = self._diff_controller.forward(command=[1.2, -0.8])
        elif self._save_count <= 400:
            wheel_action = self._diff_controller.forward(command=[-1.0, 1.5])

        if wheel_action:
            wheel_action.joint_velocities = np.hstack((wheel_action.joint_velocities, wheel_action.joint_velocities))
            print(wheel_action)
            self._ur5.apply_wheel_actions(wheel_action)
        return

    async def setup_pre_reset(self):
        if self._world.physics_callback_exists("sending_actions"):
            self._world.remove_physics_callback("sending_actions")
        self._save_count = 0
        self._world.pause()
        return

    async def setup_post_reset(self):
        self._diff_controller.reset()
        await self._world.play_async()
        self._world.pause()
        return

    def world_cleanup(self):
        self._world.pause()
        return
