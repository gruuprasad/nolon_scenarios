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

import omni.kit.commands

import numpy as np
import carb

class MoveArm(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()

        # Setting up import configuration:
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.distance_scale = 1.0

        # Get path to extension data:
        # Import URDF, prim_path contains the path the path to the usd prim in the stage.
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="/home/gp/nolon_fs/nolon/src/ur5_amr/src/robotic_arms_control/urdf/ur5.urdf",
            import_config=import_config,
            get_articulation_root=True,
        )

        self._save_count = 0

        self._scene = PhysicsContext()
        self._scene.set_physics_dt(1 / 30.0)

        return

    async def setup_post_load(self):
        self._world = self.get_world()

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        carb.log_info("setup_post_load():Success")

        return

    def send_robot_actions(self, step_size):

        self._save_count += 1

        return

    async def setup_pre_reset(self):
        if self._world.physics_callback_exists("sending_actions"):
            self._world.remove_physics_callback("sending_actions")
        self._save_count = 0
        self._world.pause()
        return

    async def setup_post_reset(self):
        await self._world.play_async()
        self._world.pause()
        return

    def world_cleanup(self):
        self._world.pause()
        return
