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

import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from .move_arm import MoveArm


class MoveArmExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.scenario_name = "move arm"
        self.category = "Nolon"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "move arm scenario",
            "doc_link": None,
            "overview": "This scenario shows simple arm movement of UR5.",
            "sample": MoveArm(),
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        # register the scenario with examples browser
        get_browser_instance().register_example(
            name=self.scenario_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.scenario_name, category=self.category)

        return
