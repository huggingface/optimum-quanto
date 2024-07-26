# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union


def is_transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None


def is_diffusers_available() -> bool:
    return importlib.util.find_spec("diffusers") is not None


if is_transformers_available():
    from .transformers_models import *


if is_diffusers_available():
    from .diffusers_models import *
