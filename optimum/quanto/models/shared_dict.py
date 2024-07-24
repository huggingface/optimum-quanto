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

import os
from collections.abc import Mapping
from typing import Any, Dict

from safetensors import safe_open


class ShardedStateDict(Mapping):
    """A pytorch state_dict stored in multiple safetensors files

    This class implements the `collections.abc.Mapping` interface.
    It can be passed to `torch.nn.Module.load_state_dict()` to recursively
    load the module tensors.
    """

    def __init__(self, base_dir: str, tensor_index: Dict[str, str]):
        self._base_dir = base_dir
        self._index = tensor_index
        self._handles = {}

    def __iter__(self):
        yield from self._index

    def __len__(self):
        return self._index.__len__()

    def __getitem__(self, key: Any) -> Any:
        filename = self._index.__getitem__(key)
        if filename not in self._handles:
            f = safe_open(os.path.join(self._base_dir, filename), framework="pytorch")
            self._handles[filename] = f
        f = self._handles[filename]
        return f.get_tensor(key)

    def __contains__(self, key: object) -> bool:
        return self._index.__contains__(key)

    def keys(self):
        return self._index.keys()
