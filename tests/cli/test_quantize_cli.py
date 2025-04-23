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

import subprocess
from tempfile import TemporaryDirectory

import pytest
from cli_helpers import requires_optimum_cli

from optimum.quanto import quantization_map


@requires_optimum_cli
@pytest.mark.parametrize("weights", ["int4", "int8"])
def test_export_decoder_cli(weights):
    from optimum.quanto import QuantizedModelForCausalLM

    model_id = "facebook/opt-125m"
    with TemporaryDirectory() as tempdir:
        subprocess.run(
            [
                "optimum-cli",
                "quanto",
                "quantize",
                "--model",
                model_id,
                "--weights",
                f"{weights}",
                tempdir,
            ],
            shell=False,
            check=True,
        )
        # Verify we can reload the quantized model
        qmodel = QuantizedModelForCausalLM.from_pretrained(tempdir)
        qmap = quantization_map(qmodel)
        for layer_qconfig in qmap.values():
            assert layer_qconfig["weights"] == f"q{weights}"
