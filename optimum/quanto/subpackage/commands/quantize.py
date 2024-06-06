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
"""Hugging Face models quantization command-line interface class."""

from typing import TYPE_CHECKING

import torch

from optimum.commands import BaseOptimumCLICommand
from optimum.exporters import TasksManager

from ...models import QuantizedTransformersModel


if TYPE_CHECKING:
    from argparse import ArgumentParser


SUPPORTED_LIBRARIES = ["transformers"]


def parse_quantize_args(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "output",
        type=str,
        help="The path to save the quantized model.",
    )
    required_group.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Hugging Face Hub model id or path to a local model.",
    )
    required_group.add_argument(
        "--weights",
        type=str,
        default="int8",
        choices=["int2", "int4", "int8", "float8"],
        help="The Hugging Face library to use to load the model.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The Hugging Face model revision.",
    )
    optional_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading the model.",
    )
    optional_group.add_argument(
        "--library",
        type=str,
        default=None,
        choices=SUPPORTED_LIBRARIES,
        help="The Hugging Face library to use to load the model.",
    )
    optional_group.add_argument(
        "--task",
        type=str,
        default=None,
        help="The model task (useful for models supporting multiple tasks).",
    )
    optional_group.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="The torch dtype to use when loading the model weights.",
    )
    optional_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use when loading the model.",
    )


class QuantizeCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_quantize_args(parser)

    def run(self):
        model_name_or_path = self.args.model
        library_name = self.args.library
        if library_name is None:
            library_name = TasksManager.infer_library_from_model(model_name_or_path)
        if library_name not in SUPPORTED_LIBRARIES:
            raise ValueError(
                f"{library_name} models are not supported by this CLI, but can be quantized using the python API directly."
            )
        task = self.args.task
        if task is None:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        torch_dtype = self.args.torch_dtype
        if torch_dtype != "auto":
            torch_dtype = torch.float16 if self.args.torch_dtype == "fp16" else torch.bfloat16
        model = TasksManager.get_model_from_task(
            task,
            model_name_or_path,
            revision=self.args.revision,
            trust_remote_code=self.args.trust_remote_code,
            framework="pt",
            torch_dtype=torch_dtype,
            device=torch.device(self.args.device),
            library_name=library_name,
            low_cpu_mem_usage=True,
        )
        weights = f"q{self.args.weights}"
        qmodel = QuantizedTransformersModel.quantize(model, weights=weights)
        qmodel.save_pretrained(self.args.output)
