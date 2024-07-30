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

"""
Adapted from
https://github.com/huggingface/diffusers/blob/69e72b1dd113927ed638f26e82738e9735385edc/src/diffusers/utils/hub_utils.py#L503C1-L601C14
"""

import os
import tempfile
from typing import Optional, Union

from huggingface_hub import create_repo, upload_folder


class PushToHubMixin:
    """
    A Mixin to push a model, scheduler, or pipeline to the Hugging Face Hub.
    """

    def _upload_folder(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all files in `working_dir` to `repo_id`.
        """
        if commit_message is None:
            commit_message = f"Upload {self.__class__.__name__}"

        return upload_folder(
            repo_id=repo_id, folder_path=working_dir, token=token, commit_message=commit_message, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your model, scheduler, or pipeline files to. It should
                contain your organization name when pushing to an organization. `repo_id` can also be a path to a local
                directory.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. The token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights to the `safetensors` format.
        """
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id

        # Save all files.
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)

            return self._upload_folder(
                tmpdir,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
