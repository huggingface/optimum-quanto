import os
import shutil
from typing import List

import torch
from torch.utils.cpp_extension import load


class Extension(object):

    def __init__(
        self,
        name: str,
        root_dir: str,
        sources: List[str],
        extra_cflags: List[str] = None,
        extra_cuda_cflags: List[str] = None,
    ):
        self.name = name
        self.sources = [f"{root_dir}/{source}" for source in sources]
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.build_directory = os.path.join(root_dir, "build")
        self._lib = None

    @property
    def lib(self):
        if self._lib is None:
            # We only load the extension when the lib is required
            version_file = os.path.join(self.build_directory, "pytorch_version.txt")
            if os.path.exists(version_file):
                # The extension has already been built: check the torch version for which it was built
                with open(version_file, "r") as f:
                    pytorch_build_version = f.read().rstrip()
                    if pytorch_build_version != torch.__version__:
                        shutil.rmtree(self.build_directory)
            os.makedirs(self.build_directory, exist_ok=True)
            self._lib = load(
                name=self.name,
                sources=self.sources,
                extra_cflags=self.extra_cflags,
                extra_cuda_cflags=self.extra_cuda_cflags,
                build_directory=self.build_directory,
            )
            if not os.path.exists(version_file):
                with open(version_file, "w") as f:
                    f.write(torch.__version__)
        return self._lib
