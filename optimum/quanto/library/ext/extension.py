import os
from typing import List

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
            os.makedirs(self.build_directory, exist_ok=True)
            self._lib = load(
                name=self.name,
                sources=self.sources,
                extra_cflags=self.extra_cflags,
                extra_cuda_cflags=self.extra_cuda_cflags,
                build_directory=self.build_directory,
            )
        return self._lib
