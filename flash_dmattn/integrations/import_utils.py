# Copyright 2025 Jingze Shi and the HuggingFace Team. All rights reserved.
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
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import Union


from transformers import is_torch_available
from transformers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # TODO: Once python 3.9 support is dropped, `importlib.metadata.packages_distributions()`
            # should be used here to map from package name to distribution names
            # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark; onnxruntime -> onnxruntime-gpu.
            # `importlib.metadata.packages_distributions()` is not available in Python 3.9.

            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            elif pkg_name == "quark":
                # TODO: remove once `importlib.metadata.packages_distributions()` is supported.
                try:
                    package_version = importlib.metadata.version("amd-quark")
                except Exception:
                    package_exists = False
            elif pkg_name == "triton":
                try:
                    package_version = importlib.metadata.version("pytorch-triton")
                except Exception:
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists



@lru_cache
def is_flash_dmattn_available():
    if not is_torch_available():
        return False

    if not _is_package_available("flash_dmattn"):
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    return True