import os
import sys
from unittest.mock import MagicMock
import importlib.machinery

# Redirect Hugging Face cache directories to the external storage volume for all tests
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")

# Mock mlx modules if they cannot be imported (e.g. on Linux CPU environments)
try:
    import mlx.core as mx
except ImportError:
    mlx_mock = MagicMock()
    mlx_mock.__spec__ = importlib.machinery.ModuleSpec("mlx", None)
    mlx_mock.__name__ = "mlx"
    sys.modules["mlx"] = mlx_mock

    mlx_core_mock = MagicMock()
    mlx_core_mock.__spec__ = importlib.machinery.ModuleSpec("mlx.core", None)
    mlx_core_mock.__name__ = "mlx.core"
    sys.modules["mlx.core"] = mlx_core_mock

    mlx_nn_mock = MagicMock()
    mlx_nn_mock.__spec__ = importlib.machinery.ModuleSpec("mlx.nn", None)
    mlx_nn_mock.__name__ = "mlx.nn"
    sys.modules["mlx.nn"] = mlx_nn_mock

    mlx_optimizers_mock = MagicMock()
    mlx_optimizers_mock.__spec__ = importlib.machinery.ModuleSpec("mlx.optimizers", None)
    mlx_optimizers_mock.__name__ = "mlx.optimizers"
    sys.modules["mlx.optimizers"] = mlx_optimizers_mock

    mlx_utils_mock = MagicMock()
    mlx_utils_mock.__spec__ = importlib.machinery.ModuleSpec("mlx.utils", None)
    mlx_utils_mock.__name__ = "mlx.utils"
    sys.modules["mlx.utils"] = mlx_utils_mock
