import os
import sys
from unittest.mock import MagicMock
import importlib.machinery

# Redirect Hugging Face cache directories to the external storage volume for all tests
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")

# Mock mlx modules if they cannot be imported (e.g. on Linux CPU environments)
try:
    if os.environ.get("FORCE_MLX_MOCK") == "1":
        raise ImportError("Forcing MLX Mock for testing")
    import mlx.core as mx
except ImportError:
    import pytest

    class MLXMock(MagicMock):
        def __getattr__(self, name):
            # Allow internal mock, dunder, and private attributes to avoid recursion
            if name.startswith("_") or name.startswith("mock_"):
                return super().__getattr__(name)
            # If the attribute is a registered submodule, return it
            full_name = f"{self.__name__}.{name}"
            if full_name in sys.modules:
                return sys.modules[full_name]
            # Otherwise, skip the current test because MLX is not installed
            pytest.skip(f"MLX is not installed; skipped test due to access of '{self.__name__}.{name}'", allow_module_level=True)

    # Set up mock modules
    mock_names = ["mlx", "mlx.core", "mlx.nn", "mlx.optimizers", "mlx.utils"]
    for name in mock_names:
        mock_obj = MLXMock()
        mock_obj.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mock_obj.__name__ = name
        sys.modules[name] = mock_obj

