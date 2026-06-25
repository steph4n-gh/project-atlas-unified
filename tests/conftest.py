import os
import sys
from unittest.mock import MagicMock

# Redirect Hugging Face cache directories to the external storage volume for all tests
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")

# Mock mlx modules if they cannot be imported (e.g. on Linux CPU environments)
try:
    import mlx.core as mx
except ImportError:
    class MLXMock(MagicMock):
        @property
        def __name__(self):
            return "mlx"

    sys.modules["mlx"] = MLXMock()
    sys.modules["mlx.core"] = MagicMock()
    sys.modules["mlx.nn"] = MagicMock()
    sys.modules["mlx.optimizers"] = MagicMock()
