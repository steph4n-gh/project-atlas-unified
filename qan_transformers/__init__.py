import os

# Redirect Hugging Face cache directories to the external storage volume
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")

from qan_transformers.modeling import make_quasicrystalline
from qan_transformers.modeling.auto import AutoQANGraftModel
from qan_transformers.math import (
    generate_e8_coordinates,
    project_e8_to_quasicrystal,
    verify_quasicrystalline_symmetries
)

