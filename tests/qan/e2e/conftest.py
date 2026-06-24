import pytest
import torch
from qan_transformers.modeling import graft_model

@pytest.fixture
def gemma_4_e2b():
    return graft_model("google/gemma-4-e2b", lightweight=True)

@pytest.fixture
def gemma_4_e4b():
    return graft_model("google/gemma-4-e4b", lightweight=True)

@pytest.fixture
def gpt_oss_20b():
    return graft_model("openai/gpt-oss-20b", lightweight=True)

@pytest.fixture
def qwen_3_6_27b():
    return graft_model("Qwen/Qwen3.6-27B", lightweight=True)

@pytest.fixture
def qwen_3_6_35b_a3b():
    return graft_model("Qwen/Qwen3.6-35B-A3B", lightweight=True)

@pytest.fixture
def all_model_names():
    return [
        "google/gemma-4-e2b",
        "google/gemma-4-e4b",
        "openai/gpt-oss-20b",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
    ]
