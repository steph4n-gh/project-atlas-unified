import pytest
import os
import json
import mlx.core as mx
from pathlib import Path

@pytest.fixture(scope="session")
def mock_gemma_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("mock_gemma")
    # Generate random weights for model
    weights = {
        "model.embed_tokens.weight": mx.random.normal((2000, 16))
    }
    mx.save_safetensors(str(tmp_dir / "model.safetensors"), weights)
    
    # Save simple config
    config = {
        "vocab_size": 2000,
        "hidden_size": 16,
        "architectures": ["Gemma2ForCausalLM"]
    }
    (tmp_dir / "config.json").write_text(json.dumps(config))
    return tmp_dir

@pytest.fixture(scope="session")
def env_with_mocks():
    env = os.environ.copy()
    root_dir = Path(__file__).resolve().parents[2]
    mocks_dir = root_dir / "tests" / "ultrametric_e2e" / "mocks"
    # Prepend mocks directory and root directory to PYTHONPATH
    env["PYTHONPATH"] = f"{mocks_dir}:{root_dir}"
    env["HF_HOME"] = str(root_dir / "tmp" / "fake_hf_home")
    return env
