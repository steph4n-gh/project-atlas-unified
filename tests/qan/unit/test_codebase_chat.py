import os
import torch
import torch.nn as nn
import pytest
from scratch.run_codebase_chat import wrap_rotary_embeddings
from qan_transformers.math.context_builder import crawl_codebase

def test_codebase_crawler(tmp_path):
    """
    Verifies that crawl_codebase recursively traverses files,
    includes supported file extensions, and ignores directories like .git and .venv.
    """
    # Create mock directory structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('hello')", encoding="utf-8")
    (src_dir / "index.js").write_text("console.log('test')", encoding="utf-8")
    
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("some git config", encoding="utf-8")
    
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "lib.py").write_text("some library", encoding="utf-8")
    
    doc_file = tmp_path / "README.md"
    doc_file.write_text("# Readme content", encoding="utf-8")
    
    # Run crawler
    files_dict = crawl_codebase(str(tmp_path))
    
    # Assertions
    assert "src/main.py" in files_dict
    assert "src/index.js" in files_dict
    assert "README.md" in files_dict
    
    # Ensure ignored folders are not crawled
    for path in files_dict.keys():
        assert ".git" not in path
        assert ".venv" not in path
        
    assert files_dict["src/main.py"] == "print('hello')"
    assert files_dict["src/index.js"] == "console.log('test')"
    assert files_dict["README.md"] == "# Readme content"

class MockRotaryEmbedding(nn.Module):
    def __init__(self, max_seq_len_cached=128):
        super().__init__()
        self.max_seq_len_cached = max_seq_len_cached
        
    def forward(self, x, position_ids, **kwargs):
        # Index directly to simulate out-of-bounds error if not wrapped
        if (position_ids >= self.max_seq_len_cached).any():
            raise IndexError("Rotary cache index out of bounds!")
        return x, position_ids

class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_emb = MockRotaryEmbedding()

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"max_position_embeddings": 128})()
        self.attn = MockAttention()

def test_rope_modulo_wrapping():
    """
    Verifies that wrap_rotary_embeddings successfully wraps position IDs
    using modulo to prevent index out of bounds errors.
    """
    model = MockModel()
    wrap_rotary_embeddings(model)
    
    x = torch.randn(1, 10, 64)
    # Test position ID larger than max_seq_len_cached (128)
    position_ids = torch.tensor([[150]])
    
    # This should succeed due to modulo wrapping (150 % 128 = 22 < 128)
    _, out_pos = model.attn.rotary_emb(x, position_ids)
    assert out_pos.item() == 22
    
    # Test position ID equal to max_seq_len_cached
    position_ids_edge = torch.tensor([[128]])
    _, out_pos_edge = model.attn.rotary_emb(x, position_ids_edge)
    assert out_pos_edge.item() == 0
