import sys
import numpy as np
import torch

class MockConfig:
    def __init__(self, vocab_size=200000):
        self.vocab_size = vocab_size

class MockEmbedding:
    def __init__(self, vocab_size=200000, dim=16):
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, dim))

class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits

class MockModel(torch.nn.Module):
    def __init__(self, vocab_size=200000, dim=16):
        super().__init__()
        self.config = MockConfig(vocab_size)
        self.embed_tokens = MockEmbedding(vocab_size, dim)
        self.vocab_size = vocab_size
        self.dim = dim

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, *args, **kwargs):
        # input_ids shape: (batch, seq_len)
        seq_len = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids)
        seq_len = max(1, seq_len)
        logits = torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32)
        # return a mock output
        return MockModelOutput(logits)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class MockTokenizer:
    def __init__(self, vocab_size=200000):
        self.vocab_size = vocab_size

    def encode(self, prompt, *args, **kwargs):
        if not prompt:
            return []
        # Return token IDs that are both > 100 (high tid enforcer) and < vocab_size (1000 in mocks)
        return [500 + (ord(c) % 400) for c in prompt]

    def decode(self, ids, *args, **kwargs):
        if not ids:
            return ""
        return "".join(chr(32 + (x % 95)) for x in ids)

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, path_or_repo, *args, **kwargs):
        return MockTokenizer()

class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, path_or_repo, *args, **kwargs):
        return MockModel()
