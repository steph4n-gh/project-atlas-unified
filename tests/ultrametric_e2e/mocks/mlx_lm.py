import mlx.core as mx
import sys
import types
from transformers import MockTokenizer

class MockMlxEmbedTokens:
    def __init__(self, vocab_size=200000, dim=16):
        self.weight = mx.zeros((vocab_size, dim))

class MockMlxSubModel:
    def __init__(self, vocab_size=200000, dim=16):
        self.embed_tokens = MockMlxEmbedTokens(vocab_size, dim)

class MockMlxModel:
    def __init__(self, vocab_size=200000, dim=16):
        self.model = MockMlxSubModel(vocab_size, dim)
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids)
        seq_len = max(1, seq_len)
        return mx.zeros((1, seq_len, self.vocab_size))

def load(path_or_repo, *args, **kwargs):
    return MockMlxModel(), MockTokenizer()

# Create a mock utils module dynamically to support load_model and load_tokenizer
utils = types.ModuleType("mlx_lm.utils")
utils.load_model = lambda path_or_repo, lazy=False, strict=True, model_config=None, *args, **kwargs: (MockMlxModel(), None)
utils.load_tokenizer = lambda path_or_repo, *args, **kwargs: MockTokenizer()

sys.modules["mlx_lm.utils"] = utils
