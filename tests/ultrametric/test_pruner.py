import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from ultrametric_ce.pruner import prune_model_layers

class TestPruner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input_model"
        self.output_dir = Path(self.temp_dir.name) / "output_model"
        self.input_dir.mkdir()
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_single_file_pruning(self):
        # 1. Write mock config.json
        config_data = {
            "text_config": {
                "num_hidden_layers": 6,
                "layer_types": ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"],
                "model_type": "gemma4"
            }
        }
        with open(self.input_dir / "config.json", "w") as f:
            json.dump(config_data, f)
            
        # 2. Write supporting mock files to copy
        (self.input_dir / "tokenizer.json").write_text('{"mock_tokenizer": true}')
        
        # 3. Create mock weights (some are layers, some are global)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": mx.array([0.0]),
            "model.layers.1.self_attn.q_proj.weight": mx.array([1.0]),
            "model.layers.2.self_attn.q_proj.weight": mx.array([2.0]),
            "model.layers.3.self_attn.q_proj.weight": mx.array([3.0]),
            "model.layers.4.self_attn.q_proj.weight": mx.array([4.0]),
            "model.layers.5.self_attn.q_proj.weight": mx.array([5.0]),
            "model.embed_tokens.weight": mx.array([10.0])
        }
        mx.save_safetensors(str(self.input_dir / "model.safetensors"), weights)
        
        # 4. Prune layers 2 and 4
        # Drops: 2, 4
        # Remaining: 0, 1, 3, 5
        # Mapped to:
        #   0 -> 0
        #   1 -> 1
        #   3 -> 2
        #   5 -> 3
        prune_model_layers(self.input_dir, self.output_dir, [2, 4])
        
        # 5. Assertions on config
        pruned_config_path = self.output_dir / "config.json"
        self.assertTrue(pruned_config_path.exists())
        with open(pruned_config_path, "r") as f:
            pruned_config = json.load(f)
        
        text_cfg = pruned_config["text_config"]
        self.assertEqual(text_cfg["num_hidden_layers"], 4)
        self.assertEqual(text_cfg["layer_types"], ["layer0", "layer1", "layer3", "layer5"])
        
        # 6. Assertions on copied files
        self.assertTrue((self.output_dir / "tokenizer.json").exists())
        
        # 7. Assertions on pruned weights
        pruned_weights_path = self.output_dir / "model.safetensors"
        self.assertTrue(pruned_weights_path.exists())
        
        pruned_weights = mx.load(str(pruned_weights_path))
        self.assertIn("model.layers.0.self_attn.q_proj.weight", pruned_weights)
        self.assertIn("model.layers.1.self_attn.q_proj.weight", pruned_weights)
        self.assertIn("model.layers.2.self_attn.q_proj.weight", pruned_weights)
        self.assertIn("model.layers.3.self_attn.q_proj.weight", pruned_weights)
        self.assertIn("model.embed_tokens.weight", pruned_weights)
        
        # Verify layer values mapping
        self.assertEqual(pruned_weights["model.layers.0.self_attn.q_proj.weight"].item(), 0.0)
        self.assertEqual(pruned_weights["model.layers.1.self_attn.q_proj.weight"].item(), 1.0)
        self.assertEqual(pruned_weights["model.layers.2.self_attn.q_proj.weight"].item(), 3.0)
        self.assertEqual(pruned_weights["model.layers.3.self_attn.q_proj.weight"].item(), 5.0)
        self.assertEqual(pruned_weights["model.embed_tokens.weight"].item(), 10.0)
        
        # Excluded layers should not exist
        self.assertNotIn("model.layers.4.self_attn.q_proj.weight", pruned_weights)
        self.assertNotIn("model.layers.5.self_attn.q_proj.weight", pruned_weights) # index 5 is renamed to 3, so old 5 is not there.

    def test_multi_file_pruning(self):
        # 1. Write mock config.json
        config_data = {
            "text_config": {
                "num_hidden_layers": 4
            }
        }
        with open(self.input_dir / "config.json", "w") as f:
            json.dump(config_data, f)
            
        # 2. Write multiple safetensors shards
        # model-00001-of-00002.safetensors: layers 0 and 1
        # model-00002-of-00002.safetensors: layers 2 and 3
        weights_1 = {
            "model.layers.0.weight": mx.array([0.0]),
            "model.layers.1.weight": mx.array([1.0])
        }
        weights_2 = {
            "model.layers.2.weight": mx.array([2.0]),
            "model.layers.3.weight": mx.array([3.0]),
            "model.embed_tokens.weight": mx.array([10.0])
        }
        mx.save_safetensors(str(self.input_dir / "model-00001-of-00002.safetensors"), weights_1)
        mx.save_safetensors(str(self.input_dir / "model-00002-of-00002.safetensors"), weights_2)
        
        # Write index.json
        index_data = {
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00001-of-00002.safetensors",
                "model.layers.2.weight": "model-00002-of-00002.safetensors",
                "model.layers.3.weight": "model-00002-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors"
            }
        }
        with open(self.input_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f)
            
        # 3. Prune layer 1
        # Drops: 1
        # Remaining: 0, 2, 3 -> mapped to 0, 1, 2
        prune_model_layers(self.input_dir, self.output_dir, [1])
        
        # 4. Check outputs
        pruned_index_path = self.output_dir / "model.safetensors.index.json"
        self.assertTrue(pruned_index_path.exists())
        with open(pruned_index_path, "r") as f:
            pruned_index = json.load(f)
            
        weight_map = pruned_index["weight_map"]
        # check remapping in index
        self.assertIn("model.layers.0.weight", weight_map)
        self.assertIn("model.layers.1.weight", weight_map)
        self.assertIn("model.layers.2.weight", weight_map)
        self.assertIn("model.embed_tokens.weight", weight_map)
        self.assertNotIn("model.layers.3.weight", weight_map) # now max mapped index is 2
        
        # check files are written
        self.assertTrue((self.output_dir / "model-00001-of-00002.safetensors").exists())
        self.assertTrue((self.output_dir / "model-00002-of-00002.safetensors").exists())
        
        # check weights content
        p1 = mx.load(str(self.output_dir / "model-00001-of-00002.safetensors"))
        p2 = mx.load(str(self.output_dir / "model-00002-of-00002.safetensors"))
        
        # model-00001 should only have layer 0 (old 0) since old 1 is dropped
        self.assertIn("model.layers.0.weight", p1)
        self.assertNotIn("model.layers.1.weight", p1) # old 1 was dropped, old 2 goes to model-00002
        self.assertEqual(p1["model.layers.0.weight"].item(), 0.0)
        
        # model-00002 should have layer 1 (old 2) and layer 2 (old 3)
        self.assertIn("model.layers.1.weight", p2)
        self.assertIn("model.layers.2.weight", p2)
        self.assertIn("model.embed_tokens.weight", p2)
        self.assertEqual(p2["model.layers.1.weight"].item(), 2.0)
        self.assertEqual(p2["model.layers.2.weight"].item(), 3.0)
        self.assertEqual(p2["model.embed_tokens.weight"].item(), 10.0)

if __name__ == "__main__":
    unittest.main()
