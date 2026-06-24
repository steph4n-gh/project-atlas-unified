import json
import os
from pathlib import Path
from typing import List, Union, Set
import mlx.core as mx

def prune_model_layers(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    layers_to_drop: List[int]
) -> None:
    """
    Offline layer pruner for MLX/safetensors models.
    Modifies configuration files and rewrites weight metadata directly on disk
    without loading the entire model into active VRAM.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Load config.json
    config_file = input_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Could not find config.json in {input_dir}")
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    # Get config layout
    text_config = config.get("text_config", config)
    orig_num_layers = text_config.get("num_hidden_layers")
    if orig_num_layers is None:
        raise ValueError("Could not find num_hidden_layers in config")
        
    # Determine remaining layers and setup mapping
    prune_set = set(layers_to_drop)
    remaining_indices = [i for i in range(orig_num_layers) if i not in prune_set]
    new_num_layers = len(remaining_indices)
    
    # Map old layer index -> new layer index
    layer_mapping = {old: new for new, old in enumerate(remaining_indices)}
    
    print(f"[Pruner] Pruning {len(prune_set)} layers out of {orig_num_layers} layers.")
    print(f"[Pruner] Remaining layers (old indices): {remaining_indices}")
    print(f"[Pruner] New layer count: {new_num_layers}")
    
    # 2. Update config.json properties
    text_config["num_hidden_layers"] = new_num_layers
    
    # Update layer types list if present
    if "layer_types" in text_config:
        text_config["layer_types"] = [
            text_config["layer_types"][i] for i in remaining_indices
        ]
        
    # Update any other layer-dependent configs (e.g. rotary embeddings)
    if "rope_scaling" in text_config:
        pass  # Adjust as needed depending on specific scaling patterns
        
    # Save updated config.json
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Copy other supporting files (tokenizers, templates, generation config, etc.)
    copy_supporting_files(input_path, output_path)
    
    # 3. Process weights
    index_file = input_path / "model.safetensors.index.json"
    if index_file.exists():
        process_multifile_weights(input_path, output_path, index_file, prune_set, layer_mapping)
    else:
        process_singlefile_weights(input_path, output_path, prune_set, layer_mapping)
        
    print("[Pruner] Pruning successfully completed!")

def copy_supporting_files(input_path: Path, output_path: Path) -> None:
    import shutil
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "special_tokens_map.json"
    ]
    for filename in files_to_copy:
        src = input_path / filename
        if src.exists():
            shutil.copy2(src, output_path / filename)

def map_weight_key(key: str, prune_set: Set[int], layer_mapping: dict) -> Union[str, None]:
    """
    Maps an old weight key to a new weight key depending on remaining layers.
    Returns None if key belongs to a pruned layer.
    """
    parts = key.split(".")
    # Safetensors layers usually structured as model.layers.<index>.* or layers.<index>.*
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                layer_index = int(parts[idx + 1])
                if layer_index in prune_set:
                    return None
                parts[idx + 1] = str(layer_mapping[layer_index])
                return ".".join(parts)
            except ValueError:
                continue
    return key

def process_singlefile_weights(
    input_path: Path,
    output_path: Path,
    prune_set: Set[int],
    layer_mapping: dict
) -> None:
    weight_file = input_path / "model.safetensors"
    # Fallback to weights.npz if safetensors doesn't exist
    is_npz = False
    if not weight_file.exists():
        weight_file = input_path / "weights.npz"
        is_npz = True
    if not weight_file.exists():
        # Maybe it is split, but check other possible names
        weight_file = input_path / "model.safetensors"
        
    print(f"[Pruner] Loading single-file weights from {weight_file.name}")
    weights = mx.load(str(weight_file))
    
    pruned_weights = {}
    for k, v in weights.items():
        new_key = map_weight_key(k, prune_set, layer_mapping)
        if new_key is not None:
            pruned_weights[new_key] = v
            
    out_name = "weights.npz" if is_npz else "model.safetensors"
    print(f"[Pruner] Saving pruned weights to {out_name}")
    mx.save_safetensors(str(output_path / out_name), pruned_weights)

def process_multifile_weights(
    input_path: Path,
    output_path: Path,
    index_file: Path,
    prune_set: Set[int],
    layer_mapping: dict
) -> None:
    print("[Pruner] Multi-file safetensors index detected. Parsing map...")
    with open(index_file, "r") as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    new_weight_map = {}
    files_to_rewrite = set()
    
    # Pre-calculate files and key mappings
    for k, filename in weight_map.items():
        new_key = map_weight_key(k, prune_set, layer_mapping)
        if new_key is not None:
            new_weight_map[new_key] = filename
            files_to_rewrite.add(filename)
            
    # Save the new index map
    new_index_data = {"metadata": index_data.get("metadata", {}), "weight_map": new_weight_map}
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(new_index_data, f, indent=4)
        
    # Process each weight file sequentially to save memory
    for filename in sorted(list(files_to_rewrite)):
        src_file = input_path / filename
        dst_file = output_path / filename
        print(f"[Pruner] Processing {filename}...")
        
        weights = mx.load(str(src_file))
        new_weights = {}
        for k, v in weights.items():
            new_key = map_weight_key(k, prune_set, layer_mapping)
            if new_key is not None:
                new_weights[new_key] = v
                
        mx.save_safetensors(str(dst_file), new_weights)
