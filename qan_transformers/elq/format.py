import json
import os
import numpy as np

MAGIC = b"ELQ1"

def align_to_128(size: int) -> int:
    return (size + 127) & ~127

def save_elq(path: str, layers_dict: dict, metadata: dict = None) -> None:
    """
    Saves ELQ quantized layers into a custom binary format at `path`.
    Uses a JSON header for extensibility and ensures all tensor offsets in the
    binary payload start at 128-byte aligned boundaries relative to the file.
    
    `layers_dict` should map layer names to a dictionary containing:
        - "scales": np.ndarray (float16)
        - "indices": np.ndarray (uint32)
        - "outliers": np.ndarray (float16)
        - "outlier_mask": np.ndarray (bool)
    """
    if metadata is None:
        metadata = {}
        
    header = {
        "metadata": metadata,
        "layers": {}
    }
    
    # Pre-calculate payload sizes and offsets to construct header
    current_payload_offset = 0
    tensor_data_list = []
    
    # Sort keys for deterministic file generation
    for layer_name in sorted(layers_dict.keys()):
        layer_tensors = layers_dict[layer_name]
        header["layers"][layer_name] = {}
        
        for name in ["scales", "indices", "outliers", "outlier_mask"]:
            arr = layer_tensors[name]
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            
            # Record dtype and shape
            dtype_str = str(arr.dtype)
            shape = list(arr.shape)
            
            # Get raw bytes
            raw_bytes = arr.tobytes()
            size = len(raw_bytes)
            
            # Align start offset
            aligned_start = align_to_128(current_payload_offset)
            pad_size = aligned_start - current_payload_offset
            
            if pad_size > 0:
                tensor_data_list.append(b"\x00" * pad_size)
                current_payload_offset = aligned_start
                
            start_offset = current_payload_offset
            end_offset = start_offset + size
            
            header["layers"][layer_name][name] = {
                "shape": shape,
                "dtype": dtype_str,
                "start": start_offset,
                "end": end_offset
            }
            
            tensor_data_list.append(raw_bytes)
            current_payload_offset = end_offset

    # Serialize JSON header
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)
    
    # Header length + magic (4 bytes) + header_len size (4 bytes)
    # Total prefix size = 8 + header_len
    # We want the payload start offset in the file to be 128-byte aligned.
    total_prefix_size = 8 + header_len
    aligned_prefix_size = align_to_128(total_prefix_size)
    header_padding = aligned_prefix_size - total_prefix_size
    
    # Build final header block
    full_header = header_bytes + b" " * header_padding
    
    # Write to file
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(np.uint32(len(full_header)).tobytes())
        f.write(full_header)
        # Write payload
        for chunk in tensor_data_list:
            f.write(chunk)

def load_elq(path: str) -> tuple[dict, dict]:
    """
    Loads ELQ quantized layers from `path`.
    Returns (layers_dict, metadata).
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic number in ELQ file: {magic}")
            
        header_len = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        
        # Read the rest of the file as raw payload bytes using zero-copy memoryview
        payload = memoryview(f.read())
        
    layers_dict = {}
    metadata = header.get("metadata", {})
    
    for layer_name, layer_tensors in header["layers"].items():
        layers_dict[layer_name] = {}
        for name, tensor_info in layer_tensors.items():
            start = tensor_info["start"]
            end = tensor_info["end"]
            shape = tensor_info["shape"]
            dtype_str = tensor_info["dtype"]
            
            raw_bytes = payload[start:end]
            arr = np.frombuffer(raw_bytes, dtype=dtype_str).copy()
            arr = arr.reshape(shape)
            layers_dict[layer_name][name] = arr
            
    return layers_dict, metadata
