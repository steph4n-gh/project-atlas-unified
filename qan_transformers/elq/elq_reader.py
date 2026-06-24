import json
import numpy as np
import os
from qan_transformers.elq.format import MAGIC

class ELQReader:
    """
    Memory-efficient reader for ELQ format files.
    Performs binary file seeks to load individual tensor byte ranges on-demand
    instead of loading the entire file payload into memory.
    """
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "rb")
        
        # Verify magic signature
        magic = self.file.read(4)
        if magic != MAGIC:
            self.file.close()
            raise ValueError(f"Invalid magic number in ELQ file: {magic}")
            
        # Read header length
        self.header_len = int(np.frombuffer(self.file.read(4), dtype=np.uint32)[0])
        
        # Read and decode JSON header
        header_bytes = self.file.read(self.header_len)
        self.header = json.loads(header_bytes.decode("utf-8"))
        
        # Calculate payload offset
        self.payload_offset = 8 + self.header_len
        self.metadata = self.header.get("metadata", {})
        
    def read_tensor(self, layer_name: str, tensor_name: str) -> np.ndarray:
        """
        Retrieves a specific tensor by layer name and tensor name on demand.
        Uses binary seek to read only the required bytes from disk.
        """
        layer_tensors = self.header["layers"].get(layer_name)
        if not layer_tensors:
            # Fallback for prefix variations or normalization mismatches
            for k, v in self.header["layers"].items():
                if k == layer_name or k.endswith("." + layer_name) or layer_name.endswith("." + k):
                    layer_tensors = v
                    break
                    
        if not layer_tensors or tensor_name not in layer_tensors:
            return None
            
        tensor_info = layer_tensors[tensor_name]
        start = tensor_info["start"]
        end = tensor_info["end"]
        shape = tensor_info["shape"]
        dtype_str = tensor_info["dtype"]
        
        # Seek directly to the tensor position in payload
        self.file.seek(self.payload_offset + start)
        raw_bytes = self.file.read(end - start)
        
        # Parse from buffer and reshape
        arr = np.frombuffer(raw_bytes, dtype=dtype_str).copy()
        return arr.reshape(shape)
        
    def close(self):
        """
        Closes the underlying binary file.
        """
        if self.file and not self.file.closed:
            self.file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
