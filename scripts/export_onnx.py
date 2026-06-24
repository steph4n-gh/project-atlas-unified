#!/usr/bin/env python
"""Export student model weights to WebGPU-compatible binary buffers.

Saves model weights as flat float32 binary file arrays alongside a
model_layout.json mapping names and shapes, ready for direct loading into
WebGPU storage buffers in the browser.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

# Insert src directory
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export student weights to WebGPU/browser formats.")
    parser.add_argument("--weights-dir", type=str, default="/Volumes/Storage/project_atlas/scratch/student_trained", help="Path to trained student checkpoint directory.")
    parser.add_argument("--out-dir", type=str, default="/Volumes/Storage/project_atlas/scratch/web_model_payload", help="Output directory for browser assets.")
    args = parser.parse_args(argv)

    weights_path = Path(args.weights_dir) / "model.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights file {weights_path} not found.")
        return 1

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights from {weights_path}...")
    weights = mx.load(str(weights_path))

    layout = {
        "model_type": "e8_regex_wizard",
        "precision": "float32",
        "tensors": {}
    }

    print("Exporting tensors as flat binary buffers...")
    for name, tensor in weights.items():
        # Ensure float32 representation for direct Javascript Float32Array compatibility
        tensor_f32 = tensor.astype(mx.float32)
        shape = list(tensor_f32.shape)
        
        # Save raw binary bytes to file
        bin_filename = f"{name}.bin"
        bin_path = out_path / bin_filename
        
        # Convert to numpy and save raw bytes
        import numpy as np
        np_arr = np.array(tensor_f32)
        with open(bin_path, "wb") as f:
            f.write(np_arr.tobytes())
            
        layout["tensors"][name] = {
            "file": bin_filename,
            "shape": shape,
            "size_bytes": np_arr.nbytes
        }
        print(f"  Exported {name} -> {bin_filename} ({np_arr.nbytes} bytes, shape={shape})")

    # Save model_layout.json
    layout_file = out_path / "model_layout.json"
    with open(layout_file, "w") as f:
        json.dump(layout, f, indent=4)
        
    print(f"\nWebGPU export completed successfully!")
    print(f"Output files saved to: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
