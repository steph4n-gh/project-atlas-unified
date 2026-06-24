import argparse
import sys
from pathlib import Path

# Add src to python path to import pruner
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultrametric_ce.pruner import prune_model_layers

def main():
    parser = argparse.ArgumentParser(description="Prune layers from a Gemma-4/Transformer model checkpoint on disk.")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Path to input model checkpoint directory.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Path to output pruned model checkpoint directory.")
    parser.add_argument("--drop-layers", "-d", type=str, default=None, help="Comma-separated list of layer indices to drop (e.g. 5,11,17).")
    parser.add_argument("--drop-every", "-n", type=int, default=None, help="Drop every N-th layer (e.g. 3 to drop layers 2, 5, 8, 11...).")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)
        
    layers_to_drop = []
    
    # 1. Parse drop list if provided
    if args.drop_layers:
        try:
            layers_to_drop = [int(x.strip()) for x in args.drop_layers.split(",") if x.strip()]
        except ValueError:
            print("Error: --drop-layers must be a comma-separated list of integers.")
            sys.exit(1)
            
    # 2. Parse drop-every if provided
    elif args.drop_every:
        if args.drop_every <= 1:
            print("Error: --drop-every must be greater than 1.")
            sys.exit(1)
            
        # We need to know num of layers to generate indices.
        # Quick parse config.json
        import json
        config_file = input_path / "config.json"
        if not config_file.exists():
            print(f"Error: Could not find config.json in {args.input_dir} to calculate layers.")
            sys.exit(1)
        with open(config_file, "r") as f:
            config = json.load(f)
        text_config = config.get("text_config", config)
        num_layers = text_config.get("num_hidden_layers")
        if num_layers is None:
            print("Error: Could not read num_hidden_layers from config.json.")
            sys.exit(1)
            
        # E.g. drop-every 3: drops 2, 5, 8, 11... (index % 3 == 2)
        layers_to_drop = [i for i in range(num_layers) if (i + 1) % args.drop_every == 0]
        
    else:
        print("Error: Must specify either --drop-layers (-d) or --drop-every (-n).")
        sys.exit(1)
        
    if not layers_to_drop:
        print("Warning: No layers were selected for pruning. Copying model configuration directly...")
        
    try:
        prune_model_layers(args.input_dir, args.output_dir, sorted(layers_to_drop))
    except Exception as e:
        print(f"Error during pruning execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
