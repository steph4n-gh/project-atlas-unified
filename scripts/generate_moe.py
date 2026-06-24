#!/usr/bin/env python
"""Task MoE CLI: interact with the UCE 8-Expert Mixture of Experts system.

Features:
- Dynamic gating / domain classification.
- Dynamic page-swapping of expert weights via mmap.
- Interactive chat loop or single-shot prompt generation.
"""

import argparse
import sys
from pathlib import Path

# Insert src directory
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrametric_ce.moe import UCEMoeRouter

def interactive_chat(router: UCEMoeRouter, max_new: int, temp: float, seed: int):
    print("\n========================================================")
    print("      UCE 8-Expert Mixture of Experts (MoE) Chat")
    print("========================================================")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to end.")
    print("VRAM active footprints are capped. Swapping happens on the fly.\n")
    
    while True:
        try:
            prompt = input("User > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break
            
            print("AI > ", end="")
            sys.stdout.flush()
            
            output = router.generate(
                prompt=prompt,
                max_new_tokens=max_new,
                temperature=temp,
                seed=seed,
                verbose=True
            )
            print(f"\nResult: {output}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\n[error] Generation failed: {e}\n")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UCE Mixture of Experts generation CLI.")
    parser.add_argument(
        "--moe-dir",
        type=str,
        default="tmp/moe",
        help="Directory containing the expert checkpoints."
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Gemma model id to resolve the tokenizer."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single prompt and exit."
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=30,
        help="Max new tokens to generate."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Generation temperature."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=16,
        help="Expert state dimension."
    )

    args = parser.parse_args(argv)

    moe_dir = ROOT / args.moe_dir
    if not moe_dir.exists():
        print(f"ERROR: MoE directory not found: {moe_dir}", file=sys.stderr)
        print("Please build and train the experts first by running compiler script.", file=sys.stderr)
        return 2

    # Instantiate the MoE Router
    print(f"[*] Initializing UCEMoeRouter (moe_dir={moe_dir})...")
    try:
        router = UCEMoeRouter(
            moe_dir=moe_dir,
            gemma_model_id=args.gemma_model,
            dim=args.dim
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize MoE Router: {e}", file=sys.stderr)
        return 1

    if args.prompt:
        # Single shot generation
        try:
            output = router.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_new,
                temperature=args.temp,
                seed=args.seed,
                verbose=True
            )
            print(f"\nResult: {output}")
        except Exception as e:
            print(f"ERROR: Generation failed: {e}", file=sys.stderr)
            return 1
    else:
        # Interactive mode
        interactive_chat(router, args.max_new, args.temp, args.seed)

    return 0

if __name__ == "__main__":
    sys.exit(main())
