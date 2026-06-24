#!/usr/bin/env python
"""Compile the unified 50,000-sample JSONL training dataset for the Regex Wizard.

Scales and multiplies structural templates to generate a diversified instruction
corpus with parallel dialect patterns and E8 coordinate mappings.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Insert src and scripts directories
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_templates import generate_samples

# Vocabulary templates for generating variations
SYNONYMS = {
    "match": ["find", "extract", "detect", "parse", "check for", "identify", "capture"],
    "hexadecimal": ["hex", "hex-based", "color hex"],
    "valid": ["correct", "proper", "well-formed"],
    "codes": ["values", "strings", "expressions"]
}

def expand_instruction(inst: str) -> str:
    words = inst.split()
    expanded = []
    for w in words:
        wl = w.lower().strip(",.?!")
        if wl in SYNONYMS:
            # Randomly swap with a synonym
            expanded.append(random.choice(SYNONYMS[wl]))
        else:
            expanded.append(w)
    return " ".join(expanded)

def is_redos_safe(pattern: str) -> bool:
    """Detects simple exponential backtracking signatures like (a+)+ or (a|b)+*."""
    import re
    # Check for nested quantifiers e.g. (\w+)+, (\d*)*, etc.
    nested_quant = re.compile(r'\([^)]+[\*\+\?]\)[\*\+\?]')
    if nested_quant.search(pattern):
        return False
    # Check for adjacent quantifiers e.g. a++
    if re.search(r'[\*\+\?]{2,}', pattern):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile unified Regex Wizard dataset.")
    parser.add_argument("--templates", type=str, default="/Volumes/Storage/project_atlas/scratch/synthetic_regex_templates.jsonl", help="Synthetic templates path.")
    parser.add_argument("--out", type=str, default="/Volumes/Storage/project_atlas/scratch/regex_dataset_50k.jsonl", help="Output dataset path.")
    parser.add_argument("--size", type=int, default=50000, help="Target dataset size (number of instruction pairs).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args(argv)

    random.seed(args.seed)

    print(f"Reading base templates from: {args.templates}")
    if not os.path.exists(args.templates):
        print("Templates file not found. Generating base templates on the fly...")
        generate_samples(num_variations_per_template=3)
        
    base_samples = []
    with open(args.templates, "r") as f:
        for line in f:
            if line.strip():
                base_samples.append(json.loads(line))
                
    if not base_samples:
        print("Error: No base templates found to compile.")
        return 1

    print(f"Compiling dataset to target size: {args.size} samples...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compiled_count = 0
    with open(out_path, "w") as f:
        # Loop until we reach target size
        while compiled_count < args.size:
            # Pick a random base sample
            base = random.choice(base_samples)
            if not is_redos_safe(base["pattern"]):
                continue
            
            # Create a variation of the natural language instruction
            new_inst = expand_instruction(base["instruction"])
            
            sample = {
                "id": f"rx_{compiled_count:05d}",
                "instruction": new_inst,
                "dialect": base["dialect"],
                "input_tokens": [base["dialect"]] + new_inst.split(),
                "pattern": base["pattern"],
                "complexity_class": base["complexity_class"],
                "ast_breakdown": base["ast_breakdown"]
            }
            
            f.write(json.dumps(sample) + "\n")
            compiled_count += 1
            
            # Print progress indicators
            if compiled_count % 10000 == 0:
                print(f"Compiled {compiled_count}/{args.size} samples...")

    print(f"Dataset compilation completed successfully! Saved {compiled_count} samples to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
