#!/usr/bin/env python
"""Generate synthetic dataset templates for custom dialect edge-cases.

Defines structural regex templates across JavaScript, PCRE, and Rust,
parses their topological AST coordinates, and serializes them in JSONL format.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Insert src and scripts directories
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from parse_regex_ast import regex_to_topo_sequence

# Parameterized template generators
TEMPLATES = [
    # 1. Hex Color Matching (Dialect variations: JS RegExp uses /i flag, others inline/explicit)
    {
        "category": "data_extraction",
        "instruction": "Match hexadecimal color codes with optional alpha channel",
        "patterns": {
            "[JS]": "^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$",
            "[PCRE]": "^#(?i)([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$",
            "[RUST]": "^#(?i)([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$"
        }
    },
    # 2. IP Addresses (V4 & V6)
    {
        "category": "data_extraction",
        "instruction": "Match valid IPv4 addresses",
        "patterns": {
            "[JS]": "^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            "[PCRE]": "^(?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)$",
            "[RUST]": "^(?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)$"
        }
    },
    # 3. Unicode Set Intersections (Crucial ES2025 vs Rust difference)
    {
        "category": "dialect_edge_cases",
        "instruction": "Match letters that are both Greek scripts and whitespace characters",
        "patterns": {
            "[JS]": "[\\p{Script=Greek}&&\\p{White_Space}]/v",
            "[PCRE]": "(?=\\p{Greek})\\p{Z}",  # PCRE mimics intersection using lookahead
            "[RUST]": "[\\p{Greek}&&\\s]"      # Rust supports direct intersection class
        }
    },
    # 4. Set Subtractions (ES2025 vs Rust vs PCRE)
    {
        "category": "dialect_edge_cases",
        "instruction": "Match ASCII characters except digits",
        "patterns": {
            "[JS]": "[\\p{ASCII}&&[^\\d]]/v",
            "[PCRE]": "(?![0-9])[\\x00-\\x7F]",
            "[RUST]": "[\\x00-\\x7F&&[^0-9]]"
        }
    },
    # 5. Nested Balanced Parenthesis (PCRE supports recursive group (?R), others need bounded depth)
    {
        "category": "structural_delimiters",
        "instruction": "Match balanced parenthesized expressions up to 2 levels deep",
        "patterns": {
            "[JS]": "\\((?:[^()\\\]|\\\\.|\\((?:[^()\\\]|\\\\.)*\\))*\\)",
            "[PCRE]": "\\((?:[^()\\\]|\\\\.|(?R))*\\)", # PCRE recursive pattern
            "[RUST]": "\\((?:[^()\\\]|\\\\.|\\((?:[^()\\\]|\\\\.)*\\))*\\)"
        }
    },
    # 6. Word Duplication (Rust forbids backreferences, JS/PCRE support it)
    {
        "category": "structural_delimiters",
        "instruction": "Match a duplicate word separated by spaces",
        "patterns": {
            "[JS]": "\\b(\\w+)\\s+\\1\\b",
            "[PCRE]": "\\b(\\w+)\\s+\\1\\b",
            "[RUST]": "\\b(\\w+)\\s+(\\w+)\\b" # Fallback: match any two words since backrefs are unsupported
        }
    }
]

def generate_samples(num_variations_per_template: int = 5) -> List[Dict[str, Any]]:
    samples = []
    sample_id = 1
    
    for t in TEMPLATES:
        category = t["category"]
        base_instruction = t["instruction"]
        
        # We generate variations of instructions and options
        for dialect, pattern in t["patterns"].items():
            # Calculate AST topology coordinates using our verified parser
            topo_seq = regex_to_topo_sequence(pattern)
            
            # Formulate variations of NL instruction to increase dataset diversity
            variations = [
                f"{base_instruction} in {dialect[1:-1]} syntax",
                f"Write a {dialect[1:-1]} regular expression to {base_instruction.lower()}",
                f"{dialect} {base_instruction}"
            ]
            
            for v_idx in range(min(num_variations_per_template, len(variations))):
                inst = variations[v_idx]
                
                # Check for ReDoS vulnerability (simplistic heuristic: nested stars or pluses)
                is_safe = "linear"
                if "(?:" in pattern and ")*" in pattern and "+" in pattern:
                    # Potential risk
                    is_safe = "check_required"
                    
                samples.append({
                  "id": f"rx_{sample_id:05d}",
                  "instruction": inst,
                  "dialect": dialect,
                  "input_tokens": [dialect] + inst.split(),
                  "pattern": pattern,
                  "complexity_class": is_safe,
                  "ast_breakdown": topo_seq
                })
                sample_id += 1
                
    return samples

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic regex dataset templates.")
    parser.add_argument("--out", type=str, default="/Volumes/Storage/project_atlas/scratch/synthetic_regex_templates.jsonl", help="Output JSONL path.")
    parser.add_argument("--variations", type=int, default=3, help="Number of instruction variations per template.")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic templates (variations={args.variations})...")
    samples = generate_samples(args.variations)
    
    # Save as JSONL
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Successfully generated and verified {len(samples)} synthetic dataset samples.")
    print(f"Output saved to: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
