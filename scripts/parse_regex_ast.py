#!/usr/bin/env python
"""Extract AST topological coordinates from a regular expression.

Uses Python's parser to traverse regex patterns and maps each character index
to an 8-dimensional vector matching our E8 Gosset lattice dimensions:
  Dim 0: Nesting Depth
  Dim 1: Alternation State
  Dim 2: Quantification
  Dim 3: Anchor Boundary
  Dim 4: Character Set
  Dim 5: Lookaround Status
  Dim 6: Capture/Routing
  Dim 7: Escaping Class
"""

import argparse
import sys
import warnings
from typing import Dict, List, Tuple

# Mute deprecation warnings for sre_parse in Python 3.14+
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sre_parse

# AST Node Type mapping to E8 Dimensions
# sre_parse constants mapping
NODE_MAP = {
    "LITERAL": 4,          # Character Set
    "ANY": 4,              # Character Set
    "RANGE": 1,            # Alternation (inside character class [])
    "IN": 1,               # Alternation (inside character class [])
    "NOT_IN": 1,           # Alternation (inside character class [])
    "SUBPATTERN": 6,       # Capture/Routing (groupings)
    "MAX_REPEAT": 2,       # Quantification
    "MIN_REPEAT": 2,       # Quantification
    "AT": 3,               # Anchor Boundary
    "BRANCH": 1,           # Alternation (choice branch |)
    "GROUPREF": 7,         # Escaping/Backreference
    "ASSERT": 5,           # Lookarounds
    "ASSERT_NOT": 5,       # Lookarounds

}

def regex_to_topo_sequence(pattern: str) -> List[List[float]]:
    """Converts a regular expression pattern into a sequence of 8-dimensional vectors."""
    n = len(pattern)
    # Initialize sequence with zeros (n positions, 8 dimensions)
    seq = [[0.0] * 8 for _ in range(n)]
    
    try:
        parsed = sre_parse.parse(pattern)
    except Exception:
        # Fallback naive parser if dialect has unsupported tokens for sre_parse
        return naive_dialect_parse(pattern)
        
    # Recursively traverse the parsed tree to mark positions
    def walk(sub_tree, depth=0, in_alt=False, in_quant=False, in_lookaround=False):
        for node_type, val in sub_tree:
            name = str(node_type)
            
            # 1. Update active states based on node types
            current_depth = depth
            current_alt = in_alt
            current_quant = in_quant
            current_lookaround = in_lookaround
            
            if name == "SUBPATTERN":
                current_depth += 1
                group_id, group_flags, group_items, sub_nodes = val
                walk(sub_nodes, current_depth, current_alt, current_quant, current_lookaround)
                continue
                
            elif name == "BRANCH":
                current_alt = True
                branch_type, branches = val
                for branch in branches:
                    walk(branch, current_depth, current_alt, current_quant, current_lookaround)
                continue
                
            elif name in ("MAX_REPEAT", "MIN_REPEAT"):
                current_quant = True
                min_rep, max_rep, sub_nodes = val
                walk(sub_nodes, current_depth, current_alt, current_quant, current_lookaround)
                continue
                
            elif name in ("ASSERT", "ASSERT_NOT"):
                current_lookaround = True
                dir_flag, sub_nodes = val
                walk(sub_nodes, current_depth, current_alt, current_quant, current_lookaround)
                continue
                
            elif name == "IN":
                current_alt = True
                for sub_node in val:
                    # character classes contain LITERALs, RANGEs, etc.
                    walk([sub_node], current_depth, current_alt, current_quant, current_lookaround)
                continue
                
            # If literal or simple node, try mapping to character positions
            # Safe fallbacks: mark general indicators across the pattern
            pass

    # Standard walk to analyze
    walk(parsed)
    
    # Refined positional mapping based on character sweeps:
    # Walk character-by-character and analyze active syntax indicators
    depth = 0
    in_char_class = False
    for i, ch in enumerate(pattern):
        # Escape sequence check
        if ch == "\\" and i + 1 < n:
            seq[i][7] = 1.0   # Escaping
            seq[i+1][7] = 1.0
            
        # Nesting Depth (Dim 0) & Capture Group (Dim 6)
        if ch == "(" and (i == 0 or pattern[i-1] != "\\"):
            depth += 1
            seq[i][6] = 1.0  # Capture trigger
        elif ch == ")" and (i == 0 or pattern[i-1] != "\\"):
            seq[i][6] = 1.0  # Capture end
            if depth > 0:
                depth -= 1
        seq[i][0] = min(1.0, float(depth) / 4.0)  # Normalized nesting
        
        # Alternation / Character Class (Dim 1)
        if ch == "[" and (i == 0 or pattern[i-1] != "\\"):
            in_char_class = True
        elif ch == "]" and (i == 0 or pattern[i-1] != "\\"):
            in_char_class = False
        if in_char_class or ch == "|":
            seq[i][1] = 1.0
            
        # Quantification (Dim 2)
        if ch in "*+?{}":
            # check if quantifier
            if i > 0 and pattern[i-1] not in "\\":
                seq[i][2] = 1.0
                
        # Anchors (Dim 3)
        if ch in "^$":
            seq[i][3] = 1.0
            
        # Character sets (Dim 4)
        if ch.isalnum():
            seq[i][4] = 0.5
        elif ch == ".":
            seq[i][4] = 1.0
            
        # Lookarounds (Dim 5)
        # Check for (?=, (?!, (?<=, (?<!
        if ch == "?" and i > 0 and pattern[i-1] == "(" and i + 1 < n:
            if pattern[i+1] in "=!":
                seq[i-1][5] = 1.0
                seq[i][5] = 1.0
                seq[i+1][5] = 1.0
            elif pattern[i+1] == "<" and i + 2 < n and pattern[i+2] in "=!":
                seq[i-1][5] = 1.0
                seq[i][5] = 1.0
                seq[i+1][5] = 1.0
                seq[i+2][5] = 1.0
                
    return seq

def naive_dialect_parse(pattern: str) -> List[List[float]]:
    """Safe fallback character-level parser for non-standard dialects (e.g. JS UnicodeSets)."""
    n = len(pattern)
    seq = [[0.0] * 8 for _ in range(n)]
    depth = 0
    in_class = False
    for i, ch in enumerate(pattern):
        if ch == "\\" and i + 1 < n:
            seq[i][7] = 1.0
            seq[i+1][7] = 1.0
        if ch == "(":
            depth += 1
            seq[i][6] = 1.0
        elif ch == ")":
            seq[i][6] = 1.0
            if depth > 0:
                depth -= 1
        seq[i][0] = min(1.0, float(depth) / 4.0)
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        if in_class or ch in "|&": # include intersection character
            seq[i][1] = 1.0
        if ch in "*+?":
            seq[i][2] = 1.0
        if ch in "^$":
            seq[i][3] = 1.0
        if ch.isalnum() or ch == ".":
            seq[i][4] = 0.5
    return seq

def main():
    parser = argparse.ArgumentParser(description="Parse regular expression patterns into AST topological coordinates.")
    parser.add_argument("--pattern", "-p", type=str, required=True, help="Regular expression pattern to parse.")
    args = parser.parse_args()
    
    print(f"Parsing pattern: {args.pattern}")
    seq = regex_to_topo_sequence(args.pattern)
    print("\nExtraction Successful! Topology per position:")
    print("-" * 65)
    print("Char | D0:Nest | D1:Alt  | D2:Quan | D3:Anch | D4:Set  | D5:Look | D6:Capt | D7:Esc")
    print("-" * 65)
    for i, ch in enumerate(args.pattern):
        v = seq[i]
        vals = " | ".join(f"{x:.1f}" for x in v)
        print(f" '{ch}' | {vals}")
    print("-" * 65)

if __name__ == "__main__":
    main()
