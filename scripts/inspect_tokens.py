import sys
import json
from pathlib import Path

# Setup python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrametric_ce.moe import UCEMoeRouter

def main():
    r = UCEMoeRouter("tmp/moe")
    gtree, gmodel = r.experts[r.gateway_name]
    
    # Tokenize some domain names
    domains = [
        "python_coder", "web_stack", "rust_systems", "database_sql", 
        "devops_infra", "ml_tensors", "markdown_config",
        "python", "coder", "web", "stack", "rust", "systems", "database", "sql",
        "devops", "infra", "ml", "tensors", "markdown", "config"
    ]
    
    print("=== Token Mapping for Domain Names in Gateway Router Tree ===")
    
    for dom in domains:
        ids = r.tokenizer.encode(dom)
        # Handle dict or lists
        if isinstance(ids, dict) and "input_ids" in ids:
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
            
        print(f"Text: {dom!r} -> Token IDs: {ids}")
        for tid in ids:
            try:
                addr = gtree.token_to_address(tid)
                lidx = gmodel.addr_to_leaf_idx.get(addr, -1)
                print(f"  Token ID {tid}: Mapped address {addr} | Leaf index {lidx}")
            except KeyError:
                print(f"  Token ID {tid}: Not registered in gateway tree")
        print("-" * 50)

if __name__ == "__main__":
    main()
