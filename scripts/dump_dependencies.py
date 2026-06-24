#!/usr/bin/env python3
import importlib.metadata
import json
import re
import sys
import argparse

def normalize_name(name):
    return name.lower().replace('_', '-')

req_re = re.compile(r'^([a-zA-Z0-9_\-]+)')
def parse_req_name(req_str):
    m = req_re.match(req_str)
    if m:
        return normalize_name(m.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Dump Python dependency graph as JSON for spectral-pruner.")
    parser.add_argument("-o", "--output", default="python-deps.json", help="Output JSON path")
    parser.add_argument("--sinks", default="cffi,cryptography,anyio,uvicorn", help="Comma-separated list of packages to treat as system sinks")
    args = parser.parse_args()

    # 1. Fetch all installed distributions
    dists = list(importlib.metadata.distributions())
    
    # Map normalized name to distribution
    dist_map = {}
    for d in dists:
        name = normalize_name(d.metadata['Name'])
        dist_map[name] = d

    # Primary root package
    root_pkg = "project-atlas-unified"
    if root_pkg not in dist_map:
        print(f"Error: Primary package '{root_pkg}' is not installed in the environment.", file=sys.stderr)
        sys.exit(1)

    # 2. Build the list of all nodes
    # We want root to be index 0.
    # We want non-sink nodes first, and sink nodes at the end.
    sink_names = {normalize_name(s.strip()) for s in args.sinks.split(",")}
    
    # Filter installed packages into normal and sink packages
    normal_pkgs = []
    sink_pkgs = []
    
    # We want root_pkg to be the first normal package
    normal_pkgs.append(root_pkg)
    
    for name in sorted(dist_map.keys()):
        if name == root_pkg:
            continue
        if name in sink_names:
            sink_pkgs.append(name)
        else:
            normal_pkgs.append(name)
            
    # All nodes: normal followed by sinks
    all_nodes = normal_pkgs + sink_pkgs
    node_to_idx = {name: i for i, name in enumerate(all_nodes)}
    
    system_start_idx = len(normal_pkgs)
    
    # 3. Build edges
    edges = []
    for u_name in all_nodes:
        u_idx = node_to_idx[u_name]
        dist = dist_map[u_name]
        for req_str in (dist.requires or []):
            v_name = parse_req_name(req_str)
            if v_name and v_name in node_to_idx:
                v_idx = node_to_idx[v_name]
                edges.append([u_idx, v_idx])

    # 4. Define sinks indices
    sink_indices = [node_to_idx[name] for name in sink_pkgs]

    # 5. Output JSON payload
    payload = {
        "nodes": all_nodes,
        "edges": edges,
        "sinks": sink_indices,
        "system_start_idx": system_start_idx
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully dumped {len(all_nodes)} nodes and {len(edges)} edges to {args.output}")
    print(f"System boundaries start at index {system_start_idx} (sinks: {list(sink_names)})")

if __name__ == "__main__":
    main()
