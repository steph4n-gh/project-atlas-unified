#!/usr/bin/env python3
"""Dump Python dependency graph as JSON for spectral-pruner auditing.

Reads installed packages from the current Python environment using
importlib.metadata, builds a dependency graph, and outputs a JSON file
compatible with the security-auditor tool.
"""
import importlib.metadata
import json
import re
import sys
import argparse

def normalize_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()

# Matches the package name at the start of a PEP 508 requirement string.
REQ_NAME_RE = re.compile(r'^([a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)')
# Matches extras-gated or platform-specific markers that should be excluded.
EXTRAS_RE = re.compile(r';\s*extra\s*==')

def parse_req_name(req_str):
    """Extract and normalize the package name from a PEP 508 requirement string.

    Returns None if the requirement is gated behind an extras marker
    (e.g., 'pytest; extra == "dev"') since those represent optional
    dependency groups that inflate the graph with phantom edges.
    """
    # Filter out extras-gated dependencies — they create phantom edges
    # to packages that may not be installed or relevant to production.
    if EXTRAS_RE.search(req_str):
        return None
    m = REQ_NAME_RE.match(req_str)
    if m:
        return normalize_name(m.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Dump Python dependency graph as JSON for spectral-pruner.")
    parser.add_argument("-o", "--output", default="python-deps.json", help="Output JSON path")
    parser.add_argument("--root", default="project-atlas-unified", help="Root package name to anchor the graph from")
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
    root_pkg = normalize_name(args.root)
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
    
    # 3. Build edges (filtering out extras-gated requirements)
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
    print(f"System boundaries start at index {system_start_idx} (sinks: {sorted(sink_names)})")

if __name__ == "__main__":
    main()
