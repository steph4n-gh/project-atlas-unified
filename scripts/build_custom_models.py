#!/usr/bin/env python
"""Master orchestration script to sequentially build and train 5 custom UCE models.

Ensures clean memory recovery by running subprocesses for each compile/precompute/distill stage.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)

# 1. Define the custom configurations for the 5 models
CONFIGS = [
    {
        "name": "transformer_2b",
        "gemma_model": "google/gemma-4-E2B-it",
        "p": 8,
        "depth": 3,
        "max_tokens": 256,
        "seed_prompt": (
            "attention softmax mlp residual layernorm learning_rate gradient loss "
            "backprop weights bias projection embeddings self_attention multi_head "
            "feed_forward query key value optimizer adamw backpropagation perplexity "
            "entropy tokenizer activation"
        ),
        "corpus_text": """
Large language models are based on the transformer architecture. The core mechanism is self_attention,
which maps a query and a set of key-value pairs to an output.
The attention function is calculated using:
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
Multi_head attention projects the query, key, and value vectors multiple times.
Each head performs the self_attention operation in parallel, allowing the model to attend to information
from different representation subspaces simultaneously.

The outputs of multi_head attention are concatenated and projected again.
A residual connection is added around the attention block, followed by layernorm:
x = layernorm(x + attention(x)).
The output then goes through a feed_forward network (also known as mlp), which consists of two linear
layers with an activation function (like gelu or swiglu) in between:
mlp(x) = linear(activation(linear(x))).
Another residual connection and layernorm are applied:
output = layernorm(x + mlp(x)).

During training, we minimize a cross-entropy loss function.
We compute the gradient of the loss with respect to the model weights and bias.
Using backpropagation (backprop), the gradient is propagated backwards through the layers.
The optimizer (typically adamw) updates the weights using the calculated gradient and learning_rate:
weights = weights - learning_rate * adjusted_gradient.
The tokenizer maps input characters to token embeddings.
Perplexity and entropy are used to measure the uncertainty of the token predictions.
By optimizing the weights, the model learns to output highly accurate next-token probabilities.
"""
    },
    {
        "name": "quantum_2b",
        "gemma_model": "google/gemma-4-E2B-it",
        "p": 8,
        "depth": 3,
        "max_tokens": 256,
        "seed_prompt": (
            "schrodinger hamiltonian eigenstate superposition entanglement wavefunction "
            "relativity thermodynamics entropy fermion boson photon spacetime quantum "
            "particle spin observables operators hilbert_space coherent decoherence "
            "cosmology black_hole graviton"
        ),
        "corpus_text": """
Quantum mechanics describes the physical properties of nature at the scale of atoms and subatomic particles.
The fundamental equation of quantum mechanics is the schrodinger equation:
H |psi> = E |psi>.
Here, H is the hamiltonian operator, representing the total energy of the system, psi is the wavefunction,
and E represents the energy eigenvalues.
The wavefunction represents the state of a quantum particle.
A quantum system can exist in a superposition of multiple states until it is measured.
When a quantum measurement occurs, the wavefunction collapses into a specific eigenstate.

Quantum entanglement is a phenomenon where two or more particles become correlated such that the state
of one particle instantly determines the state of the other, regardless of distance.
The state space of a quantum system is represented mathematically by a hilbert_space.
Observables are represented by hermitian operators.
Spin is an intrinsic form of angular momentum carried by elementary particles.
Fermions (like electrons) have half-integer spin and obey the Pauli exclusion principle.
Bosons (like photons or the graviton) have integer spin and can occupy the same quantum state.

Thermodynamics and cosmology study macro-scale systems.
Entropy is a measure of molecular disorder.
In cosmology, general relativity describes the curvature of spacetime due to mass and energy.
A black_hole is a region of spacetime where gravity is so strong that nothing, not even a photon, can escape.
Stephen Hawking showed that a black_hole can emit thermal radiation, implying it has a temperature and entropy.
At the quantum boundary, physicists study how a coherent quantum state undergoes decoherence
due to interaction with the environment, transitioning quantum behavior into classical physics.
"""
    },
    {
        "name": "algebra_4b",
        "gemma_model": "google/gemma-4-e4b-it",
        "p": 8,
        "depth": 3,
        "max_tokens": 256,
        "seed_prompt": (
            "homomorphism isomorphism automorphism subgroup abelian permutation torsion "
            "ideal quotient manifold topology category functor normal_subgroup coset "
            "kernel cyclic sylow ring_theory galois_field algebraic geometric"
        ),
        "corpus_text": """
Abstract algebra is the study of algebraic structures such as groups, rings, and fields.
A group is a set equipped with a binary operation satisfying closure, associativity, identity, and invertibility.
An abelian group is a group where the operation is commutative.
A subgroup is a subset that forms a group under the same operation.
If a subgroup is a normal_subgroup, we can define a quotient group using its cosets:
G/N = { gN : g in G }.
A cyclic group is generated by a single element.
The permutation group consists of all bijective mappings of a set to itself.
Sylow theorems describe the subgroups of prime power order in finite groups.

A mapping between two groups that preserves the operation is a group homomorphism.
The kernel of a homomorphism is the set of elements mapped to the identity.
If a homomorphism is bijective, it is called an isomorphism.
An automorphism is an isomorphism from a mathematical object to itself.
In ring_theory, an ideal is a special subset of a ring that allows the construction of quotient rings.
A field extension like a galois_field is crucial for solving polynomial equations.

Topology and geometry study continuous spaces.
A manifold is a topological space that locally resembles Euclidean space.
Topology defines properties preserved under continuous deformations, such as compactness and connectedness.
In category theory, a category consists of objects and morphisms.
A functor is a mapping between categories that preserves structure.
Algebraic and geometric properties of topological spaces are studied using homology and fundamental groups.
An element is called torsion if it has finite order.
"""
    },
    {
        "name": "graph_4b",
        "gemma_model": "google/gemma-4-e4b-it",
        "p": 8,
        "depth": 3,
        "max_tokens": 256,
        "seed_prompt": (
            "dijkstra algorithm complexity recursion directed acyclic traversal heuristic "
            "sorting search shortest_path adjacency vertices edges depth_first "
            "breadth_first avl_tree red_black dynamic_programming memoization "
            "asymptotic np_complete"
        ),
        "corpus_text": """
An algorithm is a step-by-step procedure for solving a computational problem.
We analyze the efficiency of an algorithm using asymptotic notation (Big-O complexity).
Graph algorithms operate on a set of vertices and edges.
A graph can be directed or undirected.
A directed acyclic graph (DAG) contains no directed cycles, allowing topological sorting.
Common graph traversal methods include depth_first search (DFS) and breadth_first search (BFS).
An adjacency list or matrix represents the connections between vertices.

To find the shortest_path in a weighted graph, we use dijkstra's algorithm.
Dijkstra uses a priority queue to greedily explore the closest vertices.
A* search uses a heuristic function to guide the shortest_path search towards the goal.
For self-balancing search trees, we use an avl_tree or a red_black tree,
which guarantee O(log n) operations for search, insertion, and deletion.

Dynamic_programming resolves complex problems by breaking them into overlapping subproblems.
It uses memoization to store intermediate results, avoiding redundant calculation.
Many algorithms use recursion, where a function calls itself with smaller inputs.
Sorting algorithms (like quicksort or mergesort) rearrange elements in a specific order.
Computational complexity divides problems into classes like P and NP.
A problem is np_complete if it is in NP and any problem in NP can be reduced to it in polynomial time.
"""
    }
]

def run_cmd(cmd: list[str]) -> bool:
    print(f"\nRunning command: {' '.join(cmd)}")
    sys.stdout.flush()
    res = subprocess.run(cmd, cwd=str(ROOT))
    return res.returncode == 0

def main():
    print("=== STARTING SEQUENTIAL UCE MODEL COMPILATION AND DISTILLATION GOAL ===")
    print(f"Working Directory: {ROOT}")
    print(f"Output Directory: {TMP}")
    sys.stdout.flush()

    for idx, cfg in enumerate(CONFIGS, 1):
        name = cfg["name"]
        
        tree_path = TMP / f"tree_{name}.json"
        cache_path = TMP / f"cache_{name}.pkl"
        model_path = TMP / f"uce_{name}.safetensors"

        if model_path.exists() and tree_path.exists():
            print(f"\n[skip] Model '{name}' already built. Skipping Stage {idx}/{len(CONFIGS)}.")
            continue

        print(f"\n========================================================")
        print(f" STAGE {idx}/{len(CONFIGS)}: Building and training model '{name}'")
        print(f"========================================================")
        sys.stdout.flush()

        # 2. Write custom corpus file
        corpus_path = TMP / f"corpus_{name}.txt"
        corpus_path.write_text(cfg["corpus_text"].strip(), encoding="utf-8")
        print(f"[*] Wrote custom corpus file: {corpus_path}")

        # 3. Compile the tree (lazy dequant of local snapshot embeddings)
        print(f"[*] Inducing tree from {cfg['gemma_model']}...")
        compile_cmd = [
            sys.executable, "scripts/build_tree_from_gemma.py",
            "--gemma-model", cfg["gemma_model"],
            "--p", str(cfg["p"]),
            "--depth", str(cfg["depth"]),
            "--max-tokens", str(cfg["max_tokens"]),
            "--seed-prompt", cfg["seed_prompt"],
            "--out", str(tree_path)
        ]
        if not run_cmd(compile_cmd):
            print(f"[-] ERROR: Tree induction failed for stage {name}. Exiting.")
            return 1

        # 4. Precompute target states & logits (CPU-based transformers to avoid VRAM OOM)
        print(f"[*] Pre-computing dataset cache...")
        precompute_cmd = [
            sys.executable, "scripts/precompute_dataset.py",
            "--tree-config", str(tree_path),
            "--gemma-model", cfg["gemma_model"],
            "--num-tokens", "2000" if "12b" not in name else "1000",
            "--corpus-file", str(corpus_path),
            "--out", str(cache_path)
        ]
        if not run_cmd(precompute_cmd):
            print(f"[-] ERROR: Dataset pre-computation failed for stage {name}. Exiting.")
            return 1

        # 5. Train the UCE model (using the saved cache, completely offline)
        print(f"[*] Distilling UCE model...")
        distill_cmd = [
            sys.executable, "scripts/run_distillation.py",
            "--tree-config", str(tree_path),
            "--cached-dataset", str(cache_path),
            "--phase", "1",
            "--steps", "60",
            "--out", str(model_path)
        ]
        if not run_cmd(distill_cmd):
            print(f"[-] ERROR: UCE model distillation failed for stage {name}. Exiting.")
            return 1

        print(f"[+] Distillation complete! Saved: {model_path}")
        # Clean up cache and corpus to preserve space
        try:
            cache_path.unlink()
            corpus_path.unlink()
            print(f"[*] Cleaned up temporary cache and corpus files.")
        except Exception:
            pass

    print("\n========================================================")
    print("=== ALL 5 CUSTOM MODELS BUILT AND DISTILLED SUCCESSFULLY ===")
    print("========================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
