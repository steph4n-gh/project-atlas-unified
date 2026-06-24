import time
import torch
import torch.nn as nn
import numpy as np
import json
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB, CoWMemorySwapGridDB
from qan_transformers.math.e8_projection import generate_e8_coordinates
from qan_transformers.firewall.cohomology import CohomologyFirewall
from scratch.demo_real_model import graft_huggingface_model

def encode_hypothesis_to_e8_coord(hypothesis: str, tokenizer, model, db):
    """
    Tokenizes the hypothesis, averages the token embeddings, projects to head_dim,
    and quantizes it to E8 coordinates.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    inputs = tokenizer(hypothesis, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs.input_ids)  # [1, seq_len, hidden_size]
        mean_emb = torch.mean(embeddings, dim=1)  # [1, hidden_size]
        
        # Project to head_dim of layer 0
        q_proj = model.model.layers[0].self_attn.q_proj
        head_dim = model.model.layers[0].self_attn.head_dim
        projected = q_proj(mean_emb)[:, :head_dim]
        
        # Normalize and scale to ensure the projected key doesn't collapse to the E8 origin
        projected = torch.nn.functional.normalize(projected, dim=-1) * 4.5
        
        # Swap out to E8 memory grid database
        db.swap_out_target(projected, projected)
        e8_coord = db.grid_coords[-1].cpu().numpy().tolist()
    return e8_coord

def speculate_hypothesis(agent_name, domain, problem, seeds, tokenizer, model, device, evo_seed=""):
    """
    Queries the grafted TinyLlama model to speculate on a combination of seeds and problem.
    """
    seeds_str = " and ".join(seeds)
    prompt = f"[Research Swarm] Agent: {agent_name} ({domain} Specialist).\n"
    if evo_seed:
        prompt += f"Prior Discovery Foundation: \"{evo_seed}\"\n"
    prompt += (
        f"Goal: Apply {seeds_str} to solve: {problem}.\n"
        f"Evolutionary Hypothesis: "
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=35,
            do_sample=True,
            temperature=0.85,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    clean_hypothesis = generated_text.split("\n")[0].strip()
    if not clean_hypothesis:
        clean_hypothesis = "E(s) = \\int \\Psi_{E8}(x) d\\mu_p(x) mapping topological connectivity."
    return clean_hypothesis

def main():
    print("="*80)
    print("\033[1;36m      COMBINATORIAL RESEARCH SWARM: MULTI-DOMAIN FORMULA SEARCH\033[0m")
    print("="*80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[SYSTEM] Compute backend: {device.type.upper()}")

    # 1. Load Model and Graft E8 Attention
    model_name = "nickypro/tinyllama-15M"
    print(f"[SYSTEM] Loading and grafting model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    model = graft_huggingface_model(model, sparse_ratio=0.15)

    # 2. Setup Central Database and Projections
    head_dim = model.model.layers[0].self_attn.head_dim
    shared_db = AdelicMemorySwapGridDB(d_model=head_dim, device=device.type, lock_path="/tmp/research_swarm.lock")
    shared_db.clear()

    dtype = next(model.parameters()).dtype
    W_q = torch.randn(head_dim, head_dim, device=device, dtype=dtype)
    W_k = torch.randn(head_dim, head_dim, device=device, dtype=dtype)
    shared_db.initialize_projections(W_q, W_k, is_draft=False)

    # Cohomology Firewall
    firewall = CohomologyFirewall(threshold=1.5, tau=0.05)

    # 3. Define Seeds and Domains
    math_seeds = [
        "Coxeter Concentric E8 projections",
        "Čech Cohomology obstruction rollback",
        "Vladimirov p-adic fractional derivatives",
        "Adelic Langevin MCMC optimization",
        "Discrete Morse simplicial KV collapses",
        "Gromov-Witten invariants on Calabi-Yau manifolds",
        "Atiyah-Singer Index Theorem for Dirac operators",
        "Langlands Correspondence for algebraic curves",
        "Pontryagin Duality on non-Archimedean topological groups",
        "AdS/CFT duality partition functions"
    ]

    domain_problems = {
        "Math": [
            "finding non-trivial zeros of L-functions",
            "unifying Archimedean and non-Archimedean topological groups",
            "calculating modular curve boundaries",
            "proving the Hodge Conjecture for algebraic cycles",
            "classifying topological invariants of 4-manifolds"
        ],
        "CS": [
            "sequence scaling beyond 1M tokens",
            "hardware-efficient sparse gather-scatter operations",
            "leakproof post-quantum cryptographic key exchanges",
            "optimizing compilers for topological quantum qubits",
            "leakage-free fully homomorphic lattice encryption"
        ],
        "Physics": [
            "topological protection of qubits in silicon",
            "calculating partition functions on quasicrystalline manifolds",
            "modeling quantum gravity in 8D spaces",
            "constructing anomaly-free adelic string partition functions",
            "calculating topological hall conductivity in 3D materials"
        ]
    }

    agents = [
        # Mathematics Specialists
        {"id": 1, "domain": "Math", "name": "Dr. Sophie (Algebraic Geometer)", "color": "green"},
        {"id": 2, "domain": "Math", "name": "Dr. Henri (Symplectic Topologist)", "color": "green"},
        {"id": 3, "domain": "Math", "name": "Dr. Pierre (Homological Algebraist)", "color": "green"},
        # Computer Science Specialists
        {"id": 4, "domain": "CS", "name": "Dr. Alan (Quantum Compiler Architect)", "color": "cyan"},
        {"id": 5, "domain": "CS", "name": "Dr. Claude (Lattice Cryptographer)", "color": "cyan"},
        {"id": 6, "domain": "CS", "name": "Dr. Grace (Transformer Scaling Engineer)", "color": "cyan"},
        # Physics Specialists
        {"id": 7, "domain": "Physics", "name": "Dr. Richard (Topological Qubit Physicist)", "color": "orange"},
        {"id": 8, "domain": "Physics", "name": "Dr. Emmy (String Cosmologist)", "color": "orange"},
        {"id": 9, "domain": "Physics", "name": "Dr. Paul (LQG Loop Quantum Gravity Theorist)", "color": "orange"}
    ]

    history = []
    total_steps = 4

    print(f"[SYSTEM] Swarm Initialized with {len(agents)} Domain Specialists.")
    print(f"[SYSTEM] Combined Math Seeds: {math_seeds}")
    print("-"*80)

    # 4. Combinatoric Search Loop
    for step in range(1, total_steps + 1):
        print(f"\n🔬 --- COMBINATORIAL STEP {step} ---")
        step_discoveries = []
        
        # Shuffle agents to random order of discovery
        random.shuffle(agents)
        
        for agent in agents:
            # Pick a problem matching the agent's domain
            problem = random.choice(domain_problems[agent["domain"]])
            # Choose 1-2 math seeds
            num_seeds = random.choice([1, 2])
            selected_seeds = random.sample(math_seeds, num_seeds)
            
            # Evolutionary Seed: 35% chance to fetch an approved hypothesis from prior steps
            evo_seed = ""
            if step > 1 and history and random.random() < 0.35:
                prev_step = random.choice(history)
                if prev_step["discoveries"]:
                    prev_discovery = random.choice(prev_step["discoveries"])
                    if prev_discovery["status"] == "Approved":
                        evo_seed = prev_discovery["hypothesis"]
                        
            # Autoregressively generate hypothesis using E8 attention
            hypothesis = speculate_hypothesis(
                agent["name"], agent["domain"], problem, selected_seeds, 
                tokenizer, model, device, evo_seed=evo_seed
            )
            
            # Map hypothesis to E8 coordinate
            e8_coord = encode_hypothesis_to_e8_coord(hypothesis, tokenizer, model, shared_db)
            
            # Run Cohomology check on the accumulated E8 coordinates
            coords = shared_db.grid_coords
            coords_norm = coords / (torch.linalg.norm(coords, dim=-1, keepdim=True) + 1e-6)
            attn_matrix = torch.clamp(torch.matmul(coords_norm, coords_norm.t()), min=0.0)
            
            is_fractured, cfi, alt_idx = firewall.check_obstruction(attn_matrix)
            
            # Calculate algebraic connectivity
            attn_matrix_f32 = attn_matrix.to(torch.float32)
            degrees = torch.sum(attn_matrix_f32, dim=-1).cpu().numpy()
            D = np.diag(degrees)
            W = attn_matrix_f32.cpu().numpy()
            L = D - W
            try:
                eigvals = np.linalg.eigvalsh(L)
                lambda_2 = eigvals[1] if len(eigvals) > 1 else 1.0
            except Exception:
                lambda_2 = 1.0
                
            status = "Approved" if (not is_fractured and lambda_2 >= firewall.tau) else "Fractured"
            
            print(f"  * {agent['name']} ({agent['domain']}) combined {selected_seeds} for: \"{problem}\"")
            print(f"    - Speculative Hypothesis: \"{hypothesis}\"")
            print(f"    - E8 Coordinate: {e8_coord}")
            print(f"    - Firewall Check: {status} (CFI: {cfi:.3f}, \u03bb_2: {lambda_2:.3f})")
            
            step_discoveries.append({
                "agent_name": agent["name"],
                "domain": agent["domain"],
                "color": agent["color"],
                "problem": problem,
                "seeds": selected_seeds,
                "hypothesis": hypothesis,
                "e8_coord": e8_coord,
                "cfi": float(cfi),
                "lambda_2": float(lambda_2),
                "status": status
            })

        history.append({
            "step": step,
            "discoveries": step_discoveries,
            "database_coordinates": shared_db.grid_coords.shape[0] if shared_db.grid_coords is not None else 0
        })

    # Log results to JSON
    os.makedirs("results", exist_ok=True)
    history_file = "results/research_swarm_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("\033[1;32m                  RESEARCH SWARM RUN COMPLETED\033[0m")
    print("="*80)
    print(f"Total Verified Coordinates: {shared_db.grid_coords.shape[0] if shared_db.grid_coords is not None else 0}")
    print(f"Combinatorial discoveries logged to: {history_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
