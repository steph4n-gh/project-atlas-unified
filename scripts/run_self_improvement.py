import os

# Redirect Hugging Face cache directories to the external storage volume
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")

import sys
import time
import ast
import shutil
import subprocess
import argparse
import torch
import random
import json

# Add workspace path to system path
sys.path.append("/Volumes/Storage/project_atlas")

# ---------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------

class LocalLLMBackend:
    def __init__(self, model_name="google/gemma-4-e2b-it"):
        print(f"Loading local LLM model: {model_name} on CPU...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load on CPU to avoid MPS driver collisions during benchmarking
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cpu")
        self.model.eval()

    def generate(self, prompt):
        formatted_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.3,
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class GeminiAPIBackend:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client_type = None
        
        # Try importing new google-genai SDK
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.client_type = "new"
            print("Gemini API initialized using google-genai SDK.")
        except ImportError:
            # Try importing older google-generativeai SDK
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel('gemini-3.5-flash')
                self.client_type = "old"
                print("Gemini API initialized using google-generativeai SDK.")
            except ImportError:
                raise ImportError("Neither 'google-genai' nor 'google-generativeai' packages are installed. Install via pip.")

    def generate(self, prompt):
        if self.client_type == "new":
            response = self.client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            )
            return response.text
        elif self.client_type == "old":
            response = self.client.generate_content(prompt)
            return response.text


class MockBackend:
    def __init__(self):
        self.candidate_count = 0
        print("Mock LLM Backend initialized (will return pre-defined optimizations and variations for testing).")
        
    def generate(self, prompt):
        if "AuditorAgent" in prompt:
            return '{"approved": true, "critique": "Mock approval"}'
            
        self.candidate_count += 1
        c_idx = (self.candidate_count - 1) % 4
        
        # Extract current code from the prompt
        current_code = ""
        if "```python" in prompt:
            parts = prompt.split("```python")
            current_code = parts[1].split("```")[0].strip()
            
        if not current_code:
            return 'print("No code found")'
            
        # Candidate 0: Perfect original code
        if c_idx == 0:
            return f"```python\n{current_code}\n```"
            
        # Candidate 1: Minor variation of the original code (passes pytest and audit)
        elif c_idx == 1:
            mutated = current_code
            if "import torch" in mutated:
                mutated = mutated.replace("import torch", "import torch\n# Optimized with target transformation")
            elif "import numpy" in mutated:
                mutated = mutated.replace("import numpy", "import numpy\n# Optimized with target transformation")
            else:
                mutated = "# Optimized by speculative loop\n" + mutated
            return f"```python\n{mutated}\n```"
            
        # Candidate 2: Rejected by Čech Cohomology Firewall (calls undefined function)
        elif c_idx == 2:
            mutated = current_code
            if "def " in mutated:
                lines = mutated.split("\n")
                for i, line in enumerate(lines):
                    if "def " in line and ":" in line:
                        indent = " " * (len(line) - len(line.lstrip()) + 4)
                        lines.insert(i + 1, f"{indent}undefined_cohomological_vector_call()")
                        break
                mutated = "\n".join(lines)
            return f"```python\n{mutated}\n```"
            
        # Candidate 3: Syntax error or auditor rejection
        else:
            return "```python\nThis is a syntax error { \n```"


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def run_mps_scatter_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mps_scatter_test", module_path)
    mps_scatter_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mps_scatter_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    shapes = [
        {"B": 1, "H": 8, "S": 100, "d": 256, "K_size": 15},
        {"B": 1, "H": 8, "S": 1024, "d": 256, "K_size": 128}
    ]
    
    total_time = 0.0
    iterations = 500
    warmup = 50
    
    for shape in shapes:
        B, H, S, d, K_size = shape["B"], shape["H"], shape["S"], shape["d"], shape["K_size"]
        
        q = torch.randn(B, H, 1, d, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, S, d, device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, S, d, device=device, dtype=torch.float16, requires_grad=True)
        indices = torch.randint(0, S, (B, K_size), device=device, dtype=torch.long)
        
        for _ in range(warmup):
            K_sparse, V_sparse = mps_scatter_test.mps_coordinate_gather_scatter(q, k, v, indices)
            loss = K_sparse.sum() + V_sparse.sum()
            loss.backward()
            k.grad = None
            v.grad = None
            
        if device.type == "mps":
            torch.mps.synchronize()
            
        start_time = time.perf_counter()
        for _ in range(iterations):
            K_sparse, V_sparse = mps_scatter_test.mps_coordinate_gather_scatter(q, k, v, indices)
            loss = K_sparse.sum() + V_sparse.sum()
            loss.backward()
            k.grad = None
            v.grad = None
            
        if device.type == "mps":
            torch.mps.synchronize()
        end_time = time.perf_counter()
        
        total_time += (end_time - start_time) * 1000.0 # to ms
        
    return total_time / len(shapes)


def run_cohomology_benchmark(module_path):
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("cohomology_test", module_path)
    cohomology_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cohomology_test)
    
    firewall = cohomology_test.CohomologyFirewall(threshold=1.5, tau=0.05)
    attn = np.random.randn(1, 8, 128, 128)
    
    for _ in range(10):
        firewall.check_obstruction(attn)
        
    start_time = time.perf_counter()
    for _ in range(100):
        firewall.check_obstruction(attn)
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 100.0


def run_e8_decoder_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("e8_proj_test", module_path)
    e8_proj_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e8_proj_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    decoder = e8_proj_test.ConwaySloaneE8Decoder(device=device)
    x = torch.randn(1024, 8, device=device, dtype=torch.float32)
    
    for _ in range(20):
        decoder.decode(x)
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(200):
        decoder.decode(x)
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 200.0


def run_e8_swap_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("e8_swap_test", module_path)
    e8_swap_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e8_swap_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    db = e8_swap_test.AdelicMemorySwapGridDB(d_model=64, device=device, d_model_draft=32)
    db._init_default_projection(device, torch.float32, is_draft=False)
    db._init_default_projection(device, torch.float32, is_draft=True)
    
    keys = torch.randn(128, 64, device=device, dtype=torch.float32)
    values = torch.randn(128, 64, device=device, dtype=torch.float32)
    queries = torch.randn(1, 8, 16, 64, device=device, dtype=torch.float32)
    
    db.swap_out(keys, values)
    
    for _ in range(10):
        db.swap_in_batch(queries)
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(50):
        db.swap_in_batch(queries)
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 50.0


def run_adelic_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("adelic_test", module_path)
    adelic_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adelic_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    w1 = torch.randn(128, 64, device=device, requires_grad=True)
    w2 = torch.randn(64, 8, device=device, requires_grad=True)
    optimizer = adelic_test.AdelicLangevinOptimizer([w1, w2], lr=0.01)
    
    w1.grad = torch.randn_like(w1)
    w2.grad = torch.randn_like(w2)
    
    for _ in range(5):
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(30):
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 30.0


def run_procrustes_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("procrustes_test", module_path)
    procrustes_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(procrustes_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x_src = torch.randn(1024, 64, device=device)
    x_tgt = torch.randn(1024, 64, device=device)
    
    for _ in range(20):
        procrustes_test.compute_procrustes_alignment(x_src, x_tgt)
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(200):
        procrustes_test.compute_procrustes_alignment(x_src, x_tgt)
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 200.0


def run_rag_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_test", module_path)
    rag_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_test)
    
    indexer = rag_test.LatticeIndexer(d_model=64)
    for i in range(20):
        indexer.chunks.append({"file": "fake_file.py", "text": f"fake text chunk {i} for E8 indexing"})
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    indexer.db.grid_coords = torch.randn(20, 8, device=device)
    indexer.db._init_default_projection(device, torch.float32, is_draft=False)
    
    query = "fake query to test E8 RAG lookups"
    
    for _ in range(20):
        indexer.query(query)
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(200):
        indexer.query(query)
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000.0 / 200.0


def run_attention_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("attention_test", module_path)
    attention_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    attn = attention_test.QuasicrystallineAttention(
        embed_dim=128, num_heads=4, sparse_ratio=0.15
    ).to(device=device, dtype=torch.float32)
    
    x = torch.randn(1, 64, 128, device=device, dtype=torch.float32)
    
    for _ in range(10):
        attn(x)
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(50):
        attn(x)
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 50.0


def run_lora_pipeline_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_test", module_path)
    pipeline_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    from qan_transformers.modeling import graft_model
    model = graft_model("google/gemma-4-e2b", lightweight=True)
    
    pipeline_test.inject_lora(model, r=8, lora_alpha=16)
    model = model.to(device)
    pipeline_test.mark_only_lora_as_trainable(model)
    
    inputs = torch.randint(0, model.vocab_size, (2, 8), device=device)
    targets = torch.randint(0, model.vocab_size, (2, 8), device=device)
    data = (inputs, targets)
    
    for _ in range(2):
        pipeline_test.train_loop(model, data=data, steps=2, initial_lr=0.01)
    if device.type == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    pipeline_test.train_loop(model, data=data, steps=3, initial_lr=0.01)
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 3.0


def run_rope_wrapping_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("auto_test", module_path)
    auto_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    from qan_transformers.modeling import graft_model
    model = graft_model("google/gemma-4-e2b", lightweight=True).to(device)
    
    for _ in range(10):
        auto_test.wrap_rotary_embeddings(model)
        
    start_time = time.perf_counter()
    for _ in range(50):
        auto_test.wrap_rotary_embeddings(model)
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000.0 / 50.0


def run_precision_benchmark(target, module_path):
    if target == "mps_scatter":
        return run_mps_scatter_benchmark(module_path)
    elif target == "cohomology":
        return run_cohomology_benchmark(module_path)
    elif target == "e8_decoder":
        return run_e8_decoder_benchmark(module_path)
    elif target == "e8_swap":
        return run_e8_swap_benchmark(module_path)
    elif target == "adelic":
        return run_adelic_benchmark(module_path)
    elif target == "procrustes":
        return run_procrustes_benchmark(module_path)
    elif target == "rag":
        return run_rag_benchmark(module_path)
    elif target == "attention":
        return run_attention_benchmark(module_path)
    elif target == "lora_pipeline":
        return run_lora_pipeline_benchmark(module_path)
    elif target == "rope_wrapping":
        return run_rope_wrapping_benchmark(module_path)
    else:
        raise ValueError(f"Unknown target: {target}")



def run_unit_tests():
    """
    Runs target unit and e2e tests programmatically to ensure correctness.
    """
    res = subprocess.run(
        ["pytest", "tests/unit/", "tests/e2e/test_f1_e8_attention.py"],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")
    )
    return res.returncode == 0, res.stdout + "\n" + res.stderr


def parse_code_block(text):
    """
    Extracts code block from markdown string.
    """
    if "```python" in text:
        parts = text.split("```python")
        code = parts[1].split("```")[0]
        return code.strip()
    elif "```" in text:
        parts = text.split("```")
        code = parts[1].split("```")[0]
        return code.strip()
    return text.strip()


def check_syntax(code):
    try:
        ast.parse(code)
        return True, ""
    except Exception as e:
        return False, str(e)

ADVERSARIAL_PARADIGMS = {
    "Slime Mold Routing (Physarum polycephalum)": 
        "Model the GPU memory pathways as tubular networks that expand towards food sources (optimized indices) and contract along inefficient paths. Restructure index gather-scatter so that we dynamically prune/shrink redundant coordinate paths to minimize access latency.",
    "Quantum Wavefunction Superposition": 
        "Model index selections as superpositions of quantum states. Design a unitary-like transformation that rotates index representations before gathering to avoid memory alignment decoherence on Apple Silicon.",
    "Aperiodic Tiling (Penrose Tiling)": 
        "Structure the index grid using 5-fold Penrose aperiodic symmetry instead of standard rectilinear grids. This avoids periodic cache alignment conflicts in MPS memory blocks.",
    "Topological Persistence & Simplices": 
        "Represent the coordinate index space as a simplicial complex. Design the gather-scatter operation as a coboundary boundary transition on this complex, ensuring local topological invariants prevent memory fractures.",
    "Ultrametric Diffusion (p-adic diffusion)": 
        "Structure the gather-scatter index jumping distances according to the p-adic ultrametric (base p=5). High-frequency keys should leap across tree branches to ensure non-Archimedean locality in L2 cache.",
    "Floquet Dynamical States (Temporal Gauging)": 
        "Structure the memory accesses using periodic temporal Floquet operators. Memory offsets are dynamically shifted using time-periodic driving functions, preventing static resonance modes in the MPS hardware cache.",
    "Hyperbolic Poincaré Disk Projection": 
        "Map attention weights and memory index distances onto a 2D Hyperbolic Poincaré Disk. Geodesic distance routing on the hyperbolic plane replaces Euclidean distance, allowing exponential representation packing with zero metric drift.",
    "Symplectic Hamiltonian Flow": 
        "Model memory read/write cycles as trajectories conserving a symplectic phase space Hamiltonian. The index routing uses a leapfrog integrator that guarantees conservation of memory energy invariants.",
    "Fractional Laplacian Levy Flight": 
        "Model token attention distribution as non-local anomalous diffusion governed by a fractional Laplacian operator. This allows token representations to propagate across non-adjacent context boundaries via Levy flights.",
    "Knot Theory Braid Operations": 
        "Model sequence paths as braided strands in a topological knot. The gather-scatter mapping is computed as a braid group representation matrix, minimizing entanglement of thread locks and synchronization gates on Apple Silicon."
}

TARGETS_INFO = {
    "mps_scatter": {
        "file": "qan_transformers/kernels/mps_scatter.py",
        "description": "Custom Apple Silicon MPS gather-scatter kernel in `qan_transformers/kernels/mps_scatter.py`.",
        "requirements": "You MUST preserve the class `MPSCoordinateGatherScatter(torch.autograd.Function)` with its `forward` and `backward` methods, and the primary function `mps_coordinate_gather_scatter(q, k, v, indices)`."
    },
    "cohomology": {
        "file": "qan_transformers/firewall/cohomology.py",
        "description": "Čech Cohomology firewall logic in `qan_transformers/firewall/cohomology.py`.",
        "requirements": "You MUST preserve the class `CohomologyFirewall` with its `check_obstruction` method."
    },
    "e8_decoder": {
        "file": "qan_transformers/math/e8_projection.py",
        "description": "E8 projection and Conway-Sloane E8 lattice decoder in `qan_transformers/math/e8_projection.py`.",
        "requirements": "You MUST preserve the class `ConwaySloaneE8Decoder` with its `decode` method, and the generator functions `generate_dynamic_e8_coordinates` and `generate_e8_coordinates`."
    },
    "e8_swap": {
        "file": "qan_transformers/math/e8_swap.py",
        "description": "E8 Memory Swap Grid DB in `qan_transformers/math/e8_swap.py`.",
        "requirements": "You MUST preserve the class `AdelicMemorySwapGridDB` with its `swap_out`, `swap_in`, and `swap_in_batch` methods, and the class `CoWMemorySwapGridDB`."
    },
    "adelic": {
        "file": "qan_transformers/optim/adelic.py",
        "description": "Adelic Langevin SGLD optimizer in `qan_transformers/optim/adelic.py`.",
        "requirements": "You MUST preserve the class `AdelicLangevinOptimizer` with its `step` method."
    },
    "procrustes": {
        "file": "qan_transformers/math/procrustes.py",
        "description": "Orthogonal Procrustes alignment in `qan_transformers/math/procrustes.py`.",
        "requirements": "You MUST preserve the function `compute_procrustes_alignment(X_src, X_tgt)` and the function `validate_alignment(X_src_val, X_tgt_val, M_align)`."
    },
    "rag": {
        "file": "qan_transformers/math/rag.py",
        "description": "Lattice indexer and query system for RAG in `qan_transformers/math/rag.py`.",
        "requirements": "You MUST preserve the class `LatticeIndexer` with its `embed_chunk`, `chunk_text`, and `query` methods."
    },
    "attention": {
        "file": "qan_transformers/modeling/attention.py",
        "description": "QuasicrystallineAttention layer in `qan_transformers/modeling/attention.py`.",
        "requirements": "You MUST preserve the class `QuasicrystallineAttention` with its `forward` and `_morse_collapse_cache` methods."
    },
    "lora_pipeline": {
        "file": "qan_transformers/lora/pipeline.py",
        "description": "LoRA adapter injection and backtracking training pipeline in `qan_transformers/lora/pipeline.py`.",
        "requirements": "You MUST preserve the functions `inject_lora(model, r, lora_alpha)`, `mark_only_lora_as_trainable(model)`, and `train_loop(model, data, steps)`."
    },
    "rope_wrapping": {
        "file": "qan_transformers/modeling/auto.py",
        "description": "Modulo position ID rotary embedding wrapping in `qan_transformers/modeling/auto.py`.",
        "requirements": "You MUST preserve the function `wrap_rotary_embeddings(model)`."
    }
}

def get_coder_prompt(target, current_code, baseline_ms, history, paradigm_name, paradigm_desc):
    info = TARGETS_INFO[target]
    return f"""
You are the CoderAgent for Quasicrystalline Attention Networks (QAN).
Your task is to optimize the {info["description"]}.

Here is the current code of the module:
```python
{current_code}
```

The current baseline performance (mean latency over representative benchmark tasks) is: {baseline_ms:.4f} ms.

Here is the optimization history and feedback from previous attempts:
{history}

--- ADVERSARIAL CHALLENGE (Cross-Disciplinary Inspiration) ---
You MUST optimize the module by integrating principles from: {paradigm_name}.
Concept: {paradigm_desc}
Challenge: Adapt this scientific concept into clean, valid, high-performance PyTorch/NumPy code inside the module.

Instructions:
1. Speculate on a faster implementation of this module using PyTorch/NumPy.
2. Focus on avoiding redundant copies, optimizing index views, reducing allocations, and maintaining safety.
3. {info["requirements"]}
4. You MUST output the complete, runnable Python code for the entire file. Do NOT omit imports, classes, or function names. Do NOT use placeholders.
5. Output response as a single valid Python code block wrapped in ```python ... ```. Do not include extra text.
"""


def get_auditor_prompt(proposed_code):
    return f"""
You are the AuditorAgent for Quasicrystalline Attention Networks (QAN).
Your task is to review the proposed code mutation for `qan_transformers/kernels/mps_scatter.py` for syntax correctness, logical equivalence to the original behavior, and safety.

Here is the proposed code:
```python
{proposed_code}
```

Instructions:
1. Ensure the code is syntactically valid python.
2. Verify that it contains `mps_coordinate_gather_scatter(q, k, v, indices)` and the custom autograd function `MPSCoordinateGatherScatter`.
3. Check for any dangerous operations, arbitrary code execution, or deletion of core logic.
4. Output your response as a single valid JSON object containing exactly two keys:
   "approved": true or false,
   "critique": "A brief explanation of your decision"
   
Do NOT include any markdown formatting, comments, or explanations outside the JSON block.
"""

# ---------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous Recursive Self-Improvement Loop with Adversarial Speculation")
    parser.add_argument("--backend", type=str, default="mock", choices=["gemini", "local", "mock"],
                        help="LLM backend to use for speculations (gemini, local, mock)")
    parser.add_argument("--generations", type=int, default=3, help="Number of optimization iterations")
    parser.add_argument("--model", type=str, default="google/gemma-4-e2b-it", help="Local model name for HF")
    parser.add_argument("--target", type=str, default="mps_scatter",
                        choices=["mps_scatter", "cohomology", "e8_decoder", "e8_swap", "adelic", "procrustes", "rag", "attention", "lora_pipeline", "rope_wrapping"],
                        help="Target module to self-optimize")
    args = parser.parse_args()

    log_file_path = "scratch/self_improvement.log"
    os.makedirs("scratch", exist_ok=True)
    
    def log(message):
        print(message)
        with open(log_file_path, "a") as f:
            f.write(message + "\n")

    log("=========================================================")
    log(f"Starting QAN Self-Improvement Loop at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Backend: {args.backend} | Target: {args.target} | Generations: {args.generations}")
    log("=========================================================")

    # Initialize LLM backend
    if args.backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log("ERROR: No GEMINI_API_KEY or GOOGLE_API_KEY found in environment variables.")
            sys.exit(1)
        backend = GeminiAPIBackend(api_key)
    elif args.backend == "local":
        backend = LocalLLMBackend(args.model)
    else:
        backend = MockBackend()

    target_info = TARGETS_INFO[args.target]
    kernel_path = target_info["file"]
    backup_path = kernel_path + ".bak"

    # Read original code to calculate baseline connectivity and baseline undefined nodes
    with open(kernel_path, "r") as f:
        original_code = f.read()

    from scratch.run_cohomology_audit import CallGraphVisitor, analyze_cohomology_connectivity as audit_conn
    import ast
    try:
        orig_ast = ast.parse(original_code)
        orig_visitor = CallGraphVisitor()
        orig_visitor.collect_definitions_and_imports(orig_ast)
        orig_visitor.visit(orig_ast)
        baseline_undefined = orig_visitor.undefined_nodes
    except Exception:
        baseline_undefined = set()

    # Evaluate initial baseline performance
    log(f"Running initial benchmark for target '{args.target}' to establish baseline...")
    try:
        baseline_ms = run_precision_benchmark(args.target, kernel_path)
        log(f"Baseline mean latency: {baseline_ms:.4f} ms")
    except Exception as e:
        log(f"ERROR running initial benchmark: {e}")
        sys.exit(1)

    history_logs = ""
    current_best_latency = baseline_ms
    generation = 1

    while generation <= args.generations:
        log(f"\n--- GENERATION {generation} ---")
        
        # Read current code
        with open(kernel_path, "r") as f:
            current_code = f.read()

        # Step 1: Adversarial Speculator selects or generates a cross-disciplinary paradigm
        paradigm_name = None
        paradigm_desc = None
        if args.backend in ["gemini", "local"]:
            try:
                backend_name = "Gemini API" if args.backend == "gemini" else "local model"
                log(f"[ADVERSARIAL SPECULATOR] Querying {backend_name} for a new optimization paradigm...")
                speculator_prompt = (
                    "You are the Adversarial Speculator for the QAN-ATLAS self-improvement loop.\n"
                    "Your task is to propose a new, creative, mathematically-grounded optimization paradigm "
                    "to speed up the target kernel/module code. Propose a paradigm that merges code optimizations "
                    "with advanced concepts from physics, topology, non-Euclidean geometry, or discrete mathematics "
                    "so we don't just explore the same concepts repeatedly.\n\n"
                    f"Target module: {args.target}\n"
                    f"Current code:\n```python\n{current_code}\n```\n\n"
                    "Your response must be in JSON format with two keys 'paradigm' and 'description'. Do not output any other text or markdown formatting outside the JSON block.\n"
                    "JSON format:\n"
                    "{\n"
                    "  \"paradigm\": \"Name of the paradigm\",\n"
                    "  \"description\": \"Explanation of how to apply this paradigm to optimize the target code.\"\n"
                    "}"
                )
                response_text = backend.generate(speculator_prompt)
                import re
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group(0))
                    paradigm_name = parsed_json.get("paradigm")
                    paradigm_desc = parsed_json.get("description")
            except Exception as e:
                log(f"[ADVERSARIAL SPECULATOR] Dynamic generation failed: {e}. Falling back to static list.")

        if not paradigm_name or not paradigm_desc:
            paradigm_name, paradigm_desc = random.choice(list(ADVERSARIAL_PARADIGMS.items()))

        log(f"[ADVERSARIAL SPECULATOR] Injected Paradigm: {paradigm_name}")
        log(f"Description: {paradigm_desc}")

        valid_candidates = []
        
        # We generate 4 candidate paths per generation (Langevin / Swarm approach)
        log("CoderAgent: Generating 4 candidate mutations...")
        for c_idx in range(4):
            coder_prompt = get_coder_prompt(args.target, current_code, current_best_latency, history_logs, paradigm_name, paradigm_desc)
            raw_coder_response = backend.generate(coder_prompt)
            proposed_code = parse_code_block(raw_coder_response)

            # Semantic structure check
            # We want to make sure it imports torch or numpy, has no placeholders, and has the correct key terms
            if "# ..." in proposed_code or "TODO" in proposed_code:
                continue

            # Syntax check
            is_syntax_ok, syntax_err = check_syntax(proposed_code)
            if not is_syntax_ok:
                continue

            # Step 2: Čech Cohomology Call-Graph Firewall
            from scratch.run_cohomology_audit import analyze_cohomology_connectivity, CallGraphVisitor
            import ast
            is_conn, l2, conn_msg = analyze_cohomology_connectivity(proposed_code, tau=0.05)
            
            # Check for new undefined nodes
            try:
                prop_ast = ast.parse(proposed_code)
                prop_visitor = CallGraphVisitor()
                prop_visitor.collect_definitions_and_imports(prop_ast)
                prop_visitor.visit(prop_ast)
                prop_undefined = prop_visitor.undefined_nodes
            except Exception:
                prop_undefined = set()
                
            new_undefined = prop_undefined - baseline_undefined
            if len(new_undefined) > 0:
                log(f"  -> Candidate {c_idx} REJECTED by Čech Cohomology Firewall (calls new undefined symbols: {new_undefined})!")
                continue
                
            if not is_conn:
                # If it's not connected, check if the baseline code was connected
                _, orig_l2, _ = audit_conn(original_code, tau=0.05)
                if orig_l2 >= 0.05:
                    log(f"  -> Candidate {c_idx} REJECTED by Čech Cohomology Firewall (algebraic connectivity: {l2:.4f} < 0.05)!")
                    continue
            
            # Auditor check
            auditor_prompt = get_auditor_prompt(proposed_code)
            raw_auditor_response = backend.generate(auditor_prompt)
            
            is_approved = False
            try:
                clean_resp = raw_auditor_response.strip()
                if "{" in clean_resp and "}" in clean_resp:
                    clean_resp = clean_resp[clean_resp.find("{"):clean_resp.rfind("}")+1]
                data = json.loads(clean_resp)
                is_approved = data.get("approved", False)
            except:
                is_approved = "approved" in raw_auditor_response.lower()

            if is_approved:
                valid_candidates.append(proposed_code)
                log(f"  -> Candidate {c_idx} PASSED firewall and audit.")
            else:
                log(f"  -> Candidate {c_idx} REJECTED by Auditor.")

        if not valid_candidates:
            log("No valid candidates survived this generation.")
            generation += 1
            continue

        # Step 3: Discrete Morse Candidate Collapse (if we have > 2 candidates)
        selected_candidates = valid_candidates
        if len(valid_candidates) > 2:
            log(f"DiscreteMorseContractor: Collapsing {len(valid_candidates)} candidates to 2 critical summits...")
            from scratch.run_morse_contractor import DiscreteMorseContractor
            contractor = DiscreteMorseContractor(target_count=2)
            reps, comps = contractor.contract(valid_candidates)
            selected_candidates = [valid_candidates[r] for r in reps]
            log(f"Selected candidates at indices {reps} representing the critical summits.")

        # Step 4: Profile & Verify selected candidates
        for idx, candidate in enumerate(selected_candidates):
            log(f"Profiling selected candidate {idx + 1}/{len(selected_candidates)}...")
            shutil.copyfile(kernel_path, backup_path)
            
            try:
                with open(kernel_path, "w") as f:
                    f.write(candidate)

                tests_passed, test_output = run_unit_tests()
                if not tests_passed:
                    log("  -> pytest FAILED! Restoring backup...")
                    shutil.copyfile(backup_path, kernel_path)
                    continue
                
                log("  -> pytest PASSED! Running micro-benchmark...")
                mutated_latency = run_precision_benchmark(args.target, kernel_path)
                log(f"  -> Latency: {mutated_latency:.4f} ms")

                speedup = ((current_best_latency - mutated_latency) / current_best_latency) * 100.0
                if mutated_latency < current_best_latency:
                    log(f"  -> SUCCESS: Latency reduced from {current_best_latency:.4f} ms to {mutated_latency:.4f} ms ({speedup:.2f}% speedup)!")
                    current_best_latency = mutated_latency
                    history_logs += f"\nGen {generation} ({paradigm_name}): Speedup of {speedup:.2f}%."
                    
                    # Commit to git
                    subprocess.run(["git", "add", kernel_path])
                    subprocess.run(["git", "commit", "-m", f"Auto-optimized {args.target} kernel ({paradigm_name}): latency reduced to {mutated_latency:.4f} ms"])
                    # Break out since we got a success on this generation
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    break
                else:
                    log(f"  -> REJECTED: Not faster than baseline {current_best_latency:.4f} ms. Restoring...")
                    shutil.copyfile(backup_path, kernel_path)
            except Exception as e:
                log(f"  -> ERROR during profiling: {e}. Restoring...")
                shutil.copyfile(backup_path, kernel_path)
            finally:
                if os.path.exists(backup_path):
                    os.remove(backup_path)

        generation += 1

    # End of run
    log("\n=========================================================")
    log(f"Self-Improvement Loop complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Final best latency: {current_best_latency:.4f} ms (Baseline was: {baseline_ms:.4f} ms)")
    log("=========================================================")

if __name__ == "__main__":
    main()
