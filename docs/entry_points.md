# Multi-Tier Entry Points: From Micro-Projections to Giga-Lattices

Project Atlas is not a monolithic black box. The unified codebase is designed as a highly modular, multi-tier pipeline. Developers can interface with the platform at three distinct entry points depending on hardware constraints, latency targets, and reasoning depth.

---

## 🧬 Entry Point 1: The Standalone Micro-Expert (UCE Student)

### What it is
A single, ultra-lightweight **UCE Student Model** checkpoint (`.safetensors` + `.meta.json`), typically clocking in at just **~139 KiB** of weights.

### When to use
When you need to run high-speed, local, zero-dependency semantic routing on low-power devices (such as edge processors or old laptops). Each expert is specialized in a specific domain tree (e.g., Python code parsing, SQL queries, or regex dialect mapping).

### Python Example
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Volumes/Storage/project_atlas_unified")))

from ultrametric_ce.inference import load_model_and_tree
from ultrametric_ce.distillation import text_to_address_sequence

# Load the single 139 KiB expert and its p-adic tree configuration
tree, model = load_model_and_tree("data/uce_e4b_distilled.safetensors")

# Map natural language prompt into the expert's tree coordinates
# (Uses ONLY the tokenizer; avoids loading heavy model weights)
from ultrametric_ce.gemma_interface import load_gemma_tokenizer
tokenizer = load_gemma_tokenizer("google/gemma-4-E2B-it")
prompt_addrs = text_to_address_sequence("Match email address", tokenizer, tree)

# Run direct UCE projection forward pass
probs = model.forward(prompt_addrs)
print("Projected leaf coordinates active:", probs)
```

---

## 🎛️ Entry Point 2: The Mixture of Experts (MoE Routing Gateway)

### What it is
The **`UCEMoeRouter` Gateway** orchestrating 8 developer experts under a unified subspace router. It uses memory-mapped file paging (`WeightManager`) and transactional file locks (`fcntl.flock`) to swap expert parameters on-demand under a **$15\text{ms}$ VRAM-swapping ceiling**.

### When to use
When building multi-domain developer portals, local web engines, or IDE autocomplete plugins that need to seamlessly route and resolve complex programming prompts across different contexts (Python, Rust, SQL, DevOps, and Regex) in parallel.

### Python Example
```python
from ultrametric_ce.moe import UCEMoeRouter

# Initialize the 8-Expert MoE Router pointing to the expert weights directory
router = UCEMoeRouter(
    moe_dir="/Volumes/Storage/project_atlas_unified/tmp/moe",
    gemma_model_id="google/gemma-4-E2B-it"
)

# 1. Gateway routes the prompt to the correct expert subspace
routed_expert = router.route_prompt("docker compose up -d")
print(f"Subspace routing resolved to expert: {routed_expert}") # -> "devops_infra"

# 2. Page expert weights and execute active-path generation
response = router.generate("docker compose up -d", max_new_tokens=30)
print(response)
```

---

## 🌀 Entry Point 3: The Full Foundation Model (QAN Transformer)

### What it is
The full foundation model (such as Gemma 12B or E2B) executing concentric **$E_8$ sparse attention**, **Discrete Morse KV Cache Retraction**, and **Čech Cohomology Firewall** checks. 

### When to use
When you need full, high-fidelity reasoning over massive context windows (**$200\text{k}+$ tokens**). The QAN attention layers collapse the KV cache memory footprint by $\ge 85\%$ and achieve $97.29\%$ compute sparsity, allowing you to load large contexts on standard Apple Silicon GPU setups.

### Python Example
```python
import torch
from qan_transformers.modeling.attention import QuasicrystallineAttention
from qan_transformers.firewall.cohomology import CohomologyFirewall

# 1. Instantiate the cohomology firewall
firewall = CohomologyFirewall(threshold=0.35)

# 2. Instantiate the coordinate-sparse attention layer with the firewall integrated
attn_layer = QuasicrystallineAttention(
    embed_dim=2048,
    num_heads=16,
    sparse_ratio=0.15,  # Keep 15% of KV coordinates, collapsing cache by 85%
    firewall=firewall
)

# 3. Initialize KV cache
kv_cache = {"keys": None, "values": None}

# 4. Transition module to evaluation mode
# Calling .eval() transitions the PyTorch module to evaluation mode, which is
# required for the Cohomology Firewall inline check to run (as it is bypassed
# during training mode).
attn_layer.eval()

# 5. Run forward pass with a standard sequence length
# Developer Note: While QAN supports sequence lengths of 200k+, trying to run a
# 200k context sequence in a single forward pass without chunking will materialize
# a massive 6.29-billion-element causal mask tensor (consuming 12.5GB to 25GB of
# VRAM/RAM), which can crash or freeze consumer devices. Developers should use
# a chunked prefill pipeline for ultra-long contexts.
hidden_states = torch.randn(1, 512, 2048)
attn_output, kv_cache = attn_layer(hidden_states, kv_cache=kv_cache)

# 6. Check firewall results stored inline in the KV Cache dictionary
if kv_cache.get("is_fractured", False):
    cfi = kv_cache.get("cfi", 0.0)
    print(f"Firewall Alert! Attention fracture detected (Score: {cfi}). Triggering rollback.")
```

---

## ⚙️ Quantization & Memory Footprint Entry Points (With or Without ELQ)

Project Atlas provides flexible adoption entry points for optimizing model weights and KV caches:

### A. Adopting the Quasicrystalline Attention Layer (E8 vs. Leech $\Lambda_{24}$)
*   Replace standard self-attention with `QuasicrystallineAttention(lattice='e8')` or `QuasicrystallineAttention(lattice='leech')`.
*   **E8 Lattice (Default):**
    *   **Addresses:** 240 coordinates in Shell 1.
    *   **Startup:** $< 1\text{ ms}$ (instantaneous generation).
    *   **Fit:** Ideal for low-power edge/mobile devices where local swapping latency must be minimized.
    *   **Performance:** Yields a **+69.27%** speculative decoding speedup (`34.92 tok/s`) on the 4B (E4B) model scale, and **+72.71%** speculative speedup (`9.43 tok/s`) on the 12B model scale.
*   **Leech Lattice ($\Lambda_{24}$):**
    *   **Addresses:** 196,560 coordinates in Shell 1 (819× address capacity).
    *   **Startup:** ~890 ms coordinate-mapping compilation overhead at session boot.
    *   **Fit:** Ideal for long-context foundation model inference on developer hardware.
    *   **Performance:** Tighter geometric filtering cuts speculative rollbacks, boosting generation throughput to **37.87 tok/s** (**+77.54%** speculative speedup) on the 4B (E4B) model scale, and **9.46 tok/s** (**+73.26%** speculative speedup) on the 12B model scale.
*   **Result:** Compresses the KV cache footprint by **$\ge 85\%$** at long sequences ($200\text{k}+$ context) without altering model weights. Runs in standard fp16/bf16 formats.

### B. Standard 4-bit Quantization (Without ELQ)
*   Load public 4-bit model weights (such as `mlx-community/gemma-4-E4B-it-4bit`) directly onto local devices.
*   **Result**: Fits the entire 4B parameter model weights into **~2.5 GB** of unified RAM/VRAM on standard Macbooks, bypassing host RAM limitations.

### C. Embedding Lattice Quantization (With ELQ)
*   Convert the entire model structure into our custom `.elq` quantized binary format (e.g., `gemma-4-e4b-it.elq`).
*   **Result**: Converts standard dense linear layers into hardware-optimized sliding ELQ linear gates. The system dynamically pages and evaluates only the active weights in the Metal/ANE pipeline, maintaining maximum fidelity while keeping local memory footprint at a absolute minimum.

### D. Fused Prefill Acceleration (With Fused MPS Gather-Scatter)
*   Deploy with our specialized coordinate-sparse fused MPS kernel (`mps_coordinate_gather_scatter`) loaded on the PyTorch runtime.
*   **Result**: Accelerates coordinate gather/scatter during the prefill phase on Apple Silicon, slashing routing mean latency by **~50%** (reducing mean latency from **883.3 ms to 445.0 ms**).


