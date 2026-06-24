import os
import torch
import numpy as np
import sys

# Ensure workspace imports work
sys.path.insert(0, os.getcwd())

print("================================================================================")
print("             PROJECT ATLAS (QAN-ATLAS) - FIRST RUN EXPERIENCE                   ")
print("================================================================================")

# Step 1: Initialize the Ultrametric Cognitive Engine (UCE)
print("\n[Step 1] Loading the distilled 139 KiB expert and its p-adic tree coordinates...")
from ultrametric_ce.inference import load_model_and_tree
from ultrametric_ce.gemma_interface import load_gemma_tokenizer

# Load the public expert model
model_path = "data/uce_e4b_distilled.safetensors"
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}. Please check your installation.")
    sys.exit(1)

tree, model = load_model_and_tree(model_path)
print("✓ Distilled expert and tree loaded successfully!")

# Step 2: Map natural language query to tree coordinates
print("\n[Step 2] Mapping a natural language prompt into semantic coordinates...")
tokenizer = load_gemma_tokenizer("google/gemma-4-E2B-it")
prompt = "How does concentric E8 lattice routing optimize LLM memory?"
tokens = tokenizer.encode(prompt)
print(f"Input Prompt: '{prompt}'")
print(f"Tokenized sequence (length {len(tokens)}): {tokens[:10]}...")

# Extract coordinates using UCE
from ultrametric_ce.distillation import text_to_address_sequence
coords = text_to_address_sequence(prompt, tokenizer, tree)
if not coords:
    demo_prompt = "denoted efficacious dependences"
    demo_tokens = tokenizer.encode(demo_prompt)
    coords = text_to_address_sequence(demo_prompt, tokenizer, tree)
    print(f"\nMapped E8 coordinate addresses (demonstration fallback: '{demo_prompt}'):")
    for i, coord in enumerate(coords[:3]):
        token_str = tokenizer.decode([demo_tokens[i]])
        print(f"  Token '{token_str}' -> p-adic Address: {coord}")
else:
    print(f"\nMapped E8 coordinate addresses (first 3 tokens):")
    for i, coord in enumerate(coords[:3]):
        token_str = tokenizer.decode([tokens[i]])
        print(f"  Token '{token_str}' -> p-adic Address: {coord}")


# Step 3: Run Quasicrystalline Attention with Cohomology Firewall
print("\n[Step 3] Initializing the Quasicrystalline Attention Layer & Firewall...")
from qan_transformers.modeling.attention import QuasicrystallineAttention
from qan_transformers.firewall.cohomology import CohomologyFirewall

# Initialize a small attention layer
embed_dim = 64
num_heads = 4
seq_len = 32
batch_size = 1

attn = QuasicrystallineAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_key_value_heads=2,
    sparse_ratio=0.15,
)
attn.eval() # Transition to eval mode (enables Cohomology checks)

# Generate mock activation tensors
x = torch.randn(batch_size, seq_len, embed_dim)
print(f"Input Tensor Shape: {x.shape}")

# Run forward pass (routes through E8 lattice projection and alignment)
# Pass an empty dictionary to receive the populated kv_cache
output, kv_cache = attn(x, kv_cache={})
print(f"Attention Output Shape: {output.shape}")
print("✓ Fused E8 projection and alignment executed successfully!")

# Step 4: Run the Cohomology Firewall structural check
print("\n[Step 4] Checking attention graph connectivity via Čech Cohomology...")
firewall = CohomologyFirewall(threshold=1.5, tau=0.05)
# Create a dummy attention matrix
attention_matrix = torch.matmul(x, x.transpose(-1, -2))
attention_matrix = torch.softmax(attention_matrix, dim=-1)[0] # single batch

is_fractured, cfi, alt_idx = firewall.check_obstruction(attention_matrix)
print(f"Firewall Status: {'BLOCKED (Fracture Detected)' if is_fractured else 'PASSED (Stable)'}")
print(f"Cohomology Fracture Index (CFI): {cfi:.4f}")
print(f"Alternative Route Indices (first 5): {alt_idx[:5] if isinstance(alt_idx, list) else alt_idx}")

print("\n================================================================================")
print("             FIRST RUN EXPERIENCE PASSED SUCCESSFULLY!                          ")
print("================================================================================")
