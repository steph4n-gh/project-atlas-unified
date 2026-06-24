import os

# Redirect Hugging Face cache directories to the external storage volume
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")

import sys
import time
import argparse
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup PYTHONPATH
sys.path.insert(0, os.getcwd())

import qan_transformers.mlx.modeling
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.generate import stream_generate
from qan_transformers.mlx.modeling import load_and_graft_elq_model, graft_mlx_model, patch_speculative_decoding, layer_by_layer_prefill

class LockedKVCache:
    def __init__(self, keys: Optional[mx.array], values: Optional[mx.array], offset: int, max_size: Optional[int] = None, keep: Optional[int] = None):
        self.keys = keys
        self.values = values
        self.offset = offset
        self.max_size = max_size
        self.keep = keep

from qan_transformers.math.context_builder import crawl_codebase

def format_xml_context(files_dict: Dict[str, str]) -> str:
    """
    Formats the file dictionary into a structured XML prompt context.
    """
    parts = []
    for rel_path, content in files_dict.items():
        parts.append(f'<file path="{rel_path}">\n{content}\n</file>\n')
    return "\n".join(parts)

def format_chat(messages: List[Dict[str, str]], tokenizer: Any) -> str:
    """
    Formulates a chat template, falling back to manual ChatML if apply_chat_template fails.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text

def lock_caches(model_cache: List[Any]) -> List[LockedKVCache]:
    """
    Clones and locks standard model key-value caches.
    """
    locked = []
    for c in model_cache:
        keys = mx.array(c.keys) if c.keys is not None else None
        values = mx.array(c.values) if c.values is not None else None
        offset = c.offset
        max_size = getattr(c, "max_size", None)
        keep = getattr(c, "keep", None)
        locked.append(LockedKVCache(keys, values, offset, max_size, keep))
    return locked

def restore_caches(model_cache: List[Any], locked: List[LockedKVCache]):
    """
    Restores standard caches to their locked states.
    """
    for c, lc in zip(model_cache, locked):
        c.keys = mx.array(lc.keys) if lc.keys is not None else None
        c.values = mx.array(lc.values) if lc.values is not None else None
        c.offset = lc.offset

def lock_custom_caches(model: nn.Module) -> Dict[int, Dict[str, Any]]:
    """
    Clones and locks custom grafted QuasicrystallineAttention caches.
    """
    locked = {}
    for idx, layer in enumerate(model.layers):
        if hasattr(layer.self_attn, "custom_kv_cache") and layer.self_attn.custom_kv_cache is not None:
            c = layer.self_attn.custom_kv_cache
            locked[idx] = {
                "K": mx.array(c["K"]) if c["K"] is not None else None,
                "V": mx.array(c["V"]) if c["V"] is not None else None,
                "indices": mx.array(c["indices"]) if c["indices"] is not None else None,
                "alignment_scores": mx.array(c["alignment_scores"]) if c["alignment_scores"] is not None else None,
                "seq_len": c["seq_len"]
            }
    return locked

def restore_custom_caches(model: nn.Module, locked: Dict[int, Dict[str, Any]]):
    """
    Restores custom grafted caches to their locked states.
    """
    for idx, layer in enumerate(model.layers):
        if idx in locked:
            lc = locked[idx]
            layer.self_attn.custom_kv_cache = {
                "K": mx.array(lc["K"]) if lc["K"] is not None else None,
                "V": mx.array(lc["V"]) if lc["V"] is not None else None,
                "indices": mx.array(lc["indices"]) if lc["indices"] is not None else None,
                "alignment_scores": mx.array(lc["alignment_scores"]) if lc["alignment_scores"] is not None else None,
                "seq_len": lc["seq_len"]
            }

def main():
    parser = argparse.ArgumentParser(description="QAN-ATLAS: Whole-Codebase QA Chat CLI (MLX)")
    parser.add_argument("--folder", type=str, required=True, help="Folder path of the codebase to ingest")
    parser.add_argument("--model", type=str, default="google/gemma-4-e2b-it", help="Pretrained target model identifier/path")
    parser.add_argument("--draft", type=str, default=None, help="Pretrained draft assistant model identifier/path")
    parser.add_argument("--elq-path", type=str, default=None, help="Path to ELQ quantized weights for target model")
    parser.add_argument("--graft", action="store_true", help="Graft target model with Quasicrystalline Attention (unquantized)")
    parser.add_argument("--draft-graft", action="store_true", help="Graft draft model with Quasicrystalline Attention (unquantized)")
    parser.add_argument("--turbo", action="store_true", help="Use CPU Parallel Reaction Kinetics tokenizer")
    parser.add_argument("--mlx-turbo", action="store_true", help="Use GPU BlockBPE Kinetics tokenizer")
    parser.add_argument("--sparse-ratio", type=float, default=0.15, help="Sparsity ratio for Quasicrystalline Attention")
    parser.add_argument("--no-swap", action="store_true", help="Disable Adelic Memory Swap Grid DB to save memory")
    parser.add_argument("--num-layers", type=int, default=None, help="Slice model layers to run a shorter test")
    args = parser.parse_args()
    
    target_path = Path(args.model)
    print(f"Loading tokenizer and target model from: {target_path}...", flush=True)
    target_model, target_weights = load_model(target_path, lazy=True, strict=False)
    tokenizer = load_tokenizer(target_path)
    
    if args.turbo or args.mlx_turbo:
        from qan_transformers.tokenizer import KineticsTokenizer
        print(f"[Tokenizer] Activating KineticsTokenizer (forced use_rust={args.turbo}, use_mlx={args.mlx_turbo})...", flush=True)
        tokenizer = KineticsTokenizer(args.model, use_rust=args.turbo, use_mlx=args.mlx_turbo, base_tokenizer=tokenizer)
    else:
        from qan_transformers.tokenizer import load_qan_tokenizer
        print(f"[Tokenizer] Activating KineticsTokenizer in Autopilot mode...", flush=True)
        tokenizer = load_qan_tokenizer(args.model, base_tokenizer=tokenizer)
    
    # Graft target model if requested
    if args.elq_path:
        print(f"Grafting target model with ELQ quantized weights from: {args.elq_path} (sparse_ratio={args.sparse_ratio})...", flush=True)
        target_model = load_and_graft_elq_model(target_model, args.elq_path, sparse_ratio=args.sparse_ratio)
    elif args.graft:
        print(f"Grafting target model with Quasicrystalline Attention (unquantized) (sparse_ratio={args.sparse_ratio})...", flush=True)
        target_model = graft_mlx_model(target_model, sparse_ratio=args.sparse_ratio)
        
    # Disable Swap DB if requested
    if args.no_swap:
        print("[ELQ] Disabling Adelic Memory Swap Grid DB on target model layers.", flush=True)
        for m in target_model.modules():
            if m.__class__.__name__ == "QuasicrystallineAttention" and hasattr(m, "swap_db") and m.swap_db is not None:
                m.swap_db.enabled = False
        
    target_model.tokenizer = tokenizer
        
    # Load draft model if requested
    draft_model = None
    if args.draft:
        draft_path = Path(args.draft)
        print(f"Loading draft assistant model from: {draft_path}...", flush=True)
        draft_model, draft_weights = load_model(draft_path, lazy=True, strict=False)
        del draft_weights
        
        if args.draft_graft:
            print(f"Grafting draft model with Quasicrystalline Attention (unquantized) (sparse_ratio={args.sparse_ratio})...", flush=True)
            draft_model = graft_mlx_model(draft_model, sparse_ratio=args.sparse_ratio)
            
        if args.no_swap:
            print("[ELQ] Disabling Adelic Memory Swap Grid DB on draft model layers.", flush=True)
            for m in draft_model.modules():
                if m.__class__.__name__ == "QuasicrystallineAttention" and hasattr(m, "swap_db") and m.swap_db is not None:
                    m.swap_db.enabled = False
            
        draft_model.tokenizer = tokenizer
            
        # Patch speculative decoding in mlx_lm
        gen_mod = sys.modules.get("mlx_lm.generate")
        patch_speculative_decoding(gen_mod)
        
    # Free loaded weights dictionaries to preserve RAM
    del target_weights
    import gc
    gc.collect()
    mx.clear_cache()
    # Slice model layers if requested to run a shorter test
    if args.num_layers is not None:
        target_text_model = target_model.language_model.model if hasattr(target_model, "language_model") else (target_model.model if hasattr(target_model, "model") else target_model)
        target_text_model.layers = target_text_model.layers[:args.num_layers]
        print(f"[ELQ] Sliced target model layers to first {args.num_layers} layers.", flush=True)
        if draft_model is not None:
            draft_text_model = draft_model.language_model.model if hasattr(draft_model, "language_model") else (draft_model.model if hasattr(draft_model, "model") else draft_model)
            draft_text_model.layers = draft_text_model.layers[:args.num_layers]
            print(f"[ELQ] Sliced draft model layers to first {args.num_layers} layers.", flush=True)
            
    # 1. Ingest/Index folder
    print(f"[Ingestion] Crawling codebase recursively: {args.folder}")
    codebase_files = crawl_codebase(args.folder)
    codebase_corpus = format_xml_context(codebase_files)
    
    num_files = len(codebase_files)
    print(f"[Ingestion Completed] Ingested {num_files} files ({len(codebase_corpus):,} characters of structured XML context).")
    
    if not codebase_corpus.strip():
        print("Error: No supported text or code files found in the specified folder.")
        sys.exit(1)
        
    # 2. Tokenize the entire codebase
    print("[Tokenization] Tokenizing codebase corpus...")
    codebase_token_ids = tokenizer.encode(codebase_corpus, add_special_tokens=False)
    codebase_token_ids = mx.array([codebase_token_ids], dtype=mx.int32)
    total_tokens = codebase_token_ids.shape[1]
    print(f"[Tokenization Completed] Codebase tokenized into {total_tokens:,} tokens.")
    
    # 3. Chunked prefill
    chunk_size = 16384
    start_prefill = time.time()
    print(f"[Prefill] Ingesting codebase layer-by-layer (size={chunk_size}) on target model...", flush=True)
    
    target_cache = target_model.make_cache()
    logits, _ = layer_by_layer_prefill(target_model, codebase_token_ids, target_cache, chunk_size=chunk_size)
        
    # Prefill codebase on draft cache if draft model is present and not a unified assistant
    draft_cache = None
    if draft_model:
        draft_cache = draft_model.make_cache()
        is_assistant = (
            hasattr(draft_model, "post_projection") or
            "Assistant" in draft_model.__class__.__name__ or
            getattr(draft_model, "model_type", None) in ["gemma4_assistant", "gemma4_unified_assistant"]
        )
        if not is_assistant:
            print(f"[Prefill] Ingesting codebase layer-by-layer (size={chunk_size}) on draft model...", flush=True)
            layer_by_layer_prefill(draft_model, codebase_token_ids, draft_cache, chunk_size=chunk_size)
                
    print(f"[Prefill Completed] Ingested codebase in {time.time() - start_prefill:.2f} seconds.", flush=True)
    
    # 4. Lock KV-cache state
    print("[Locking] Freezing KV-cache state as read-only codebase memory...")
    locked_target_cache = lock_caches(target_cache)
    locked_draft_cache = lock_caches(draft_cache) if draft_cache else None
    
    if hasattr(target_model, "language_model"):
        target_text_model = target_model.language_model.model
    else:
        target_text_model = target_model.model
        
    draft_text_model = None
    if draft_model:
        if hasattr(draft_model, "language_model"):
            draft_text_model = draft_model.language_model.model
        else:
            draft_text_model = draft_model.model
            
    locked_target_custom = lock_custom_caches(target_text_model)
    locked_draft_custom = lock_custom_caches(draft_text_model) if draft_text_model else {}
    
    # Setup conversation
    messages = [
        {"role": "user", "content": "Please register the entire codebase files I have preloaded into your memory. I will now ask you questions about the files, classes, modules, and code layout of this project."},
        {"role": "assistant", "content": "Understood. The codebase context has been locked into my Quasicrystalline memory cache. I am ready to explain code structures, locate functions, and answer your development questions!"}
    ]
    
    print("\n" + "="*70)
    print("      QAN-ATLAS CODESPACE CHAT ACTIVE: DIRECT CODEBASE INTERACTION (MLX)")
    print("="*70)
    print("Ask questions about code structures, locate modules, or summarize code files!")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("\033[1;33mYou: \033[0m")
        if user_input.strip().lower() == "exit":
            print("Exiting codebase chat session.")
            break
            
        if not user_input.strip():
            continue
            
        # Restore KV caches to locked codebase state
        restore_caches(target_cache, locked_target_cache)
        if draft_cache and locked_draft_cache:
            restore_caches(draft_cache, locked_draft_cache)
            
        restore_custom_caches(target_text_model, locked_target_custom)
        if draft_text_model:
            restore_custom_caches(draft_text_model, locked_draft_custom)
            
        messages.append({"role": "user", "content": user_input})
        prompt_text = format_chat(messages, tokenizer)
        
        locked_len = locked_target_cache[0].offset
        print(f"  [Context Info] Total sequence context size: {locked_len:,} tokens")
        print("  [Processing] Searching locked memory context and generating response...")
        
        start_gen = time.time()
        print("\033[1;32mAlpha-Atlas: \033[0m", end="", flush=True)
        
        # Setup combined prompt cache
        if draft_cache:
            combined_cache = target_cache + draft_cache
        else:
            combined_cache = target_cache
            
        gen_kwargs = {
            "max_tokens": 256,
            "prompt_cache": combined_cache,
            "fused_tokenization": True
        }
        if draft_model:
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["num_draft_tokens"] = 2
            
        response_text = ""
        for response in stream_generate(target_model, tokenizer, prompt_text, **gen_kwargs):
            print(response.text, end="", flush=True)
            response_text += response.text
            
        print(f"\n  [Latency] Generated in {time.time() - start_gen:.2f} seconds.")
        print("-" * 70 + "\n")
        
        messages.append({"role": "assistant", "content": response_text.strip()})

if __name__ == "__main__":
    main()
