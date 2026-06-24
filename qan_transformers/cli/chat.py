import os
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from qan_transformers.math.rag import LatticeIndexer, crawl_codebase
from qan_transformers.modeling.auto import wrap_rotary_embeddings

def prepare_chat_input(tokenizer, messages, device) -> torch.Tensor:
    """
    Formats conversation robustly using HF chat template or custom fallback,
    and returns a 2D tensor on the target device. Handles strings, dicts,
    BatchEncoding, lists, and raw tensors gracefully.
    """
    try:
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        parts = [f"User: {msg['content']}\n" if msg["role"] == "user" else f"Assistant: {msg['content']}\n" for msg in messages]
        parts.append("Assistant: ")
        prompt_str = "".join(parts)
        token_ids = tokenizer.encode(prompt_str, return_tensors="pt")
        
    # Extract tensor robustly from tokenizer output
    if isinstance(token_ids, torch.Tensor):
        t = token_ids
    elif isinstance(token_ids, str):
        t = tokenizer.encode(token_ids, return_tensors="pt")
    elif hasattr(token_ids, "data") and isinstance(token_ids.data, dict) and "input_ids" in token_ids.data:
        t = token_ids.data["input_ids"]
    elif isinstance(token_ids, list):
        t = torch.tensor([token_ids])
    else:
        # Final fallback
        t = torch.tensor(token_ids)
        if t.ndim == 1:
            t = t.unsqueeze(0)
            
    return t.to(device)


def do_chat(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print(f"Targeting device: {device} | precision: {dtype}")
    
    # 1. Ingest/Index folder
    indexer = None
    codebase_token_ids = None
    total_tokens = 0
    codebase_corpus = ""
    
    if args.rag:
        print(f"[RAG Ingestion] Initializing LatticeIndexer and indexing folder: {args.folder}")
        indexer = LatticeIndexer(d_model=64)
        indexer.index_directory(args.folder)
        num_indexed = len(indexer.chunks)
        print(f"[RAG Ingestion Completed] Successfully indexed {num_indexed} chunks into E8 database.")
        if num_indexed == 0:
            print("Error: No supported text or code files found in the specified folder.")
            sys.exit(1)
    else:
        print(f"[Ingestion] Crawling codebase recursively: {args.folder}")
        codebase_corpus = crawl_codebase(args.folder)
        print(f"[Ingestion Completed] Ingested {len(codebase_corpus):,} characters of codebase context.")
        
        if not codebase_corpus.strip():
            print("Error: No supported text or code files found in the specified folder.")
            sys.exit(1)
            
    # 2. Initialize Tokenizer & Model
    print(f"[Model Loading] Loading tokenizer and causal LM model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if getattr(args, "turbo", False) or getattr(args, "mlx_turbo", False):
        from qan_transformers.tokenizer import KineticsTokenizer
        print(f"[Tokenizer] Activating KineticsTokenizer (forced use_rust={getattr(args, 'turbo', False)}, use_mlx={getattr(args, 'mlx_turbo', False)})...", flush=True)
        tokenizer = KineticsTokenizer(args.model, use_rust=getattr(args, 'turbo', False), use_mlx=getattr(args, 'mlx_turbo', False), base_tokenizer=tokenizer)
    else:
        from qan_transformers.tokenizer import load_qan_tokenizer
        print(f"[Tokenizer] Activating KineticsTokenizer in Autopilot mode...", flush=True)
        tokenizer = load_qan_tokenizer(args.model, base_tokenizer=tokenizer)
    
    # Check if we should use lightweight model configuration in tests or mock environments
    from qan_transformers.modeling import AutoQANGraftModel
    if "gemma-4-e2b" in args.model:
        # Load lightweight model configuration for local validation
        model = AutoQANGraftModel.from_pretrained(args.model, sparse_ratio=0.15, framework="pt", lightweight=True)
    else:
        # Load standard model
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        # Graft in-place
        from qan_transformers.modeling import make_quasicrystalline
        model = make_quasicrystalline(model)
        
    model.to(device)
    model.eval()
    
    # 4. Enforce RoPE wrapping boundary
    wrap_rotary_embeddings(model)
    
    if not args.rag:
        # 5. Tokenize the entire codebase
        print("[Tokenization] Tokenizing codebase corpus...")
        codebase_token_ids = tokenizer.encode(codebase_corpus, return_tensors="pt").to(device)
        total_tokens = codebase_token_ids.shape[1]
        print(f"[Tokenization Completed] Codebase tokenized into \033[1;33m{total_tokens:,} tokens\033[0m.")
        
        # 6. Run Chunked Prefill
        chunk_size = 2048
        start_prefill = time.time()
        print(f"[Prefill] Ingesting codebase chunk-by-chunk (size={chunk_size})...")
        
        for i in range(0, total_tokens, chunk_size):
            chunk = codebase_token_ids[:, i : i + chunk_size]
            pos_ids = torch.arange(i, min(total_tokens, i + chunk_size), device=device).unsqueeze(0)
            with torch.no_grad():
                _ = model(chunk, position_ids=pos_ids, use_cache=True)
                
        print(f"[Prefill Completed] Ingested codebase in \033[1;32m{time.time() - start_prefill:.2f} seconds\033[0m.")
        if hasattr(tokenizer, "organism") and tokenizer.organism is not None:
            tokenizer.organism.snapshot()
        
        # 7. Lock codebase memory context
        print("[Locking] Freezing KV-cache coordinates as read-only codebase memory...")
        for m in model.modules():
            cache = getattr(m, "custom_kv_cache", None)
            if cache is not None and isinstance(cache, dict) and cache.get("K", None) is not None:
                m.locked_book_cache = {
                    "K": cache["K"].clone() if cache["K"] is not None else None,
                    "V": cache["V"].clone() if cache["V"] is not None else None,
                    "indices": cache["indices"].clone() if cache["indices"] is not None else None,
                    "alignment_scores": cache["alignment_scores"].clone() if cache["alignment_scores"] is not None else None,
                    "seq_len": cache["seq_len"]
                }
            
    # 8. Setup chat template
    messages = [
        {"role": "user", "content": "Please register the entire codebase files I have preloaded into your memory. I will now ask you questions about the files, classes, modules, and code layout of this project."},
        {"role": "assistant", "content": "Understood. The codebase context has been locked into my Quasicrystalline memory cache. I am ready to explain code structures, locate functions, and answer your development questions!"}
    ]
    
    print("\n" + "="*70)
    print("\033[1;36m      QAN-ATLAS CODESPACE CHAT ACTIVE: DIRECT CODEBASE INTERACTION\033[0m")
    print("="*70)
    print("Ask questions about code structures, locate modules, or summarize code files!")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("\033[1;33mYou: \033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting codebase chat session.")
            break
            
        if user_input.strip().lower() == "exit":
            print("Exiting codebase chat session.")
            break
            
        if not user_input.strip():
            continue
            
        if args.rag:
            # Clear previous turn's custom KV cache
            for m in model.modules():
                if hasattr(m, "custom_kv_cache"):
                    m.custom_kv_cache = {
                        "K": None,
                        "V": None,
                        "indices": None,
                        "alignment_scores": None,
                        "seq_len": 0
                    }
            
            # Query E8 database for relevant codebase chunks
            print(f"  [RAG] Searching E8 lattice database for: '{user_input}'...")
            matched = indexer.query(user_input)
            
            if matched:
                print(f"\033[1;35m  [RAG] Retrieved {len(matched)} matching codebase chunks:\033[0m")
                for i, item in enumerate(matched[:5]):
                    rel_f = os.path.relpath(item["file"], args.folder)
                    print(f"    - {rel_f} (chunk {i+1})")
                    
                # Concatenate matched text to prefill context
                rag_context = ""
                for item in matched:
                    rel_f = os.path.relpath(item["file"], args.folder)
                    rag_context += f"\n--- FILE: {rel_f} ---\n{item['text']}\n"
                    
                # Tokenize and prefill context
                rag_token_ids = tokenizer.encode(rag_context, return_tensors="pt").to(device)
                total_rag_tokens = rag_token_ids.shape[1]
                print(f"  [RAG Prefill] Prefilling {total_rag_tokens} tokens of retrieved context...")
                
                chunk_size = 2048
                for i in range(0, total_rag_tokens, chunk_size):
                    chunk = rag_token_ids[:, i : i + chunk_size]
                    pos_ids = torch.arange(i, min(total_rag_tokens, i + chunk_size), device=device).unsqueeze(0)
                    with torch.no_grad():
                        _ = model(chunk, position_ids=pos_ids, use_cache=True)
            else:
                print("  [RAG] No matching codebase chunks found.")
        else:
            # Restore prefilled codebase memory at the start of each turn
            for m in model.modules():
                if hasattr(m, "locked_book_cache") and m.locked_book_cache is not None:
                    m.custom_kv_cache = {
                        "K": m.locked_book_cache["K"].clone() if m.locked_book_cache["K"] is not None else None,
                        "V": m.locked_book_cache["V"].clone() if m.locked_book_cache["V"] is not None else None,
                        "indices": m.locked_book_cache["indices"].clone() if m.locked_book_cache["indices"] is not None else None,
                        "alignment_scores": m.locked_book_cache["alignment_scores"].clone() if m.locked_book_cache["alignment_scores"] is not None else None,
                        "seq_len": m.locked_book_cache["seq_len"]
                    }
                
        messages.append({"role": "user", "content": user_input})
        
        # Ingest user input and generate response
        input_ids = prepare_chat_input(tokenizer, messages, device)
        S_prompt = input_ids.shape[1]
        print(f"  [Context Info] Total sequence context size: {S_prompt:,} tokens")
        print("  [Processing] Searching locked memory context and generating response...")
        
        start_gen = time.time()
        
        # Determine total tokens in locked cache (if present)
        locked_len = 0
        for m in model.modules():
            if hasattr(m, "locked_book_cache") and m.locked_book_cache is not None:
                locked_len = m.locked_book_cache.get("seq_len", 0)
                break
                
        # Pre-allocate output buffer and inputs for decoding
        max_new_tokens = 128
        B = input_ids.shape[0]
        
        # Pre-allocated output tokens buffer
        output_ids = torch.zeros((B, max_new_tokens), dtype=torch.long, device=device)
        output_ids_cpu = [0] * max_new_tokens
        
        # Pre-allocated single-token input tensor for autoregressive steps
        step_input = torch.zeros((B, 1), dtype=torch.long, device=device)
        
        # Pre-allocated position ID tensor
        step_pos = torch.zeros((B, 1), dtype=torch.long, device=device)
        
        from transformers.cache_utils import DynamicCache
        past_key_values = DynamicCache()
        
        # Set up stop token IDs
        stop_token_ids = set()
        if tokenizer.eos_token_id is not None:
            stop_token_ids.add(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            stop_token_ids.add(tokenizer.pad_token_id)
        for tok in ["<turn|>", "<eos>", "<pad>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and not isinstance(tid, dict) and tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)
                
        actual_generated_len = 0
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    # Initial prompt prefill step
                    # Position IDs start from locked_len to locked_len + S_prompt
                    position_ids = torch.arange(locked_len, locked_len + S_prompt, device=device).unsqueeze(0)
                    outputs = model(input_ids, position_ids=position_ids, use_cache=True, past_key_values=past_key_values)
                else:
                    # Autoregressive generation steps
                    # Reuse pre-allocated single-token input and position ID tensors (zero-allocation)
                    step_input[0, 0] = output_ids[0, step - 1]
                    step_pos[0, 0] = locked_len + S_prompt + step - 1
                    outputs = model(step_input, position_ids=step_pos, use_cache=True, past_key_values=past_key_values)
                    
                logits = outputs[0] if isinstance(outputs, tuple) or isinstance(outputs, list) else getattr(outputs, "logits", outputs)
                next_token_logits = logits[:, -1, :]
                
                # Temperature & Top-K/Top-P filtering
                temperature = 0.7
                top_k = 50
                top_p = 0.9
                
                if temperature > 0.0:
                    next_token_logits = next_token_logits / temperature
                    if top_k > 0:
                        val_k, _ = torch.topk(next_token_logits, top_k)
                        indices_to_remove = next_token_logits < val_k[..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                        
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')
                        
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                token_id = next_token[0, 0].item()
                output_ids[0, step] = token_id
                output_ids_cpu[step] = token_id
                actual_generated_len += 1
                
                if token_id in stop_token_ids:
                    break
                    
        response_ids = output_ids_cpu[:actual_generated_len]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(f"\033[1;32mAplha-Atlas: \033[0m{response_text.strip()}")
        print(f"  [Latency] Generated in {time.time() - start_gen:.2f} seconds.")
        print("-" * 70 + "\n")
        
        # Append response to keep conversation history intact
        messages.append({"role": "assistant", "content": response_text.strip()})
        if hasattr(tokenizer, "organism") and tokenizer.organism is not None:
            tokenizer.organism.snapshot()
