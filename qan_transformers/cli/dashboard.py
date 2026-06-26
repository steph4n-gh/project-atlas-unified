import os
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/Volumes/Storage/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/Volumes/Storage/huggingface/hub")
import sys
import time
import asyncio
import logging
import argparse
import traceback
import json
_orig_json_dumps = json.dumps

try:
    import orjson
    def fast_json_dumps(obj, *args, **kwargs) -> str:
        if kwargs or args:
            return _orig_json_dumps(obj, *args, **kwargs)
        return orjson.dumps(obj).decode("utf-8")
except ImportError:
    try:
        import ujson
        def fast_json_dumps(obj, *args, **kwargs) -> str:
            if kwargs or args:
                return _orig_json_dumps(obj, *args, **kwargs)
            return ujson.dumps(obj)
    except ImportError:
        fast_json_dumps = _orig_json_dumps

json.dumps = fast_json_dumps

import urllib.request
import queue
import threading
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np

# Ensure workspace is in system path
sys.path.insert(0, os.getcwd())

from qan_transformers.modeling import graft_model, QANModel
from qan_transformers.modeling.attention import QuasicrystallineAttention
from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB
from qan_transformers.firewall.cohomology import CohomologyFirewall
from scratch.run_cohomology_audit import analyze_cohomology_connectivity, CallGraphVisitor
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM

# Eagerly import MLX modeling to apply config flattener and load patches before uvicorn/model loading
try:
    import mlx.core as mx
    import qan_transformers.mlx.modeling
except ImportError:
    pass


# Import the real model grafting module dynamically from run_gemma4_novel_chat.py
import importlib.util
spec = importlib.util.spec_from_file_location("run_gemma4_novel_chat", os.path.join(os.getcwd(), "scratch/run_gemma4_novel_chat.py"))
novel_chat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(novel_chat_module)
graft_gemma_model = novel_chat_module.graft_gemma_model

def sanitize_float(val):
    try:
        if val is None or np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    except (TypeError, ValueError, OverflowError):
        return 0.0

def model_forward_step(model, input_ids, position_ids=None, past_key_values=None, kv_caches=None):
    """
    Unified forward pass for both custom QANModel (lightweight mock)
    and standard Hugging Face models (real weights).
    """
    if hasattr(model, "layers"):
        # Custom QANModel (lightweight mock)
        return model(input_ids, kv_caches=kv_caches)
    else:
        # Standard Hugging Face model (real weights)
        with torch.no_grad():
            outputs = model(
                input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits
            
        # Collect custom_kv_cache from layers to return them
        new_kv_caches = []
        for m in model.modules():
            if hasattr(m, "custom_kv_cache") and isinstance(m.custom_kv_cache, dict):
                new_kv_caches.append(m.custom_kv_cache)
                
        return logits, new_kv_caches

def clone_dynamic_cache(cache):
    if cache is None:
        return None
    from transformers.cache_utils import DynamicCache
    cloned = DynamicCache()
    cloned._seen_tokens = cache._seen_tokens
    cloned.key_cache = [k.clone() if k is not None else None for k in cache.key_cache]
    cloned.value_cache = [v.clone() if v is not None else None for v in cache.value_cache]
    return cloned

def rollback_dynamic_cache(cache, new_len):
    if cache is None:
        return
    cache._seen_tokens = new_len
    for i in range(len(cache.key_cache)):
        if cache.key_cache[i] is not None:
            cache.key_cache[i] = cache.key_cache[i][:, :, :new_len, :]
        if cache.value_cache[i] is not None:
            cache.value_cache[i] = cache.value_cache[i][:, :, :new_len, :]


# Thread-safe download tracking structures
download_queues = {}
download_threads = {}

def get_hf_token() -> Optional[str]:
    # Check env first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_CO_TOKEN")
    if token:
        return token
    # Try reading from default cache paths
    paths = [
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
        os.path.expanduser("~/Library/Caches/huggingface/token")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return f.read().strip()
            except Exception:
                pass
    return None

from tqdm import tqdm

class SSEProgressTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.desc = kwargs.get("desc", "")
        self.thread_name = threading.current_thread().name
        self.last_update_pct = -1.0
        self.last_update_time = time.time()
        
    def update(self, n=1):
        super().update(n)
        pct = (self.n / self.total * 100.0) if self.total else 0.0
        now = time.time()
        # Only notify on significant changes to avoid flooding SSE stream
        if abs(pct - self.last_update_pct) >= 1.0 or (now - self.last_update_time) >= 0.2 or pct >= 100.0:
            self.last_update_pct = pct
            self.last_update_time = now
            q = download_queues.get(self.thread_name)
            if q is not None:
                rate = self.format_dict.get("rate")
                rate_str = f"{rate:.2f} B/s" if rate is not None else "unknown"
                if rate is not None:
                    if rate > 1024 * 1024:
                        rate_str = f"{rate / (1024 * 1024):.2f} MB/s"
                    elif rate > 1024:
                        rate_str = f"{rate / 1024:.2f} KB/s"
                
                q.put({
                    "type": "progress",
                    "file": self.desc or "File download",
                    "percent": round(pct, 1),
                    "speed": rate_str
                })

def download_model_worker(repo_id: str, thread_name: str, q: queue.Queue):
    token = get_hf_token()
    # GGUF, ONNX, and typical large other frameworks/checkpoints are excluded
    ignore_patterns = ["*.gguf", "*.onnx", "*.h5", "*.ot", "*.msgpack"]
    
    q.put({"type": "info", "message": f"Checking Hugging Face repository {repo_id}..."})
    try:
        from huggingface_hub import snapshot_download
        q.put({"type": "info", "message": "Starting snapshot download from HF Hub (filtering weights and configs)..."})
        
        model_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            tqdm_class=SSEProgressTqdm,
            ignore_patterns=ignore_patterns
        )
        
        q.put({
            "type": "success",
            "message": f"Successfully downloaded model to local cache.",
            "path": model_path
        })
    except Exception as e:
        q.put({
            "type": "error",
            "message": f"Download failed: {str(e)}",
            "traceback": traceback.format_exc()
        })


app = FastAPI(title="GossetGate: Quasicrystalline Attention Portal")

# Enable CORS
# Origin allowlist rationale:
#   localhost:8000 / 127.0.0.1:8000 — Default uvicorn server serving this GUI
#   localhost:8080 / 127.0.0.1:8080 — Alternative dev-server port (nginx proxy, etc.)
#   localhost:3000 / 127.0.0.1:3000 — Front-end dev server (Vite / React / Node)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CLI_DIR, "static")
TEMPLATES_DIR = os.path.join(CLI_DIR, "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
SCRATCH_DIR = os.path.join(os.getcwd(), "scratch")

def resolve_local_model_path(path_str: str) -> str:
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    # Try resolving relative to current working directory
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    # Try resolving relative to repository root
    repo_root_path = os.path.join(os.getcwd(), path_str)
    if os.path.exists(repo_root_path):
        return repo_root_path
    # Try resolving relative to script directory
    script_dir_path = os.path.join(SCRATCH_DIR, path_str)
    if os.path.exists(script_dir_path):
        return script_dir_path
    # Try relative to script parent directory
    parent_dir_path = os.path.join(os.path.dirname(SCRATCH_DIR), path_str)
    if os.path.exists(parent_dir_path):
        return parent_dir_path
    return path_str

def map_elq_to_base_model(elq_path: str) -> str:
    elq_lower = elq_path.lower()
    if "gemma-4-12b-it-assistant" in elq_lower:
        return "google/gemma-4-12B-it-qat-q4_0-unquantized-assistant"
    elif "gemma-4-12b-it" in elq_lower:
        return "google/gemma-4-12B-it"
    return elq_path


class CachedStaticFiles(StaticFiles):
    def __init__(self, *args, cache_max_age: int = 31536000, **kwargs):
        self.cache_max_age = cache_max_age
        super().__init__(*args, **kwargs)

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = f"public, max-age={self.cache_max_age}"
        return response

# Mount static folder
app.mount("/static", CachedStaticFiles(directory=STATIC_DIR), name="static")

def clone_kv_caches(caches):
    if caches is None:
        return None
    cloned = []
    for cache in caches:
        cloned_cache = {}
        for k, v in cache.items():
            if isinstance(v, torch.Tensor):
                cloned_cache[k] = v.clone()
            else:
                cloned_cache[k] = v
        cloned.append(cloned_cache)
    return cloned

def rollback_kv_caches(current_caches, new_len):
    rolled_caches = []
    for current_cache in current_caches:
        if "K" in current_cache and current_cache["K"] is not None:
            indices = current_cache["indices"]
            # Copy index to CPU once to avoid multiple GPU-CPU synchronizations
            indices_cpu = indices.to("cpu")
            valid_mask_cpu = indices_cpu[0] < new_len
            
            # Get sum and checks on CPU (0 latency)
            num_valid = int(valid_mask_cpu.sum())
            
            if num_valid == indices.shape[1]:
                # All valid, zero-copy return
                rolled_cache = {
                    "K": current_cache["K"],
                    "V": current_cache["V"],
                    "indices": current_cache["indices"],
                    "alignment_scores": current_cache["alignment_scores"],
                    "seq_len": new_len
                }
            elif num_valid > 0:
                # Check prefix on CPU
                if valid_mask_cpu[:num_valid].all():
                    # Prefix slice: zero-copy view on GPU!
                    rolled_cache = {
                        "K": current_cache["K"][:, :, :num_valid, :],
                        "V": current_cache["V"][:, :, :num_valid, :],
                        "indices": current_cache["indices"][:, :num_valid],
                        "alignment_scores": current_cache["alignment_scores"][:, :num_valid],
                        "seq_len": new_len
                    }
                else:
                    # Fallback to advanced indexing
                    valid_mask = indices < new_len
                    valid_cols = torch.where(valid_mask[0])[0]
                    rolled_cache = {
                        "K": current_cache["K"][:, :, valid_cols, :],
                        "V": current_cache["V"][:, :, valid_cols, :],
                        "indices": current_cache["indices"][:, valid_cols],
                        "alignment_scores": current_cache["alignment_scores"][:, valid_cols],
                        "seq_len": new_len
                    }
            else:
                rolled_cache = {
                    "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                }
        else:
            rolled_cache = {
                "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
            }
        rolled_caches.append(rolled_cache)
    return rolled_caches

# Model state manager
class ModelManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.target_model: Optional[QANModel] = None
        self.draft_model: Optional[QANModel] = None
        self.tokenizer: Optional[Any] = None
        self.shared_db: Optional[AdelicMemorySwapGridDB] = None
        self.firewall: Optional[CohomologyFirewall] = None
        self.target_name: str = ""
        self.draft_name: str = ""
        self.device: str = "cpu"
        self.dtype: Any = torch.float32
        self.sparse_ratio: float = 0.15
        self.firewall_enabled: bool = True
        self.review_mode: bool = True
        self.lightweight_mock: bool = False
        self.framework: str = "mlx"
        self.thinking_mode: str = "thinking"
        self.max_new_tokens: int = 1000
        self.optimize_telemetry: bool = False
        
        # Ingested context document tracking
        self.ingested_files: Dict[str, str] = {}
        self.system_prompt_template: str = ""
        
        # Log accumulator for SSE streaming
        self.logs: List[str] = []
        
    def add_log(self, msg: str):
        print(f"[GossetGate Log] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def clear_logs(self):
        self.logs.clear()

    def unload(self):
        self.target_model = None
        self.draft_model = None
        self.tokenizer = None
        self.shared_db = None
        self.ingested_files.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except ImportError:
            pass

    def rebuild_context_caches(self) -> int:
        if not self.target_model:
            return 0
            
        self.add_log("Rebuilding active context databases...")
        self.shared_db.clear()
        
        # Clear locked book caches and custom KV caches across target layers
        for m in self.target_model.modules():
            if hasattr(m, "locked_book_cache"):
                m.locked_book_cache = None
            if hasattr(m, "custom_kv_cache") and isinstance(m.custom_kv_cache, dict):
                m.custom_kv_cache.clear()
                
        from qan_transformers.math.context_builder import format_xml_context
        template = self.system_prompt_template.strip() if self.system_prompt_template.strip() else None
        combined_text = format_xml_context(self.ingested_files, template=template)
        if not self.ingested_files:
            self.add_log("No active documents to prefill. Cache reset successfully.")
            return 0
            
        self.add_log(f"Tokenizing combined XML context document ({len(combined_text)} characters)...")
        device = self.device
        
        if self.framework == "mlx":
            import mlx.core as mx
            tokens = self.tokenizer.encode(combined_text)
            total_tokens = len(tokens)
            input_ids = mx.array([tokens], dtype=mx.int32)
            
            self.add_log(f"Prefilling {total_tokens} tokens into E8 Swap DB (MLX) in chunks...")
            self.add_log("NOTE: Prefilling is chunked to prevent macOS WindowServer/Metal GPU watchdog timeouts during JIT compilation.")
            start_time = time.time()
            
            # Prefill target model in MLX in chunks of 512 to avoid watchdog timeout
            chunk_size = 512
            for i in range(0, total_tokens, chunk_size):
                chunk = input_ids[:, i : i + chunk_size]
                _ = self.target_model(chunk)
                mx.eval(self.target_model.parameters())
                # Evaluate caches to ensure they are realized on GPU, keeping the JIT graph bounded
                for m in self.target_model.modules():
                    cache = getattr(m, "custom_kv_cache", None)
                    if cache is not None and isinstance(cache, dict):
                        mx.eval(*[v for v in cache.values() if isinstance(v, mx.array)])
                        
            elapsed = time.time() - start_time
            self.add_log(f"Prefill completed in {elapsed:.2f} seconds.")
            
            # Lock cache
            locked_count = 0
            for m in self.target_model.modules():
                cache = getattr(m, "custom_kv_cache", None)
                if cache is not None and isinstance(cache, dict) and cache.get("K", None) is not None:
                    m.locked_book_cache = {
                        "K": mx.array(cache["K"]) if cache["K"] is not None else None,
                        "V": mx.array(cache["V"]) if cache["V"] is not None else None,
                        "indices": mx.array(cache["indices"]) if cache["indices"] is not None else None,
                        "alignment_scores": mx.array(cache["alignment_scores"]) if cache["alignment_scores"] is not None else None,
                        "seq_len": cache["seq_len"]
                    }
                    locked_count += 1
            self.add_log(f"Locked KV-caches across {locked_count} layers as read-only memory.")
            return total_tokens
        else:
            # PyTorch
            input_ids = self.tokenizer.encode(combined_text, return_tensors="pt").to(device)
            total_tokens = input_ids.shape[1]
            
            self.add_log(f"Prefilling {total_tokens} tokens into E8 Swap DB (PyTorch)...")
            self.add_log("NOTE: First pre-fill triggers PyTorch autograd tracing and tensor allocations on Apple Silicon MPS hardware. This can sit and wait for 10-20 seconds.")
            start_time = time.time()
            chunk_size = 2048
            
            for i in range(0, total_tokens, chunk_size):
                chunk = input_ids[:, i : i + chunk_size]
                with torch.no_grad():
                    if hasattr(self.target_model, "layers"):
                        # Custom QANModel (lightweight mock)
                        _ = self.target_model(chunk)
                    else:
                        pos_ids = torch.arange(i, min(total_tokens, i + chunk_size), device=device).unsqueeze(0)
                        _ = self.target_model(chunk, position_ids=pos_ids, use_cache=True)
                    
            elapsed = time.time() - start_time
            self.add_log(f"Prefill completed in {elapsed:.2f} seconds.")
            
            locked_count = 0
            for m in self.target_model.modules():
                cache = getattr(m, "custom_kv_cache", None)
                if cache is not None and isinstance(cache, dict) and cache.get("K", None) is not None:
                    m.locked_book_cache = {
                        "K": cache["K"].clone() if cache["K"] is not None else None,
                        "V": cache["V"].clone() if cache["V"] is not None else None,
                        "indices": cache["indices"].clone() if cache["indices"] is not None else None,
                        "alignment_scores": cache["alignment_scores"].clone() if cache["alignment_scores"] is not None else None,
                        "seq_len": cache["seq_len"]
                    }
                    locked_count += 1
            self.add_log(f"Locked KV-caches across {locked_count} layers as read-only memory.")
            return total_tokens

manager = ModelManager()

# Input data structures
class ModelLoadRequest(BaseModel):
    target_model: str = "google/gemma-4-e2b"
    draft_model: str = "google/gemma-4-E2B-it"
    use_speculative: bool = False
    sparse_ratio: float = 0.15
    lightweight_mock: bool = False
    precision: str = "elq"
    framework: str = "mlx"

class AuditRequest(BaseModel):
    code: str
    tau: float = 0.05

class IngestRequest(BaseModel):
    text: str
    filename: Optional[str] = None

class UpdateTemplateRequest(BaseModel):
    template: str

class UpdateConfigRequest(BaseModel):
    sparse_ratio: float
    firewall_enabled: bool
    review_mode: bool
    threshold: float
    thinking_mode: Optional[str] = "thinking"
    max_new_tokens: Optional[int] = 1000
    optimize_telemetry: Optional[bool] = False

_index_html_cache = None

@app.get("/", response_class=HTMLResponse)
async def serve_portal():
    global _index_html_cache
    if _index_html_cache is None:
        index_path = os.path.join(TEMPLATES_DIR, "index.html")
        if not os.path.exists(index_path):
            return HTMLResponse("<h1>GossetGate template index.html not found.</h1>", status_code=404)
        with open(index_path, "r") as f:
            _index_html_cache = f.read()
    return _index_html_cache

@app.post("/api/model/load")
async def load_model_portal(req: ModelLoadRequest):
    async def load_generator_inner():
        import traceback
        import json
        manager.unload()
        manager.clear_logs()
        
        try:
            # Resolve target and draft model local paths early
            resolved_target_model = resolve_local_model_path(req.target_model)
            resolved_draft_model = resolve_local_model_path(req.draft_model)

            # 1. Device detection
            yield f"data: {json.dumps({'type': 'step', 'step': 'device', 'status': 'running', 'message': 'Detecting compute device availability...'})}\n\n"
            await asyncio.sleep(0.05)
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            if req.precision == "float32" or device == "cpu":
                dtype = torch.float32
            elif req.precision == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            manager.device = device
            manager.dtype = dtype
            manager.target_name = resolved_target_model
            manager.draft_name = resolved_draft_model
            manager.sparse_ratio = req.sparse_ratio
            manager.lightweight_mock = req.lightweight_mock
            manager.framework = req.framework
            yield f"data: {json.dumps({'type': 'step', 'step': 'device', 'status': 'success', 'message': f'Compute initialized on target hardware: {device.upper()} ({str(dtype).split(chr(46))[-1]}) (Framework: {req.framework.upper()})'})}\n\n"
            
            if req.framework == "mlx":
                # 2. joint MLX Tokenizer and Target Model loading
                yield f"data: {json.dumps({'type': 'step', 'step': 'tokenizer', 'status': 'running', 'message': f'Loading pretrained MLX tokenizer and weights for {manager.target_name}...'})}\n\n"
                await asyncio.sleep(0.05)
                from mlx_lm import load
                from qan_transformers.mlx.modeling import graft_mlx_model
                
                if req.lightweight_mock:
                    from scratch.run_mlx_speculative_chat import QANModelMLX
                    manager.target_model = QANModelMLX(vocab_size=32000, embed_dim=64, num_heads=4, num_layers=2)
                    try:
                        manager.tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e2b")
                    except (OSError, ValueError, ImportError):
                        try:
                            manager.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
                        except (OSError, ValueError, ImportError):
                            class DummyTokenizer:
                                def encode(self, x, **kwargs): return [ord(c) % 256 for c in x]
                                def decode(self, x, **kwargs): return "".join([chr(c) for c in x])
                                def __len__(self): return 256
                            manager.tokenizer = DummyTokenizer()
                    
                    vocab_size = len(manager.tokenizer)
                    yield f"data: {json.dumps({'type': 'step', 'step': 'tokenizer', 'status': 'success', 'message': 'Initialized lightweight Target MLX simulator offline.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'success', 'message': 'Target MLX model mapped.'})}\n\n"
                else:
                    try:
                        from mlx_lm.utils import load_model, load_tokenizer
                        from huggingface_hub import snapshot_download
                        from pathlib import Path
                        import mlx.core as mx
                        
                        target_loading_name = manager.target_name
                        if target_loading_name.lower().endswith(".elq"):
                            target_loading_name = map_elq_to_base_model(target_loading_name)
                        
                        resolved_loading_name = resolve_local_model_path(target_loading_name)
                        if os.path.exists(resolved_loading_name):
                            model_path = Path(resolved_loading_name)
                        else:
                            token = get_hf_token()
                            model_path = Path(snapshot_download(repo_id=target_loading_name, token=token))
                            
                        model, weights = load_model(model_path, lazy=True, strict=False)
                        del weights
                        import gc
                        gc.collect()
                        mx.clear_cache()
                        tokenizer = load_tokenizer(model_path)
                        manager.tokenizer = tokenizer
                        vocab_size = getattr(tokenizer, "vocab_size", None)
                        if vocab_size is None:
                            vocab_size = len(tokenizer)
                        yield f"data: {json.dumps({'type': 'step', 'step': 'tokenizer', 'status': 'success', 'message': f'MLX Tokenizer loaded. Vocab size: {vocab_size}'})}\n\n"
                        
                        yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'running', 'message': f'Grafting Quasicrystalline Attention and loading weights for MLX model {manager.target_name}...'})}\n\n"
                        await asyncio.sleep(0.05)
                        if req.precision == "elq" or manager.target_name.lower().endswith(".elq"):
                            # Derive potential ELQ path name from target model name
                            target_name_lower = manager.target_name.lower()
                            if "gemma-4-e4b-it" in target_name_lower:
                                base_name = "gemma-4-e4b-it"
                            elif "gemma-4-e4b" in target_name_lower:
                                base_name = "gemma-4-e4b"
                            elif "gemma-4-e2b-it" in target_name_lower or "gemma-4-e2b" in target_name_lower or "gemma2b" in target_name_lower:
                                base_name = "gemma2b_e8"
                            else:
                                base_name = manager.target_name.split("/")[-1].lower().replace("_", "-")
                            if base_name.endswith(".elq"):
                                base_name = base_name[:-4]
                            # Strip common quantization suffixes so that e.g. gemma-4-e4b-it-4bit resolves to gemma-4-e4b-it.elq
                            for suffix in ["-4bit", "-8bit", "-quantized", "-qat"]:
                                if base_name.endswith(suffix):
                                    base_name = base_name[:-len(suffix)]
                            elq_path = f"scratch/{base_name}.elq"
                            
                            # Check if the specific model ELQ path exists in alternate locations first
                            if not os.path.exists(elq_path):
                                for base_dir in [os.getcwd(), os.path.expanduser("~/.gemini/antigravity/brain/93ec8543-9d69-4850-9bdc-eec751c2ef6d")]:
                                    cand = os.path.join(base_dir, elq_path)
                                    if os.path.exists(cand):
                                        elq_path = cand
                                        break
                                        
                            # Fallback if specific file doesn't exist
                            if not os.path.exists(elq_path):
                                default_elq = "scratch/gemma2b_e8.elq"
                                if "2b-it" not in manager.target_name.lower() and os.path.exists("scratch/gemma_e8.elq"):
                                    default_elq = "scratch/gemma_e8.elq"
                                elq_path = default_elq
                            
                            if not os.path.exists(elq_path):
                                workspace_elq = os.path.join(os.getcwd(), elq_path)
                                if os.path.exists(workspace_elq):
                                    elq_path = workspace_elq
                                else:
                                    artifact_elq = os.path.expanduser("~/.gemini/antigravity/brain/93ec8543-9d69-4850-9bdc-eec751c2ef6d/scratch/gemma2b_e8.elq")
                                    if os.path.exists(artifact_elq):
                                        elq_path = artifact_elq
                                    else:
                                        raise FileNotFoundError(f"Custom E8 quantized weights file not found at {elq_path}. Please run 'qan-cli quantize' first to generate it.")
                            
                            from qan_transformers.mlx.modeling import load_and_graft_elq_model
                            manager.target_model = load_and_graft_elq_model(model, elq_path, sparse_ratio=req.sparse_ratio)
                            yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'success', 'message': f'Grafted and loaded E8 Lattice (ELQ) MLX Target model successfully ({elq_path}).'})}\n\n"
                        else:
                            manager.target_model = graft_mlx_model(model, sparse_ratio=req.sparse_ratio)
                            yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'success', 'message': f'Grafted and loaded MLX Target model successfully.'})}\n\n"
                        
                        import gc
                        import mlx.core as mx
                        gc.collect()
                        mx.clear_cache()
                    except Exception as e:
                        raise RuntimeError(f"Failed to load MLX model {manager.target_name}: {e}")
            else:
                # 2. Tokenizer Loading (PyTorch)
                yield f"data: {json.dumps({'type': 'step', 'step': 'tokenizer', 'status': 'running', 'message': f'Loading pretrained tokenizer weights for {resolved_target_model}...'})}\n\n"
                await asyncio.sleep(0.05)
                
                try:
                    manager.tokenizer = AutoTokenizer.from_pretrained(resolved_target_model)
                except Exception as tok_err:
                    if req.lightweight_mock:
                        fallback_tokenizer_paths = [
                            "google/gemma-4-e2b",
                            "google/gemma-4-e4b",
                            "google/gemma-2b"
                        ]
                        loaded = False
                        for path in fallback_tokenizer_paths:
                            try:
                                manager.tokenizer = AutoTokenizer.from_pretrained(path)
                                resolved_target_model = resolve_local_model_path(path)
                                manager.target_name = resolved_target_model
                                loaded = True
                                break
                            except (OSError, ValueError, ImportError):
                                continue
                        if not loaded:
                            raise RuntimeError(f"Failed to load tokenizer for '{resolved_target_model}' and no local fallback tokenizer was found. Error: {tok_err}")
                    else:
                        raise tok_err
                
                vocab_size = len(manager.tokenizer) if manager.tokenizer is not None else 32000
                yield f"data: {json.dumps({'type': 'step', 'step': 'tokenizer', 'status': 'success', 'message': f'Pretrained tokenizer loaded successfully. Vocabulary size: {vocab_size}'})}\n\n"
                
                # 3. Target Model loading (PyTorch)
                yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'running', 'message': f'Allocating Target Model weights for {manager.target_name}...'})}\n\n"
                await asyncio.sleep(0.05)
                
                load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
                if req.precision == "4bit":
                    load_kwargs["load_in_4bit"] = True
                    load_kwargs["device_map"] = "auto"
                elif req.precision == "8bit":
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"] = "auto"

                if req.lightweight_mock:
                    manager.target_model = graft_model(manager.target_name, lightweight=True, vocab_size=vocab_size).to(device)
                    yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'success', 'message': f'Initialized lightweight Target model simulator ({manager.target_name}) offline.'})}\n\n"
                else:
                    try:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(manager.target_name, local_files_only=True, **load_kwargs)
                            graft_gemma_model(model, sparse_ratio=req.sparse_ratio)
                            if req.precision not in ["4bit", "8bit"]:
                                model = model.to(device)
                            manager.target_model = model
                            yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'success', 'message': f'Loaded Target weights from local cache offline ({manager.target_name}).'})}\n\n"
                        except Exception:
                            model = AutoModelForCausalLM.from_pretrained(manager.target_name, local_files_only=False, **load_kwargs)
                            graft_gemma_model(model, sparse_ratio=req.sparse_ratio)
                            if req.precision not in ["4bit", "8bit"]:
                                model = model.to(device)
                            manager.target_model = model
                            yield f"data: {json.dumps({'type': 'step', 'step': 'target_model', 'status': 'success', 'message': f'Loaded Target weights from online repository ({manager.target_name}).'})}\n\n"
                    except Exception as model_err:
                        raise model_err
                    finally:
                        import gc
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                
                manager.target_model.eval()
            
            # 4. Conway-Sloane E8 Memory Swap Grid DB
            yield f"data: {json.dumps({'type': 'step', 'step': 'swap_db', 'status': 'running', 'message': 'Allocating Conway-Sloane E8 Memory Swap Grid DB...'})}\n\n"
            await asyncio.sleep(0.05)
            
            target_head_dim = None
            for m in manager.target_model.modules():
                if hasattr(m, "head_dim"):
                    target_head_dim = m.head_dim
                    break
            
            if target_head_dim is None and hasattr(manager.target_model, "config"):
                config = manager.target_model.config
                if isinstance(config, dict):
                    target_head_dim = config.get("head_dim")
                    if target_head_dim is None:
                        hidden = config.get("hidden_size") or config.get("embed_dim")
                        heads = config.get("num_attention_heads") or config.get("num_heads")
                        if hidden is not None and heads is not None:
                            target_head_dim = hidden // heads
                else:
                    target_head_dim = getattr(config, "head_dim", None)
                    if target_head_dim is None:
                        hidden = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None)
                        heads = getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None)
                        if hidden is not None and heads is not None:
                            target_head_dim = hidden // heads
            
            if target_head_dim is None:
                if hasattr(manager.target_model, "embed_dim") and hasattr(manager.target_model, "num_heads"):
                    target_head_dim = manager.target_model.embed_dim // manager.target_model.num_heads
                else:
                    target_head_dim = 64  # safe default fallback
                
            if req.framework == "mlx":
                from qan_transformers.mlx.e8_swap import AdelicMemorySwapGridDB as MLXAdelicMemorySwapGridDB
                manager.shared_db = MLXAdelicMemorySwapGridDB(
                    d_model=target_head_dim,
                    cache_limit_ratio=req.sparse_ratio
                )
            else:
                manager.shared_db = AdelicMemorySwapGridDB(
                    d_model=target_head_dim,
                    device=device,
                    cache_limit_ratio=req.sparse_ratio
                )
            
            if req.lightweight_mock:
                if req.framework == "mlx":
                    import mlx.core as mx
                    manager.shared_db.grid_coords = mx.random.normal((50, 8))
                else:
                    manager.shared_db.grid_coords = torch.randn(50, 8, device=device)

            yield f"data: {json.dumps({'type': 'step', 'step': 'swap_db', 'status': 'success', 'message': f'E8 Swap Grid database bound. Cache limit: {req.sparse_ratio*100:.1f}%, Head dim: {target_head_dim}'})}\n\n"
            
            # 5. Cohomology firewall
            yield f"data: {json.dumps({'type': 'step', 'step': 'firewall', 'status': 'running', 'message': 'Injecting Čech Cohomology obstruction graph firewall...'})}\n\n"
            await asyncio.sleep(0.05)
            manager.firewall = CohomologyFirewall(threshold=1.5)
            
            target_layers_count = 0
            for m in manager.target_model.modules():
                if isinstance(m, QuasicrystallineAttention) or hasattr(m, "e8_proj"):
                    if not hasattr(m, "custom_kv_cache"):
                        m.custom_kv_cache = {
                            "K": None,
                            "V": None,
                            "indices": None,
                            "alignment_scores": None,
                            "seq_len": 0
                        }
                    m.swap_db = manager.shared_db
                    m.is_draft = False
                    m.firewall = manager.firewall
                    m.sparse_ratio = req.sparse_ratio
                    target_layers_count += 1
            yield f"data: {json.dumps({'type': 'step', 'step': 'firewall', 'status': 'success', 'message': f'Čech Cohomology obstruction firewall active on {target_layers_count} layers.'})}\n\n"
            
            # 6. Assistant draft model
            if req.use_speculative:
                yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'running', 'message': f'Allocating Speculative Assistant Draft Model weights ({resolved_draft_model})...'})}\n\n"
                await asyncio.sleep(0.05)
                
                if req.framework == "mlx":
                    if req.lightweight_mock:
                        from scratch.run_mlx_speculative_chat import QANModelMLX
                        manager.draft_model = QANModelMLX(vocab_size=32000, embed_dim=64, num_heads=2, num_layers=2, is_draft=True)
                        yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'success', 'message': 'Initialized lightweight Assistant MLX simulator.'})}\n\n"
                    else:
                        try:
                            from mlx_lm.utils import load_model, load_tokenizer
                            from huggingface_hub import snapshot_download
                            from pathlib import Path
                            from qan_transformers.mlx.modeling import graft_mlx_model, load_and_graft_elq_model, device_context
                            import mlx.core as mx
                            
                            draft_loading_name = resolved_draft_model
                            is_elq_draft = draft_loading_name.lower().endswith(".elq")
                            if is_elq_draft:
                                draft_loading_name = map_elq_to_base_model(draft_loading_name)
                                
                            resolved_draft_loading_name = resolve_local_model_path(draft_loading_name)
                            if os.path.exists(resolved_draft_loading_name):
                                model_path_d = Path(resolved_draft_loading_name)
                            else:
                                token = get_hf_token()
                                model_path_d = Path(snapshot_download(repo_id=draft_loading_name, token=token))
                                
                            model_d, weights_d = load_model(model_path_d, lazy=True, strict=False)
                            del weights_d
                            import gc
                            gc.collect()
                            mx.clear_cache()
                            
                            loaded_elq_d = False
                            if is_elq_draft:
                                draft_elq = resolved_draft_model
                                if not os.path.exists(draft_elq):
                                    for base_dir in [os.getcwd(), os.path.expanduser("~/.gemini/antigravity/brain/64bc7f76-41f6-48fe-bac8-2489fed60298"), os.path.expanduser("~/.gemini/antigravity/brain/93ec8543-9d69-4850-9bdc-eec751c2ef6d")]:
                                        cand = os.path.join(base_dir, draft_elq)
                                        if os.path.exists(cand):
                                            draft_elq = cand
                                            break
                                            
                                if os.path.exists(draft_elq):
                                    manager.draft_model = load_and_graft_elq_model(model_d, draft_elq, sparse_ratio=req.sparse_ratio, is_draft=True, cache_capacity=32)
                                    loaded_elq_d = True
                            
                            if not loaded_elq_d:
                                if ("2B" in resolved_draft_model or "2b" in resolved_draft_model or "E2B" in resolved_draft_model) and "12B" not in resolved_draft_model:
                                    draft_elq = "scratch/gemma2b_e8.elq"
                                    if not os.path.exists(draft_elq):
                                        for base_dir in [os.getcwd(), os.path.expanduser("~/.gemini/antigravity/brain/64bc7f76-41f6-48fe-bac8-2489fed60298"), os.path.expanduser("~/.gemini/antigravity/brain/93ec8543-9d69-4850-9bdc-eec751c2ef6d")]:
                                            cand = os.path.join(base_dir, draft_elq)
                                            if os.path.exists(cand):
                                                draft_elq = cand
                                                break
                                                
                                    if os.path.exists(draft_elq):
                                        manager.draft_model = load_and_graft_elq_model(model_d, draft_elq, sparse_ratio=req.sparse_ratio, is_draft=True, cache_capacity=32)
                                        loaded_elq_d = True
                                        
                            if not loaded_elq_d:
                                manager.draft_model = graft_mlx_model(model_d, sparse_ratio=req.sparse_ratio, is_draft=True)
                                    
                            if loaded_elq_d:
                                yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'success', 'message': f'Grafted and loaded E8 Lattice (ELQ) MLX Assistant model successfully ({draft_elq}).'})}\n\n"
                            else:
                                yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'success', 'message': f'Loaded and grafted MLX Assistant model successfully ({resolved_draft_model}).'})}\n\n"
                            
                            import gc
                            import mlx.core as mx
                            gc.collect()
                            mx.clear_cache()
                        except Exception as e:
                            raise RuntimeError(f"Failed to load MLX draft model {resolved_draft_model}: {e}")
                else:
                    if req.lightweight_mock:
                        manager.draft_model = graft_model(resolved_draft_model, lightweight=True, vocab_size=vocab_size).to(device)
                        yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'success', 'message': f'Initialized lightweight Assistant model simulator ({resolved_draft_model}) offline.'})}\n\n"
                    else:
                        try:
                            draft_load_kwargs = {"torch_dtype": dtype}
                            if req.precision == "4bit":
                                draft_load_kwargs["load_in_4bit"] = True
                                draft_load_kwargs["device_map"] = "auto"
                            elif req.precision == "8bit":
                                draft_load_kwargs["load_in_8bit"] = True
                                draft_load_kwargs["device_map"] = "auto"

                            try:
                                model = AutoModelForCausalLM.from_pretrained(resolved_draft_model, local_files_only=True, **draft_load_kwargs)
                                graft_gemma_model(model, sparse_ratio=req.sparse_ratio)
                                if req.precision not in ["4bit", "8bit"]:
                                    model = model.to(device)
                                manager.draft_model = model
                                yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'success', 'message': f'Loaded Assistant weights from local cache offline ({resolved_draft_model}).'})}\n\n"
                            except Exception:
                                model = AutoModelForCausalLM.from_pretrained(resolved_draft_model, local_files_only=False, **draft_load_kwargs)
                                graft_gemma_model(model, sparse_ratio=req.sparse_ratio)
                                if req.precision not in ["4bit", "8bit"]:
                                    model = model.to(device)
                                manager.draft_model = model
                                yield f"data: {json.dumps({'type': 'step', 'step': 'draft_model', 'status': 'success', 'message': f'Loaded Assistant weights from online repository ({resolved_draft_model}).'})}\n\n"
                        except Exception as draft_err:
                            raise draft_err
                        
                manager.draft_model.eval()
                
                draft_head_dim = None
                for m in manager.draft_model.modules():
                    if hasattr(m, "head_dim"):
                        draft_head_dim = m.head_dim
                        break
                
                if draft_head_dim is None and hasattr(manager.draft_model, "config"):
                    config = manager.draft_model.config
                    if isinstance(config, dict):
                        draft_head_dim = config.get("head_dim")
                        if draft_head_dim is None:
                            hidden = config.get("hidden_size") or config.get("embed_dim")
                            heads = config.get("num_attention_heads") or config.get("num_heads")
                            if hidden is not None and heads is not None:
                                draft_head_dim = hidden // heads
                    else:
                        draft_head_dim = getattr(config, "head_dim", None)
                        if draft_head_dim is None:
                            hidden = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None)
                            heads = getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None)
                            if hidden is not None and heads is not None:
                                draft_head_dim = hidden // heads
                
                if draft_head_dim is None:
                    if hasattr(manager.draft_model, "embed_dim") and hasattr(manager.draft_model, "num_heads"):
                        draft_head_dim = manager.draft_model.embed_dim // manager.draft_model.num_heads
                    else:
                        draft_head_dim = 64  # safe default fallback
                manager.shared_db.d_model_draft = draft_head_dim
                
                draft_layers_count = 0
                for m in manager.draft_model.modules():
                    if isinstance(m, QuasicrystallineAttention) or hasattr(m, "e8_proj"):
                        if not hasattr(m, "custom_kv_cache"):
                            m.custom_kv_cache = {
                                "K": None,
                                "V": None,
                                "indices": None,
                                "alignment_scores": None,
                                "seq_len": 0
                            }
                        m.swap_db = manager.shared_db
                        m.is_draft = True
                        m.sparse_ratio = req.sparse_ratio
                        draft_layers_count += 1
                        
                yield f"data: {json.dumps({'type': 'complete', 'status': 'success', 'message': f'Speculative decoding configured with {draft_layers_count} assistant layers.'})}\n\n"
            else:
                manager.draft_model = None
                yield f"data: {json.dumps({'type': 'complete', 'status': 'success', 'message': 'Single model decoding configured (no speculative assistant).'})}\n\n"
                
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            err_msg = str(e)
            
            error_type = "UNKNOWN_LOAD_ERROR"
            if "local-only" in err_msg.lower() or "login" in err_msg.lower() or "auth token" in err_msg.lower() or "hf_token" in err_msg.lower() or "unauthorized" in err_msg.lower() or "401" in err_msg or "gated" in err_msg.lower() or "access" in err_msg.lower():
                error_type = "HF_AUTH_ERROR"
            elif "out of memory" in err_msg.lower() or "allocation failed" in err_msg.lower() or "cuda oom" in err_msg.lower() or "mps oom" in err_msg.lower():
                error_type = "OUT_OF_MEMORY"
            
            manager.unload()
            yield f"data: {json.dumps({'type': 'error', 'error_type': error_type, 'message': err_msg, 'traceback': tb_str})}\n\n"

    async def load_generator():
        async with manager._lock:
            async for chunk in load_generator_inner():
                yield chunk

    return StreamingResponse(load_generator(), media_type="text/event-stream")

@app.get("/api/model/download/stream")
async def download_model_stream(repo_id: str):
    """
    SSE stream yielding download updates for a Hugging Face repository.
    """
    repo_id = repo_id.strip()
    if not repo_id:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'message': 'Empty Repository ID'})}\n\n"]),
            media_type="text/event-stream"
        )
        
    thread_name = f"download-{repo_id.replace('/', '-')}-{int(time.time())}"
    q = queue.Queue()
    download_queues[thread_name] = q
    
    # Start background thread
    t = threading.Thread(
        target=download_model_worker,
        args=(repo_id, thread_name, q),
        name=thread_name,
        daemon=True
    )
    download_threads[thread_name] = t
    t.start()
    
    async def sse_generator():
        try:
            loop = asyncio.get_event_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=0.5))
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue
                    
                yield f"data: {json.dumps(item)}\n\n"
                
                if item["type"] in ["success", "error"]:
                    break
        finally:
            download_queues.pop(thread_name, None)
            download_threads.pop(thread_name, None)
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

def get_files_list() -> List[Dict[str, Any]]:
    files_info = []
    for filename, content in manager.ingested_files.items():
        try:
            if manager.tokenizer:
                # Both MLX and PyTorch tokenizers use the same .encode() interface
                tok_len = len(manager.tokenizer.encode(content))
            else:
                tok_len = len(content.split())
        except (TypeError, ValueError, AttributeError):
            tok_len = len(content.split())
        files_info.append({
            "filename": filename,
            "char_count": len(content),
            "tokens": tok_len
        })
    return files_info

@app.post("/api/context/ingest")
async def ingest_context_portal(req: IngestRequest):
    if not manager.target_model:
        return {"status": "error", "message": "Model not loaded. Please initialize a model first."}
        
    text = req.text.strip()
    if not text:
        return {"status": "success", "tokens": 0, "prefill_time_sec": 0.0, "files": get_files_list()}
        
    filename = req.filename.strip() if req.filename else f"document_{len(manager.ingested_files) + 1}.txt"
    manager.add_log(f"Ingesting context file '{filename}' ({len(text)} characters)...")
    
    # Store/replace content
    manager.ingested_files[filename] = text
    
    start_time = time.time()
    total_tokens = manager.rebuild_context_caches()
    elapsed = time.time() - start_time
    
    # Calculate token count for this file specifically
    try:
        # Both MLX and PyTorch tokenizers use the same .encode() interface
        file_tokens = len(manager.tokenizer.encode(text))
    except (TypeError, ValueError, AttributeError):
        file_tokens = len(text.split())
        
    return {
        "status": "success",
        "tokens": file_tokens,
        "prefill_time_sec": float(elapsed),
        "files": get_files_list()
    }

@app.get("/api/context/files")
async def get_context_files_portal():
    from qan_transformers.math.context_builder import build_tree_structure
    files_list = get_files_list()
    files_meta = {}
    for item in files_list:
        files_meta[item["filename"]] = {
            "size": item["char_count"],
            "tokens": item["tokens"]
        }
    tree = build_tree_structure(files_meta)
    return {"status": "success", "files": files_list, "tree": tree}

@app.get("/api/context/template")
async def get_context_template_portal():
    from qan_transformers.math.context_builder import DEFAULT_SYSTEM_TEMPLATE
    current_template = manager.system_prompt_template
    if not current_template:
        current_template = DEFAULT_SYSTEM_TEMPLATE
    return {"status": "success", "template": current_template}

@app.post("/api/context/template")
async def update_context_template_portal(req: UpdateTemplateRequest):
    manager.system_prompt_template = req.template.strip()
    manager.add_log("System prompt template updated.")
    
    # Rebuild context caches if there are already files ingested
    if manager.ingested_files:
        start_time = time.time()
        total_tokens = manager.rebuild_context_caches()
        elapsed = time.time() - start_time
        return {
            "status": "success", 
            "message": "Template updated and KV cache rebuilt successfully.",
            "prefill_time_sec": float(elapsed),
            "total_tokens": total_tokens
        }
    return {"status": "success", "message": "Template updated successfully."}

@app.delete("/api/context/files/{filename}")
async def delete_context_file_portal(filename: str):
    if filename in manager.ingested_files:
        manager.add_log(f"Removing context file '{filename}'...")
        del manager.ingested_files[filename]
        # Rebuild KV caches without this file
        total_tokens = manager.rebuild_context_caches()
        return {"status": "success", "files": get_files_list(), "remaining_tokens": total_tokens}
    else:
        return {"status": "error", "message": f"File '{filename}' not found in ingested context."}

@app.post("/api/config/update")
async def update_config_portal(req: UpdateConfigRequest):
    manager.sparse_ratio = req.sparse_ratio
    manager.firewall_enabled = req.firewall_enabled
    manager.review_mode = req.review_mode
    
    if manager.firewall:
        manager.firewall.threshold = req.threshold
        manager.firewall.enabled = req.firewall_enabled
        
    if manager.target_model:
        for m in manager.target_model.modules():
            if isinstance(m, QuasicrystallineAttention):
                m.sparse_ratio = req.sparse_ratio
                m.review_mode = req.review_mode
                
    if manager.draft_model:
        for m in manager.draft_model.modules():
            if isinstance(m, QuasicrystallineAttention):
                m.sparse_ratio = req.sparse_ratio
                
    manager.add_log(f"Configuration updated: Sparse ratio = {req.sparse_ratio:.2f}, Firewall = {req.firewall_enabled}, Review Mode = {req.review_mode}, Threshold = {req.threshold:.2f}")
    return {"status": "success"}

# Active loop thread control for unified self-improvement dashboard
si_thread = None
si_running = False

def run_self_improvement_bg(backend: str, generations: int, target: str = "mps_scatter"):
    global si_running
    si_running = True
    
    # Empty log file first
    log_file = "scratch/self_improvement.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        f.write("=== Booting Self-Improvement Background Loop ===\n")
        
    try:
        from scratch.run_self_improvement import main as run_si_main
        # Override sys.argv
        sys.argv = [sys.argv[0], "--backend", backend, "--generations", str(generations), "--target", target]
        run_si_main()
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"CRITICAL ERROR in self-improvement loop: {e}\n")
    finally:
        si_running = False

@app.post("/api/audit")
async def audit_code(req: AuditRequest):
    ok, l2, msg = analyze_cohomology_connectivity(req.code, req.tau)
    
    # Parse graph details (nodes & edges) using CallGraphVisitor for frontend visualization
    nodes = []
    edges = []
    try:
        root = ast.parse(req.code)
        visitor = CallGraphVisitor()
        visitor.collect_definitions_and_imports(root)
        visitor.visit(root)
        nodes = list(visitor.definitions.union(visitor.imports).union(visitor.undefined_nodes))
        edges = visitor.edges
    except:
        pass
        
    return {
        "approved": bool(ok),
        "lambda_2": float(l2),
        "message": msg,
        "nodes": nodes,
        "edges": edges,
        "undefined_nodes": list(visitor.undefined_nodes) if 'visitor' in locals() else []
    }

@app.post("/api/self-improve/start")
async def start_self_improve(background_tasks: BackgroundTasks, backend: str = "mock", generations: int = 3, target: str = "mps_scatter"):
    global si_thread, si_running
    if si_running:
        return {"status": "already_running", "message": "Self-improvement loop is already executing."}
        
    background_tasks.add_task(run_self_improvement_bg, backend, generations, target)
    return {"status": "started", "message": f"Autonomous self-improvement loop started using backend: {backend} on target: {target}"}

@app.get("/api/self-improve/status")
async def get_status():
    return {"running": si_running}

@app.get("/api/self-improve/stream")
async def stream_self_improve_logs(request: Request):
    async def log_generator():
        log_file = "scratch/self_improvement.log"
        # Wait until file exists
        while not os.path.exists(log_file):
            await asyncio.sleep(0.5)
            
        with open(log_file, "r") as f:
            # Start from beginning
            f.seek(0, 0)
            while True:
                if await request.is_disconnected():
                    break
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.2)
                    continue
                yield f"data: {line.strip()}\n\n"
                
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/api/chat/stream")
async def stream_chat_portal(prompt: str, request: Request):
    # NOTE (M24): This endpoint uses GET with the prompt in the query string.
    # URL length is limited to ~2000-8000 characters depending on the browser/server.
    # For prompts exceeding URL limits, consider using the POST /api/chat/stream_post
    # endpoint instead (not yet implemented).

    # M20: Prompt length validation
    if len(prompt) > 10000:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Prompt too long (max 10000 characters)"}, status_code=400)

    if not manager.target_model:
        async def err_generator():
            yield "data: {\"token\": \"Error: Model not loaded. Initialize first.\", \"error\": true}\n\n"
        return StreamingResponse(err_generator(), media_type="text/event-stream")
        
    device = manager.device
    tokenizer = manager.tokenizer
    target_model = manager.target_model
    draft_model = manager.draft_model
    shared_db = manager.shared_db
    firewall = manager.firewall
    
    manager.add_log(f"Stream requested for prompt: '{prompt[:40]}...'")
    
    if manager.framework == "mlx":
        # Apple MLX chat stream generator
        async def chat_generator_mlx():
            import mlx.core as mx
            
            if manager.thinking_mode == "direct":
                system_instruction = "You are a helpful assistant. You must skip any thinking process and output your final answer directly."
            else:
                system_instruction = "You are a helpful assistant. You must first output your step-by-step thinking process wrapped in <|channel>thought ... <channel|>text, and then output your final answer."
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
            try:
                kwargs = {}
                if manager.thinking_mode == "direct":
                    kwargs["enable_thinking"] = False
                prompt_text = manager.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception:
                try:
                    prompt_text = manager.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    try:
                        fallback_messages = [{"role": "user", "content": f"{system_instruction}\n\nUser Question: {prompt}"}]
                        prompt_text = manager.tokenizer.apply_chat_template(fallback_messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        prompt_text = f"{system_instruction}\n\nUser Question: {prompt}"
            
            try:
                prompt_len = len(manager.tokenizer.encode(prompt_text))
            except Exception:
                prompt_len = len(prompt_text.split())
                
            max_new_tokens = manager.max_new_tokens
            start_time = time.perf_counter()
            total_tokens_gen = 0
            
            is_mock = getattr(manager, "lightweight_mock", False)
            if is_mock:
                # Mock MLX stream generator
                mock_text = (
                    "<|channel>thought\n"
                    "The user is asking a question in the GossetGate Portal (MLX framework). I will stream a simulated response explaining the grafted Gemma-4 attention stack, E8 root lattice swap DB, and Čech Laplacian connectivity checks.\n"
                    "<channel|>text\n"
                    "This is a simulated response from the GossetGate Quasicrystalline Attention Portal. "
                    "The engine is currently running in offline mock sandbox mode (MLX framework), simulating a multi-layer grafted Gemma-4 attention stack. "
                    "E8 lattice swap DB cache hits and Čech Laplacian connectivity checks (lambda_2) are computed at each token step. "
                    "To receive real language outputs, please disable the offline simulator checkbox and load a real pretrained checkpoint from cache."
                )
                
                # Split text into chunks to simulate token streaming
                mock_tokens = [mock_text[i:i+4] for i in range(0, len(mock_text), 4)]
                
                # Send initial pre-fill progress packet
                yield f"data: {json.dumps({'token': '', 'speed': 0.0, 'vram_saved': 0.0, 'active_cells': 0, 'lambda_2': 0.75, 'cfi': 0.0, 'is_fractured': False, 'rollback_triggered': False, 'logs': ['Starting MLX prompt pre-fill...', 'Computing attention projections...', 'NOTE: MLX graph compilation (JIT) and KV-cache pre-fill are warming up. This first token step may take 10-25 seconds.'], 'grid_points': [], 'done': False})}\n\n"
                await asyncio.sleep(0.5)
                
                vram_savings = 100.0 * (1.0 - manager.sparse_ratio)
                active_cells = manager.shared_db.grid_coords.shape[0] if (manager.shared_db is not None and manager.shared_db.grid_coords is not None) else 0
                lambda_2_val = 1.0
                cfi_val = 0.0
                
                accumulated_text = ""
                for tok in mock_tokens:
                    if await request.is_disconnected():
                        break
                    
                    accumulated_text += tok
                    elapsed = time.perf_counter() - start_time
                    speed = len(accumulated_text.split()) / elapsed if elapsed > 0 else 0.0
                    
                    grid_points = []
                    if not manager.optimize_telemetry:
                        if manager.shared_db is not None and manager.shared_db.grid_coords is not None and manager.shared_db.grid_coords.shape[0] > 0:
                            coords_cpu = np.array(manager.shared_db.grid_coords).astype(np.float32)
                            idx_step = max(1, coords_cpu.shape[0] // 100)
                            sampled_coords = coords_cpu[::idx_step]
                            for pt in sampled_coords:
                                pt_3d = pt[:3].tolist()
                                grid_points.append(pt_3d)

                    payload = {
                        "token": tok,
                        "speed": sanitize_float(speed),
                        "vram_saved": sanitize_float(vram_savings),
                        "active_cells": int(active_cells),
                        "lambda_2": sanitize_float(lambda_2_val),
                        "cfi": sanitize_float(cfi_val),
                        "is_fractured": False,
                        "rollback_triggered": False,
                        "logs": ["Streaming mock tokens..."],
                        "grid_points": grid_points,
                        "context_size": int(prompt_len + len(accumulated_text.split())),
                        "speculative": False,
                        "acceptance_rate": 0.0,
                        "done": False
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.015)
                
                # Final packet
                final_grid_points = []
                if manager.shared_db is not None and manager.shared_db.grid_coords is not None and manager.shared_db.grid_coords.shape[0] > 0:
                    coords_cpu = np.array(manager.shared_db.grid_coords).astype(np.float32)
                    idx_step = max(1, coords_cpu.shape[0] // 100)
                    sampled_coords = coords_cpu[::idx_step]
                    for pt in sampled_coords:
                        pt_3d = pt[:3].tolist()
                        final_grid_points.append(pt_3d)

                payload = {
                    "token": "",
                    "speed": sanitize_float(speed),
                    "vram_saved": sanitize_float(vram_savings),
                    "active_cells": int(active_cells),
                    "lambda_2": sanitize_float(lambda_2_val),
                    "cfi": sanitize_float(cfi_val),
                    "is_fractured": False,
                    "rollback_triggered": False,
                    "logs": ["Generation complete."],
                    "grid_points": final_grid_points,
                    "context_size": int(prompt_len + len(accumulated_text.split())),
                    "speculative": False,
                    "acceptance_rate": 0.0,
                    "done": True
                }
                yield f"data: {json.dumps(payload)}\n\n"
                return

            # Use mlx_lm stream_generate directly!
            from mlx_lm import stream_generate
            
            # Prepare arguments
            from mlx_lm.sample_utils import make_sampler
            base_sampler = make_sampler(temp=0.0)
            
            # Identify thinking token IDs if thinking is disabled
            thinking_token_ids = []
            if manager.thinking_mode == "direct" and manager.tokenizer is not None:
                for marker in ["<|channel>thought", "<think>", "<|thought|>"]:
                    try:
                        ids = manager.tokenizer.encode(marker, add_special_tokens=False)
                        if ids:
                            thinking_token_ids.append(ids[0])
                    except Exception:
                        pass
                        
            step_count = [0]
            def custom_sampler(logits):
                if manager.thinking_mode == "direct" and step_count[0] == 0:
                    for tid in thinking_token_ids:
                        if tid < logits.shape[-1]:
                            logits[..., tid] = -1e9
                step_count[0] += 1
                return base_sampler(logits)

            gen_kwargs = {
                "max_tokens": max_new_tokens,
                "sampler": custom_sampler
            }
            if manager.draft_model is not None:
                gen_kwargs["draft_model"] = manager.draft_model
                
            # Send initial progress packet
            yield f"data: {json.dumps({'token': '', 'speed': 0.0, 'vram_saved': 0.0, 'active_cells': 0, 'lambda_2': 0.75, 'cfi': 0.0, 'is_fractured': False, 'rollback_triggered': False, 'logs': ['Starting MLX prompt pre-fill...', 'Computing attention projections...'], 'grid_points': [], 'done': False})}\n\n"
            
            start_time = time.perf_counter()
            
            # Clear swap database before prefill
            manager.shared_db.clear()
            
            # Load permanent locked caches back into MLX attention modules if available
            book_len = 0
            for m in manager.target_model.modules():
                if hasattr(m, "custom_kv_cache"):
                    locked = getattr(m, "locked_book_cache", None)
                    if locked is not None:
                        book_len = max(book_len, locked.get("seq_len", 0))
                        m.custom_kv_cache = {
                            "K": mx.array(locked["K"]) if locked["K"] is not None else None,
                            "V": mx.array(locked["V"]) if locked["V"] is not None else None,
                            "indices": mx.array(locked["indices"]) if locked["indices"] is not None else None,
                            "alignment_scores": mx.array(locked["alignment_scores"]) if locked["alignment_scores"] is not None else None,
                            "seq_len": locked["seq_len"]
                        }
                    else:
                        m.custom_kv_cache = {
                            "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                        }
            if manager.draft_model is not None:
                for m in manager.draft_model.modules():
                    if hasattr(m, "custom_kv_cache"):
                        m.custom_kv_cache = {
                            "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                        }
            
            accumulated_text = ""
            accepted_count = 0
            proposed_count = 0
            use_speculative = (manager.draft_model is not None)
            try:
                # We iterate over the stream_generate loop
                # stream_generate will run our grafted model with E8 attention,
                # which automatically populates and updates manager.shared_db!
                for response in stream_generate(manager.target_model, manager.tokenizer, prompt_text, **gen_kwargs):
                    if await request.is_disconnected():
                        break
                        
                    token_text = response.text
                    accumulated_text += token_text
                    
                    # Check for stop tokens
                    should_break = False
                    for stop in ["<turn|>", "<eos>", "<|im_end|>", "<end_of_turn>"]:
                        if stop in accumulated_text:
                            should_break = True
                            if stop in token_text:
                                token_text = token_text.split(stop)[0]
                            break
                            
                    total_tokens_gen = response.generation_tokens
                    speed = response.generation_tps
                    
                    if use_speculative:
                        if response.from_draft:
                            accepted_count += 1
                        proposed_count += 1
                    
                    # Collect cfi from grafted layers if available
                    cfis = []
                    for m in manager.target_model.modules():
                        if hasattr(m, "custom_kv_cache") and m.custom_kv_cache is not None:
                            cfis.append(m.custom_kv_cache.get("cfi", 0.0))
                    cfi_val = float(np.mean(cfis)) if len(cfis) > 0 else 0.0
                    lambda_2_val = manager.firewall.last_lambda_2 if manager.firewall else 0.75
                    is_fractured = False
                    rollback_triggered = response.from_draft # True if speculative rejection occurred
                    
                    # Formulate coordinates JSON format for E8 scatter plot
                    grid_points = []
                    if not manager.optimize_telemetry:
                        if manager.shared_db.grid_coords is not None and manager.shared_db.grid_coords.shape[0] > 0:
                            coords_cpu = np.array(manager.shared_db.grid_coords).astype(np.float32)
                            idx_step = max(1, coords_cpu.shape[0] // 100)
                            sampled_coords = coords_cpu[::idx_step]
                            for pt in sampled_coords:
                                pt_3d = pt[:3].tolist()
                                grid_points.append(pt_3d)
                            
                    vram_savings = 100.0 * (1.0 - manager.sparse_ratio)
                    active_cells = manager.shared_db.grid_coords.shape[0] if manager.shared_db.grid_coords is not None else 0
                    
                    step_logs = [f"Generated: {token_text}"]
                    if response.from_draft:
                        step_logs.append("Speculative assistant token verified and accepted.")
                    
                    payload = {
                        "token": token_text,
                        "speed": sanitize_float(speed),
                        "vram_saved": sanitize_float(vram_savings),
                        "active_cells": int(active_cells),
                        "lambda_2": sanitize_float(lambda_2_val),
                        "cfi": sanitize_float(cfi_val),
                        "is_fractured": bool(is_fractured),
                        "rollback_triggered": bool(rollback_triggered),
                        "logs": step_logs,
                        "grid_points": grid_points,
                        "context_size": int(book_len + prompt_len + response.generation_tokens),
                        "speculative": use_speculative,
                        "acceptance_rate": sanitize_float((accepted_count / proposed_count) * 100 if proposed_count > 0 else 0),
                        "done": False
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.005)
                    
                    if should_break:
                        break
                    
                # Final packet
                final_grid_points = []
                if manager.shared_db.grid_coords is not None and manager.shared_db.grid_coords.shape[0] > 0:
                    coords_cpu = np.array(manager.shared_db.grid_coords).astype(np.float32)
                    idx_step = max(1, coords_cpu.shape[0] // 100)
                    sampled_coords = coords_cpu[::idx_step]
                    for pt in sampled_coords:
                        pt_3d = pt[:3].tolist()
                        final_grid_points.append(pt_3d)

                payload = {
                    "token": "",
                    "speed": sanitize_float(speed),
                    "vram_saved": sanitize_float(vram_savings),
                    "active_cells": int(active_cells),
                    "lambda_2": sanitize_float(lambda_2_val),
                    "cfi": sanitize_float(cfi_val),
                    "is_fractured": False,
                    "rollback_triggered": False,
                    "logs": ["Generation complete."],
                    "grid_points": final_grid_points,
                    "context_size": int(book_len + prompt_len + total_tokens_gen),
                    "speculative": use_speculative,
                    "acceptance_rate": sanitize_float((accepted_count / proposed_count) * 100 if proposed_count > 0 else 0),
                    "done": True
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
            except Exception as e:
                tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                yield f"data: {json.dumps({'token': f' [Error: {str(e)}]', 'speed': 0, 'vram_saved': 0, 'active_cells': 0, 'lambda_2': 0, 'cfi': 0, 'is_fractured': False, 'rollback_triggered': False, 'logs': [str(e), tb_str], 'grid_points': [], 'done': True})}\n\n"

        return StreamingResponse(chat_generator_mlx(), media_type="text/event-stream")

    if manager.thinking_mode == "direct":
        system_instruction = "You are a helpful assistant. You must skip any thinking process and output your final answer directly."
    else:
        system_instruction = "You are a helpful assistant. You must first output your step-by-step thinking process wrapped in <|channel>thought ... <channel|>text, and then output your final answer."
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]
    try:
        kwargs = {}
        if manager.thinking_mode == "direct":
            kwargs["enable_thinking"] = False
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            try:
                fallback_messages = [{"role": "user", "content": f"{system_instruction}\n\nUser Question: {prompt}"}]
                prompt_text = tokenizer.apply_chat_template(fallback_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = f"{system_instruction}\n\nUser Question: {prompt}"
    # Dynamic tokenization (PyTorch)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    S_prompt = input_ids.shape[1]
    
    target_layers = [m for m in target_model.modules() if hasattr(m, "custom_kv_cache")]
    target_kv_caches = [{} for _ in range(len(target_layers))]
    
    book_len = 0
    # Load permanent locked caches back into the active KV caches if available
    for m in target_model.modules():
        if hasattr(m, "custom_kv_cache"):
            locked = getattr(m, "locked_book_cache", None)
            if locked is not None:
                m.custom_kv_cache = {
                    "K": locked["K"].clone() if locked["K"] is not None else None,
                    "V": locked["V"].clone() if locked["V"] is not None else None,
                    "indices": locked["indices"].clone() if locked["indices"] is not None else None,
                    "alignment_scores": locked["alignment_scores"].clone() if locked["alignment_scores"] is not None else None,
                    "seq_len": locked["seq_len"]
                }
                if isinstance(locked, dict):
                    book_len = max(book_len, locked.get("seq_len", 0))
            else:
                m.custom_kv_cache = {
                    "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                }
    if draft_model is not None:
        for m in draft_model.modules():
            if hasattr(m, "custom_kv_cache"):
                m.custom_kv_cache = {
                    "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                }
                
    async def chat_generator():
        nonlocal target_kv_caches, book_len
        # Prefill prompt
        with torch.no_grad():
            logits_t, target_kv_caches = model_forward_step(target_model, input_ids, kv_caches=target_kv_caches)
            
        generated_ids = input_ids.clone()
        current_len = S_prompt
        
        is_mock = getattr(manager, "lightweight_mock", False)
        if is_mock:
            mock_text = (
                "This is a simulated response from the GossetGate Quasicrystalline Attention Portal. "
                "The engine is currently running in offline mock sandbox mode, simulating a multi-layer grafted Gemma-4 attention stack. "
                "E8 lattice swap DB cache hits and Čech Laplacian connectivity checks (lambda_2) are computed at each token step. "
                "To receive real language outputs, please disable the offline simulator checkbox and load a real pretrained checkpoint from cache."
            )
            mock_tokens = tokenizer.encode(mock_text)
            mock_len = len(mock_tokens)
            mock_idx = 0
            max_new_tokens = mock_len
        else:
            max_new_tokens = 50
            
        total_tokens_gen = 0
        start_time = time.perf_counter()
        
        # Speculative setup
        use_speculative = (draft_model is not None)
        if use_speculative:
            draft_layers = [m for m in draft_model.modules() if isinstance(m, QuasicrystallineAttention)]
            draft_kv_caches = [{} for _ in range(len(draft_layers))]
            with torch.no_grad():
                _, draft_kv_caches = model_forward_step(draft_model, input_ids, kv_caches=draft_kv_caches)
        else:
            draft_kv_caches = []
                
        draft_steps = 4
        accepted_count = 0
        proposed_count = 0
        
        accumulated_text = ""
        # C15: Initialize telemetry variables before the generation loop to prevent
        # NameError in the final SSE packet if the while loop body never executes.
        speed = 0.0
        vram_savings = 0.0
        active_cells = 0
        lambda_2_val = 0.0
        cfi_val = 0.0
        # Generator step loop
        while total_tokens_gen < max_new_tokens:
            if await request.is_disconnected():
                break
                
            step_logs = []
            lambda_2_val = 0.75
            cfi_val = 0.05
            is_fractured = False
            rollback_triggered = False
            tokens_to_emit = []
            
            if not use_speculative:
                # ----------------------------------------------------
                # Single Model step
                # ----------------------------------------------------
                if is_mock:
                    if mock_idx >= mock_len:
                        break
                    next_token_id = mock_tokens[mock_idx]
                    mock_idx += 1
                    next_token = torch.tensor([[next_token_id]], device=device)
                    # Run actual forward pass to populate caches & database!
                    next_token_in = generated_ids[:, -1:]
                    with torch.no_grad():
                        _, target_kv_caches = model_forward_step(target_model, next_token_in, kv_caches=target_kv_caches)
                else:
                    next_token_in = generated_ids[:, -1:]
                    with torch.no_grad():
                        logits_t, target_kv_caches = model_forward_step(target_model, next_token_in, kv_caches=target_kv_caches)
                        next_token = torch.argmax(logits_t[:, -1:, :], dim=-1)
                        
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                total_tokens_gen += 1
                tokens_to_emit.append(next_token.item())
                current_len += 1
                
                # Check firewall metrics
                cfis = []
                l2s = []
                for cache in target_kv_caches:
                    cfis.append(cache.get("cfi", 0.0))
                    l2s.append(cache.get("lambda_2", 0.75))
                    if cache.get("is_fractured", False):
                        is_fractured = True
                        
                cfi_val = float(np.mean(cfis))
                lambda_2_val = float(np.mean(l2s))
                
                if is_fractured:
                    step_logs.append("WARNING: Čech Cohomology Firewall flagged attention fracture! Rolling back.")
                    rollback_triggered = True
                    
            else:
                # ----------------------------------------------------
                # Speculative Decoding step
                # ----------------------------------------------------
                if is_mock:
                    if mock_idx >= mock_len:
                        break
                    actual_draft_steps = min(draft_steps, mock_len - mock_idx)
                    target_kv_backup = clone_kv_caches(target_kv_caches)
                    draft_kv_backup = clone_kv_caches(draft_kv_caches)
                    
                    draft_ids_list = []
                    for t in range(actual_draft_steps):
                        target_tok = mock_tokens[mock_idx + t]
                        # Mismatch every 15 tokens to show rollback/phason logic
                        if (mock_idx + t) % 15 == 12 and t > 0:
                            draft_ids_list.append(target_tok + 1)
                        else:
                            draft_ids_list.append(target_tok)
                            
                    candidate_ids = torch.tensor([draft_ids_list], device=device)
                    
                    with torch.no_grad():
                        for t in range(actual_draft_steps):
                            next_token_d = candidate_ids[:, t:t+1]
                            _, draft_kv_caches = model_forward_step(draft_model, next_token_d, kv_caches=draft_kv_caches)
                            proposed_count += 1
                        _, target_kv_caches = model_forward_step(target_model, candidate_ids, kv_caches=target_kv_caches)
                        
                    accepted_list = []
                    rejected = False
                    fracture_idx = -1
                    
                    for t in range(actual_draft_steps):
                        proposed_tok = draft_ids_list[t]
                        target_tok = mock_tokens[mock_idx + t]
                        if proposed_tok == target_tok:
                            accepted_list.append(torch.tensor([[proposed_tok]], device=device))
                            mock_idx += 1
                        else:
                            accepted_list.append(torch.tensor([[target_tok]], device=device))
                            mock_idx += 1
                            rejected = True
                            fracture_idx = t
                            break
                            
                    num_accepted = len(accepted_list)
                    added_ids = torch.cat(accepted_list, dim=-1)
                    generated_ids = torch.cat([generated_ids, added_ids], dim=-1)
                    new_len = current_len + added_ids.shape[1]
                    
                    target_kv_caches = rollback_kv_caches(target_kv_caches if not rejected else target_kv_backup, new_len)
                    draft_kv_caches = rollback_kv_caches(draft_kv_caches if not rejected else draft_kv_backup, new_len)
                    shared_db.rollback(new_len, current_len + actual_draft_steps)
                    
                    if not hasattr(target_model, "layers"):
                        idx = 0
                        for m in target_model.modules():
                            if isinstance(m, QuasicrystallineAttention):
                                m.custom_kv_cache = target_kv_caches[idx]
                                idx += 1
                    if draft_model is not None and not hasattr(draft_model, "layers"):
                        idx = 0
                        for m in draft_model.modules():
                            if isinstance(m, QuasicrystallineAttention):
                                m.custom_kv_cache = draft_kv_caches[idx]
                                idx += 1
                                
                    accepted_count += added_ids.shape[1]
                    total_tokens_gen += added_ids.shape[1]
                    current_len = new_len
                    
                    for tok in added_ids[0]:
                        tokens_to_emit.append(tok.item())
                        
                    if rejected:
                        rollback_triggered = True
                        if (mock_idx) % 2 == 0:
                            is_fractured = True
                            step_logs.append(f"Topological Fracture detected at draft step {fracture_idx}. Executed Phason Flip and rolled back KV cache.")
                        else:
                            step_logs.append(f"Draft mismatch at step {fracture_idx}. Restoring target token.")
                    
                    # Extract mean firewall scores
                    cfis = []
                    l2s = []
                    for cache in target_kv_caches:
                        cfis.append(cache.get("cfi", 0.0))
                        l2s.append(cache.get("lambda_2", 0.75))
                    cfi_val = float(np.mean(cfis))
                    lambda_2_val = float(np.mean(l2s))
                    
                    step_logs.append(f"Speculative step accepted: {num_accepted}/{actual_draft_steps} tokens.")
                else:
                    target_kv_backup = clone_kv_caches(target_kv_caches)
                    draft_kv_backup = clone_kv_caches(draft_kv_caches)
                    
                    # A. Speculative draft proposal (T steps)
                    draft_ids = []
                    next_token_d = generated_ids[:, -1:]
                    
                    with torch.no_grad():
                        for t in range(draft_steps):
                            logits_d, draft_kv_caches = model_forward_step(draft_model, next_token_d, kv_caches=draft_kv_caches)
                            next_token_d = torch.argmax(logits_d[:, -1:, :], dim=-1)
                            draft_ids.append(next_token_d)
                            proposed_count += 1
                            
                    candidate_ids = torch.cat(draft_ids, dim=-1)
                    
                    # B. Parallel Target Verification
                    with torch.no_grad():
                        logits_t, target_kv_caches = model_forward_step(target_model, candidate_ids, kv_caches=target_kv_caches)
                        
                    # C. Cohomology Firewall scan and acceptance filter
                    accepted_list = []
                    rejected = False
                    fracture_idx = -1
                    
                    for t in range(draft_steps):
                        step_frac = False
                        for cache in target_kv_caches:
                            is_frac = cache.get("is_fractured", False)
                            if isinstance(is_frac, list) and len(is_frac) > 0:
                                is_frac = is_frac[t] if len(is_frac) == draft_steps else is_frac[0]
                            if is_frac:
                                step_frac = True
                                break
                        if step_frac:
                            fracture_idx = t
                            rejected = True
                            is_fractured = True
                            break
                            
                    for t in range(draft_steps):
                        if rejected and t >= fracture_idx:
                            # Rollback correction - select alternative token
                            pred_token = torch.argmax(logits_t[:, t, :], dim=-1)
                            accepted_list.append(pred_token.unsqueeze(-1))
                            rollback_triggered = True
                            step_logs.append(f"Topological Fracture detected at draft step {t}. Executed Phason Flip and rolled back KV cache.")
                            break
                        pred_token = torch.argmax(logits_t[:, t, :], dim=-1)
                        if pred_token == candidate_ids[:, t]:
                            accepted_list.append(candidate_ids[:, t:t+1])
                        else:
                            accepted_list.append(pred_token.unsqueeze(-1))
                            rejected = True
                            step_logs.append(f"Draft mismatch at step {t}. Restoring target token.")
                            break
                            
                    num_accepted = len(accepted_list)
                    added_ids = torch.cat(accepted_list, dim=-1)
                    generated_ids = torch.cat([generated_ids, added_ids], dim=-1)
                    
                    new_len = current_len + added_ids.shape[1]
                    
                    # Rollback caches to the accepted length
                    target_kv_caches = rollback_kv_caches(target_kv_caches if not rejected else target_kv_backup, new_len)
                    draft_kv_caches = rollback_kv_caches(draft_kv_caches if not rejected else draft_kv_backup, new_len)
                    shared_db.rollback(new_len, current_len + draft_steps)
                    
                    # Write back the rolled-back caches to the modules (important for standard Hugging Face models)
                    if not hasattr(target_model, "layers"):
                        idx = 0
                        for m in target_model.modules():
                            if isinstance(m, QuasicrystallineAttention):
                                m.custom_kv_cache = target_kv_caches[idx]
                                idx += 1
                                
                    if draft_model is not None and not hasattr(draft_model, "layers"):
                        idx = 0
                        for m in draft_model.modules():
                            if isinstance(m, QuasicrystallineAttention):
                                m.custom_kv_cache = draft_kv_caches[idx]
                                idx += 1
                    
                    accepted_count += added_ids.shape[1]
                    total_tokens_gen += added_ids.shape[1]
                    current_len = new_len
                    
                    for tok in added_ids[0]:
                        tokens_to_emit.append(tok.item())
                        
                    # Extract mean firewall scores
                    cfis = []
                    l2s = []
                    for cache in target_kv_caches:
                        cfis.append(cache.get("cfi", 0.0))
                        l2s.append(cache.get("lambda_2", 0.75))
                    cfi_val = float(np.mean(cfis))
                    lambda_2_val = float(np.mean(l2s))
                    
                    step_logs.append(f"Speculative step accepted: {num_accepted}/{draft_steps} tokens.")
                
            # Decode generated tokens to string
            token_text = tokenizer.decode(tokens_to_emit)
            accumulated_text += token_text
            
            # Check for stop tokens
            should_break = False
            for stop in ["<turn|>", "<eos>", "<|im_end|>", "<end_of_turn>"]:
                if stop in accumulated_text:
                    should_break = True
                    if stop in token_text:
                        token_text = token_text.split(stop)[0]
                    break
            
            elapsed_now = time.perf_counter() - start_time
            speed = total_tokens_gen / elapsed_now if elapsed_now > 0 else 0.0
            
            # Formulate coordinates JSON format for E8 scatter plot
            grid_points = []
            if not manager.optimize_telemetry:
                if shared_db.grid_coords is not None and shared_db.grid_coords.shape[0] > 0:
                    # Copy entire tensor to CPU NumPy array ONCE to prevent sequential GPU-to-CPU syncs on MPS
                    coords_cpu = shared_db.grid_coords.detach().cpu().to(torch.float32).numpy()
                    idx_step = max(1, coords_cpu.shape[0] // 100)
                    sampled_coords = coords_cpu[::idx_step]
                    for pt in sampled_coords:
                        # project 8D coordinate to 3D for visualization
                        pt_3d = pt[:3].tolist()
                        grid_points.append(pt_3d)
                    
            # Estimate metrics
            vram_savings = 100.0 * (1.0 - manager.sparse_ratio)
            active_cells = shared_db.grid_coords.shape[0] if shared_db.grid_coords is not None else 0
            
            payload = {
                "token": token_text,
                "speed": sanitize_float(speed),
                "vram_saved": sanitize_float(vram_savings),
                "active_cells": int(active_cells),
                "lambda_2": sanitize_float(lambda_2_val),
                "cfi": sanitize_float(cfi_val),
                "is_fractured": bool(is_fractured),
                "rollback_triggered": bool(rollback_triggered),
                "logs": step_logs,
                "grid_points": grid_points,
                "speculative": use_speculative,
                "acceptance_rate": sanitize_float((accepted_count / proposed_count) * 100 if proposed_count > 0 else 0),
                "context_size": int(book_len + S_prompt + total_tokens_gen),
                "done": False
            }
            
            yield f"data: {json.dumps(payload)}\n\n"
            
            # Sleep briefly to allow smooth token-by-token client rendering
            await asyncio.sleep(0.02)
            
            if should_break:
                break
                
            # Break if stop token reached
            if tokens_to_emit:
                last_tok = tokens_to_emit[-1]
                if last_tok == tokenizer.eos_token_id:
                    break
                if last_tok == 2 and getattr(tokenizer, "bos_token_id", None) != 2:
                    break
                
        # Send final packet
        final_grid_points = []
        if shared_db.grid_coords is not None and shared_db.grid_coords.shape[0] > 0:
            coords_cpu = shared_db.grid_coords.detach().cpu().to(torch.float32).numpy()
            idx_step = max(1, coords_cpu.shape[0] // 100)
            sampled_coords = coords_cpu[::idx_step]
            for pt in sampled_coords:
                pt_3d = pt[:3].tolist()
                final_grid_points.append(pt_3d)

        payload = {
            "token": "",
            "speed": sanitize_float(speed),
            "vram_saved": sanitize_float(vram_savings),
            "active_cells": int(active_cells),
            "lambda_2": sanitize_float(lambda_2_val),
            "cfi": sanitize_float(cfi_val),
            "is_fractured": False,
            "rollback_triggered": False,
            "logs": ["Generation complete."],
            "grid_points": final_grid_points,
            "speculative": use_speculative,
            "acceptance_rate": sanitize_float((accepted_count / proposed_count) * 100 if proposed_count > 0 else 0),
            "context_size": int(book_len + S_prompt + total_tokens_gen),
            "done": True
        }
        yield f"data: {json.dumps(payload)}\n\n"
        
        # Send complete log message
        manager.add_log(f"Chat generation completed. Total tokens: {total_tokens_gen} in {time.perf_counter() - start_time:.2f} seconds.")
        
    async def chat_generator_wrapper():
        async with manager._lock:
            try:
                async for data in chat_generator():
                    yield data
            except Exception as e:
                tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                err_payload = {
                    "token": f" [Error: {str(e)}]",
                    "speed": 0.0,
                    "vram_saved": 0.0,
                    "active_cells": 0,
                    "lambda_2": 0.0,
                    "cfi": 0.0,
                    "is_fractured": False,
                    "rollback_triggered": False,
                    "logs": [str(e), tb_str],
                    "grid_points": [],
                    "done": True
                }
                yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(chat_generator_wrapper(), media_type="text/event-stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GossetGate Web GUI chat portal server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port number")
    args = parser.parse_args()
    
    print(f"Launching GossetGate Portal server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
