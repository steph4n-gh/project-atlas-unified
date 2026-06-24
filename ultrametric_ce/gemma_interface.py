"""Gemma interface for quantized loading (via mlx_lm) and embedding/logit extraction.

This module supports the tree induction (Task 4) and later distillation phases.
It is deliberately isolated: `import ultrametric_ce.gemma_interface` succeeds
without mlx_lm installed. The dependency is only resolved inside load_gemma()
(and related), which raises a clear actionable ImportError.

Prerequisites / notes:
- Real usage requires `pip install mlx-lm` (already in the dev venv for this worktree).
- Gemma weights are NOT shipped in this repository. User must supply a path
  or HF repo id loadable by mlx_lm (e.g. local converted dir, or
  'mlx-community/gemma-2-2b-4bit', 'mlx-community/gemma-2-9b-4bit', etc.).
- Quantized loading: choose a pre-quantized mlx-community repo or run conversion
  yourself; mlx_lm.load() will load the safetensors directly in 4-bit etc.
- For tests: everything is mockable via duck-typed model/tokenizer objects
  (see test_gemma_* in tests/test_tree.py). No real weights or mlx_lm needed
  for unit tests of extraction helpers.
- Embeddings for tree induction are taken from model.model.embed_tokens.weight
  (the tied input embeddings; also used for unembedding via as_linear).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

import mlx.core as mx

__all__ = [
    "GemmaInterface",
    "load_gemma",
    "load_gemma_transformers",
    "get_embeddings",
    "get_logits",
    "extract_embeddings_from_mlx_snapshot",
    "find_local_gemma_on_storage",
    "load_gemma_tokenizer",
]


def _ensure_mlx_lm():
    """Internal helper: lazy import mlx_lm or raise clear error."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mlx_lm is not installed. Real Gemma loading (quantized or otherwise) "
            "requires the mlx-lm package.\n"
            "Install with: pip install mlx-lm\n"
            "Then provide a Gemma model via --gemma-model or direct path/HF id "
            "(e.g. 'mlx-community/gemma-2-2b-4bit').\n"
            "Gemma weights are not included in this repo; the user must supply them.\n"
            "For unit tests / synthetic tree building, use mocks or the synthetic "
            "path in scripts/build_tree_from_gemma.py (no Gemma required)."
        ) from exc


def _ensure_transformers():
    """Internal helper: lazy import transformers+torch or raise clear error (for gemma4 native arch)."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for loading Gemma-4 models via transformers backend "
            "(native arch support for gemma-4-E* / 12B etc in dev versions).\n"
            "In this env they should be present (dev install). If missing: pip install transformers torch\n"
            "Provide e.g. --gemma-model google/gemma-4-E2B-it (or larger). These load to the active "
            "HF cache (set HF_HOME=/Volumes/Storage/huggingface_cache to use the storage drive cache)."
        ) from exc


class GemmaInterface:
    """Light wrapper around a loaded model (mlx_lm or transformers).

    Supports:
      - mlx_lm quantized loads (for compatible gemma2 etc)
      - transformers loads for gemma-4 series (google/gemma-4-E* etc) which often fail mlx_lm
        due to extra attn params (k_norm etc) in snapshots; uses native arch in dev transformers.
      - direct snapshot dequant for mlx-community 4bit gemma4 embed extraction (storage cache).

    Exposes helpers for:
      - get_embeddings(token_ids=None) -> np.ndarray of shape (V|k, D)
      - get_logits(input_ids) -> np.ndarray of last-token logits (V,)

    The wrapper works with real loaded models and with pure-Python mocks
    for testing (duck typing on .model.embed_tokens.weight and callable model).
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """Store the loaded (or mocked) model and tokenizer.

        No heavy validation here; real shape checks happen on first use.
        Auto-detects backend for get_*/forward dispatch (mlx vs transformers/torch).
        """
        self.model = model
        self.tokenizer = tokenizer
        # backend hint for dispatch (mlx default, or "transformers" for torch models)
        tname = str(type(model))
        self._backend = "transformers" if ("transformers" in tname or hasattr(model, "config")) else "mlx"

    @property
    def vocab_size(self) -> int:
        """Best-effort vocab size (from args or tokenizer or embed weight)."""
        m = self.model
        if hasattr(m, "args") and hasattr(m.args, "vocab_size"):
            return int(m.args.vocab_size)
        if hasattr(m, "model") and hasattr(m.model, "args") and hasattr(m.model.args, "vocab_size"):
            return int(m.model.args.vocab_size)
        # fallback via weight
        try:
            w = self._get_embed_weight()
            return int(w.shape[0])
        except Exception:
            # last resort
            if hasattr(self.tokenizer, "vocab_size"):
                return int(self.tokenizer.vocab_size)
            return 0

    def _get_embed_weight(self) -> Any:
        """Locate the embed_tokens weight, tolerating outer/inner model + transformers."""
        m = self.model
        # transformers path (google gemma-4 etc)
        if self._backend == "transformers" or hasattr(m, "get_input_embeddings"):
            try:
                emb = m.get_input_embeddings()
                if hasattr(emb, "weight"):
                    return emb.weight
            except Exception:
                pass
            for subname in ("model", "language_model"):
                sub = getattr(m, subname, None)
                if sub is not None:
                    if hasattr(sub, "get_input_embeddings"):
                        try:
                            return sub.get_input_embeddings().weight
                        except Exception:
                            pass
                    if hasattr(sub, "embed_tokens") and hasattr(sub.embed_tokens, "weight"):
                        return sub.embed_tokens.weight
            # direct on top
            if hasattr(m, "embed_tokens") and hasattr(m.embed_tokens, "weight"):
                return m.embed_tokens.weight
        # Common structure: loaded = (Model(), tok) where Model has .model = GemmaModel (mlx)
        candidate = None
        if hasattr(m, "model") and hasattr(m.model, "embed_tokens"):
            candidate = m.model.embed_tokens
        elif hasattr(m, "embed_tokens"):
            candidate = m.embed_tokens
        else:
            # try deeper or direct
            candidate = getattr(getattr(m, "model", m), "embed_tokens", None)
        if candidate is None or not hasattr(candidate, "weight"):
            raise AttributeError(
                "Could not locate embed_tokens.weight on the provided model. "
                "Expected structure after mlx_lm.load: model.model.embed_tokens.weight "
                "(or model.embed_tokens.weight for some wrappers). "
                "For transformers: get_input_embeddings().weight or language_model.model.embed_tokens."
            )
        return candidate.weight

    def get_embeddings(
        self, token_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        """Return token embeddings as numpy array.

        If token_ids is None: returns full (vocab_size, hidden_dim) matrix
        (the order matches tokenizer ids 0..V-1).

        If token_ids given: returns selected rows in the provided order.

        Always returns a CPU numpy float32 array (materialized).
        Works for both float and quantized loads (np.array forces dequant/eval).
        """
        w = self._get_embed_weight()
        # Materialize (important for lazy or quantized views)
        if hasattr(w, "detach"):  # torch.Tensor
            arr = w.detach().cpu().numpy().astype(np.float32, copy=True)
        else:
            if hasattr(w, "eval"):
                w.eval()
            arr = np.array(w, dtype=np.float32, copy=True)
        if token_ids is not None:
            if not token_ids:
                return np.empty((0, arr.shape[1]), dtype=np.float32)
            idx = np.asarray(token_ids, dtype=np.int64)
            # clip? no, let advanced index error if bad
            arr = arr[idx]
        return arr

    def get_logits(
        self, input_ids: Union[List[int], mx.array, np.ndarray]
    ) -> np.ndarray:
        """Run a forward pass and return the *last position* logits for next-token.

        input_ids: list[int] (preferred for simplicity), or mx.array / np of shape
        (seq_len,) or (1, seq_len). Batch >1 not supported in this helper (MVP);
        only the final token position of the (first) sequence is returned.

        Returns: np.ndarray of shape (vocab_size,) float32, ready for argmax/softmax.
        Supports both mlx (uint) and transformers/torch (long) models.
        """
        if self._backend == "transformers" or hasattr(self.model, "config"):
            import torch
            if isinstance(input_ids, (list, tuple)):
                inp = torch.tensor([input_ids], dtype=torch.long)
            elif isinstance(input_ids, np.ndarray):
                inp = torch.from_numpy(input_ids.astype(np.int64))
                if inp.ndim == 1:
                    inp = inp.unsqueeze(0)
            else:
                # torch or other
                inp = input_ids
                if hasattr(inp, "ndim") and inp.ndim == 1:
                    inp = inp.unsqueeze(0)
            with torch.no_grad():
                out = self.model(input_ids=inp)
            # standard causal lm output
            if hasattr(out, "logits"):
                last = out.logits[0, -1, :].detach().cpu().float().numpy()
            else:
                # tolerate raw tensor returns in some wrappers
                logits_t = out
                if hasattr(logits_t, "detach"):
                    last = logits_t[0, -1, :].detach().cpu().float().numpy()
                else:
                    last = np.asarray(logits_t).reshape(-1)[-self.vocab_size or 0 :]
            return last.astype(np.float32, copy=False)

        # --- mlx path (original) ---
        if isinstance(input_ids, (list, tuple)):
            # Left-pad to next power of 2 to collapse JIT compile shapes
            n = len(input_ids)
            if n > 0:
                next_pow2 = 1 << (n - 1).bit_length()
                next_pow2 = max(next_pow2, 2)  # minimum shape
                pad_len = next_pow2 - n
                if pad_len > 0:
                    # Pad with token 0 (typically <pad> or <bos>)
                    input_ids = [0] * pad_len + list(input_ids)
            inp = mx.array([input_ids], dtype=mx.uint32)
        elif isinstance(input_ids, np.ndarray):
            inp = mx.array(input_ids)
            if inp.ndim == 1:
                inp = inp[None, :]
        else:
            # assume mx.array
            inp = input_ids
            if inp.ndim == 1:
                inp = inp[None, :]

        # Call the model. For Gemma the __call__ on the top-level Model returns logits.
        # Some wrappers may need model.model(...); we try top level first.
        try:
            out = self.model(inp)
        except Exception:
            # fallback for inner-model style mocks / wrappers
            inner = getattr(self.model, "model", self.model)
            out = inner(inp)

        # out expected (B, S, V)
        if hasattr(out, "eval"):
            out.eval()
        # Cast to float32 inside MLX to avoid PEP 3118 buffer size mismatch under Python 3.14
        if hasattr(out, "astype"):
            out_f32 = out.astype(mx.float32)
        else:
            out_f32 = out
        logits = np.array(out_f32)
        # take last position of first batch elem
        if logits.ndim != 3:
            # tolerate some shapes in mocks/tests
            last = logits.reshape(-1)[-self.vocab_size or logits.shape[-1]:]
        else:
            last = logits[0, -1, :]
        return last.astype(np.float32, copy=False)


def load_gemma(
    path_or_repo: str,
    *,
    tokenizer_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    revision: Optional[str] = None,
    backend: str = "auto",
    **extra,
) -> GemmaInterface:
    """Load a Gemma model (and tokenizer) via mlx_lm.load or transformers (for gemma-4).

    backend: "auto" (default), "mlx", "transformers".
      - auto: use transformers for google/gemma-4* ids (native arch); else mlx.
      - transformers: force torch/transformers path (needed for gemma-4-E*/12B that have
        extra attn params not in mlx_lm's Gemma impl; uses dev transformers gemma4 support).
      - mlx: force mlx_lm (good for gemma-2 4bit etc in storage cache).

    For mlx-community gemma-4 4bit snapshots in HF cache (e.g. on storage drive), full
    mlx load often fails with "Received N parameters not in model"; use backend=transformers
    with the google/ equivalent, or for *embed extraction only* use the direct
    extract_embeddings_from_mlx_snapshot on the local snapshot dir.

    The storage drive cache (multiple layouts under /Volumes/Storage/huggingface_cache
    plus scratch copies) is preferred: we auto-probe known roots for short repo ids
    (e.g. "mlx-community/gemma-4-E4B-4bit" or "google/gemma-4-E2B-it") and use the local
    snapshot/flat dir if present. Set HF_HOME=/Volumes/Storage/huggingface_cache (or
    pass a full path) to control/force it. "or you can download" will also land there.

    Raises:
        ImportError: with clear message if backend dep missing.
        Load errors propagate (user must have the weights / net access).

    Returns:
        GemmaInterface wrapping the (model, tokenizer).
    """
    # Prefer local copy on the storage drive if the user gave a short id
    resolved = _find_local_gemma_on_storage(path_or_repo)
    if resolved:
        path_or_repo = resolved

    # Robust dispatch: short "google/..." or resolved local google snapshot/flat path (bare "google-gemma-4-12B-it-4bit" etc. from storage /hub)
    pstr = str(path_or_repo).lower()
    is_google_gemma4 = (
        "google/gemma-4" in pstr
        or "google-gemma-4" in pstr
        or "models--google--gemma-4" in pstr
    )
    if backend == "transformers" or (backend == "auto" and is_google_gemma4):
        return load_gemma_transformers(
            path_or_repo,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            revision=revision,
            **extra,
        )

    _ensure_mlx_lm()
    from mlx_lm.utils import load_model as mlx_load_model, load_tokenizer as mlx_load_tokenizer

    load_kwargs: dict = {
        "tokenizer_config": tokenizer_config or {},
        "model_config": model_config or {},
        "adapter_path": adapter_path,
        "lazy": lazy,
        "revision": revision,
    }
    load_kwargs.update({k: v for k, v in extra.items() if v is not None})

    strict = load_kwargs.pop("strict", True)
    lazy = load_kwargs.get("lazy", False)
    model_config = load_kwargs.get("model_config", {})

    from pathlib import Path as LocalPath
    model, _ = mlx_load_model(LocalPath(path_or_repo), lazy=lazy, strict=strict, model_config=model_config)
    tokenizer = mlx_load_tokenizer(LocalPath(path_or_repo))


    return GemmaInterface(model, tokenizer)




# Convenience free functions (optional ergonomic use; delegate to interface)

def get_embeddings(
    path_or_repo: str,
    token_ids: Optional[List[int]] = None,
    **load_kwargs,
) -> np.ndarray:
    """One-shot: load Gemma and return (subset of) embeddings as np.ndarray.

    Primarily useful for scripts. For repeated access, use GemmaInterface directly.
    """
    iface = load_gemma(path_or_repo, **load_kwargs)
    return iface.get_embeddings(token_ids)


def get_logits(
    path_or_repo: str,
    input_ids: Union[List[int], mx.array],
    **load_kwargs,
) -> np.ndarray:
    """One-shot: load Gemma, run forward on input_ids, return last logits."""
    iface = load_gemma(path_or_repo, **load_kwargs)
    return iface.get_logits(input_ids)


# =============================================================================
# Direct / robust extract for mlx-community gemma-4 4bit snapshots (storage cache)
# and transformers loader (for gemma-4 teacher + full interface)
# =============================================================================

def _extract_embeddings_mlx_snapshot(
    snapshot_dir: str | Path, token_ids: Optional[List[int]] = None
) -> np.ndarray:
    """Extract float32 embed matrix directly from mlx-community *-4bit gemma4 snapshot dir.

    Uses mx.load + mx.dequantize on language_model.model.embed_tokens.{weight,scales,biases}.
    Bypasses mlx_lm.load (which fails for gemma-4 due to extra k_norm/k_proj.* in weights vs model class).
    This enables tree induction from real embeddings using the 4bit snapshots on storage drive
    (e.g. /Volumes/Storage/huggingface_cache/.../models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<hash>).

    Returns (V, D) float32 np (or sliced). Matches shapes from transformers google/gemma-4-E* .
    """
    snap = Path(snapshot_dir)
    # Support HF snapshot layout (models--.../snapshots/<hash>/model.safetensors)
    # and flat local/scratch dirs on storage (e.g. /Volumes/Storage/project_atlas/scratch/gemma-4-E2B-it-4bit/model.safetensors or the .safetensors itself)
    wpath = None
    if snap.is_file() and snap.suffix == ".safetensors":
        wpath = snap
    else:
        cand = snap / "model.safetensors"
        if cand.exists():
            wpath = cand
        else:
            # flat dir or other layout: look for a model*.safetensors at top level
            for f in list(snap.glob("model*.safetensors")):
                wpath = f
                break
    if wpath is None or not wpath.exists():
        raise FileNotFoundError(f"No model.safetensors (or model-*.safetensors) found under or as: {snap}. "
                                "Pass a HF snapshot dir, flat local 4bit dir (e.g. scratch gemma-4-E* on storage), "
                                "or the .safetensors file itself. Storage drive has models under "
                                "/Volumes/Storage/huggingface_cache/hub/... and /project_atlas/scratch/.")
    import mlx.core as mx_local  # local name to avoid shadowing

    # If sharded (index.json present), find the shard containing the embed key and load only that safetensors.
    # This supports larger 12B 4bit qat etc. that are sharded on the storage cache.
    index_path = snap / "model.safetensors.index.json" if snap.is_dir() else None
    if index_path and index_path.exists() and wpath.name != "model.safetensors":  # only if we didn't pick a specific shard
        try:
            import json as _json
            idx = _json.loads(index_path.read_text())
            wmap = idx.get("weight_map", {})
            shard_name = None
            for k in ["language_model.model.embed_tokens.weight", "model.embed_tokens.weight"]:
                if k in wmap:
                    shard_name = wmap[k]
                    break
            if shard_name:
                shard_path = snap / shard_name
                if shard_path.exists():
                    wpath = shard_path
                    print(f"[extract] using sharded {shard_name} for embed key")
        except Exception:
            pass  # fall back to the wpath we have

    weights = mx_local.load(str(wpath))
    # common key under gemma4 mlx snapshots
    candidates = [
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ]
    w = None
    skey = bkey = None
    for ck in candidates:
        if ck in weights:
            w = weights[ck]
            skey = ck.replace(".weight", ".scales")
            bkey = ck.replace(".weight", ".biases")
            break
    if w is None:
        # last resort scan
        for k in list(weights.keys()):
            if "embed_tokens" in k and k.endswith(".weight"):
                w = weights[k]
                skey = k.replace(".weight", ".scales")
                bkey = k.replace(".weight", ".biases")
                break
    if w is None:
        raise KeyError("Could not find embed_tokens.weight in snapshot safetensors")
    scales = weights.get(skey)
    biases = weights.get(bkey)

    if token_ids is not None:
        if not token_ids:
            # handle empty token_ids list immediately
            D = w.shape[1] * 8 if biases is not None else w.shape[1] # best effort
            return np.empty((0, D), dtype=np.float32)
        idx = mx_local.array(token_ids, dtype=mx_local.int64)
        w = w[idx]
        if scales is not None:
            scales = scales[idx]
        if biases is not None:
            biases = biases[idx]
        token_ids = None  # clear to avoid double slicing

    # dequant; group 64 / 4bit / affine is common for these; fall back to auto
    emb = None
    for gs in (64, 32, None):
        try:
            emb = mx_local.dequantize(
                w, scales, biases, group_size=gs, bits=4, mode="affine"
            )
            break
        except Exception:
            continue
    if emb is None:
        # fallback: if already float-ish (rare), use as-is after cast
        emb = w.astype(mx_local.float32) if hasattr(w, "astype") else w
    # dequant often yields bfloat16; astype float32 for reliable numpy buffer
    if hasattr(emb, "astype") and str(emb.dtype) != "float32":
        emb = emb.astype(mx_local.float32)
    arr = np.array(emb, dtype=np.float32, copy=True)
    if token_ids is not None:
        if not token_ids:
            return np.empty((0, arr.shape[1]), dtype=np.float32)
        idx = np.asarray(token_ids, dtype=np.int64)
        arr = arr[idx]
    return arr


def extract_embeddings_from_mlx_snapshot(
    snapshot_dir: str | Path, token_ids: Optional[List[int]] = None
) -> np.ndarray:
    """Public: extract embeddings from local mlx gemma4 4bit snapshot (for tree build).

    See _extract... for details. Safe to call even if mlx_lm full load would fail.
    """
    return _extract_embeddings_mlx_snapshot(snapshot_dir, token_ids)


# Common roots on this user's storage drive where gemma-4 weights live
# (multiple HF cache layouts + scratch copies from other work). We probe these
# to prefer local copies of "all the models" without re-download when user
# passes short repo ids like "mlx-community/gemma-4-E4B-4bit" or "google/gemma-4-E2B-it".
_STORAGE_GEMMA_ROOTS = [
    "/Volumes/Storage/huggingface_cache",  # catches direct models-- at cache root + children /hub /huggingface/hub
    "/Volumes/Storage/huggingface_cache/hub",
    "/Volumes/Storage/huggingface_cache/huggingface/hub",
    "/Volumes/Storage/project_atlas/scratch",
]


def _find_local_gemma_on_storage(model_id_or_path: str) -> str | None:
    """If the arg is a short HF id (org/name) or name, look for a matching local
    snapshot or flat dir under the known storage cache/scratch roots. Returns
    the best local path (latest snapshot dir or flat dir) or None.
    This lets scripts use e.g. --gemma-model mlx-community/gemma-4-E4B-4bit
    and automatically use the copy on the storage drive if present.
    """
    if not model_id_or_path or os.path.exists(model_id_or_path):
        return None if not os.path.exists(model_id_or_path) else model_id_or_path

    # Normalize short id to the HF "models--org--name" form used in cache dirs
    if "/" in model_id_or_path:
        org, name = model_id_or_path.split("/", 1)
        cache_name = f"models--{org}--{name}"
    else:
        cache_name = model_id_or_path  # e.g. a bare "gemma-4-E2B-it-4bit" in scratch

    for root in _STORAGE_GEMMA_ROOTS:
        base = Path(root) / cache_name
        if base.exists():
            if (base / "snapshots").exists():
                # HF layout - pick latest valid snapshot containing actual weights
                snaps = sorted((base / "snapshots").glob("*"))
                for snap in reversed(snaps):
                    # Check if there are weights files > 10MB
                    has_weights = False
                    try:
                        for f in snap.glob("*"):
                            if f.suffix in (".safetensors", ".bin") and f.stat().st_size > 10 * 1024 * 1024:
                                has_weights = True
                                break
                    except Exception:
                        pass
                    if has_weights:
                        return str(snap)
            else:
                # flat scratch/local dir
                return str(base)

        # Also try if the arg itself matches a flat name under scratch
        if root.endswith("scratch"):
            flat = Path(root) / model_id_or_path
            if flat.exists() and flat.is_dir():
                return str(flat)

    # Fallback glob under all roots for bare or non-models-- naming seen in storage
    # (e.g. "google-gemma-4-12B-it-4bit" flat sharded under /hub, or other variants, or direct under cache root)
    base = model_id_or_path.split("/")[-1] if "/" in model_id_or_path else model_id_or_path
    for root in _STORAGE_GEMMA_ROOTS:
        for cand in Path(root).glob(f"**/*{base}*"):
            if cand.is_dir() and "gemma-4" in str(cand).lower() and ".no_exist" not in str(cand) and ".locks" not in str(cand):
                if (cand / "snapshots").exists():
                    snaps = sorted((cand / "snapshots").glob("*"))
                    if snaps:
                        return str(snaps[-1])
                elif (cand / "config.json").exists() or list(cand.glob("*.safetensors*")):
                    return str(cand)

    return None


# Public alias (the _ version is used internally by load_gemma etc. for the auto storage preference)
find_local_gemma_on_storage = _find_local_gemma_on_storage


def load_gemma_tokenizer(
    path_or_repo: str,
    **kwargs,
) -> Any:
    """Load *only* the tokenizer for a Gemma (real or toy) without loading the full model weights.

    This enables text<->token roundtrips for UCE generation on real high-tid Gemma trees
    without paying the cost (or loading the attention model) of a full load_gemma just for encode/decode.
    Uses transformers AutoTokenizer (compatible with google/gemma-4-* and mlx-community gemma ids;
    tokenizers are the same). Local paths and storage-resolved bare dirs are supported.
    """
    _ensure_transformers()
    from transformers import AutoTokenizer
    # resolve to local storage path if short id matches something on drive
    resolved = _find_local_gemma_on_storage(path_or_repo) or path_or_repo
    tok = AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=True,
        **{k: v for k, v in kwargs.items() if v is not None},
    )
    return tok


def load_gemma_transformers(
    path_or_repo: str,
    *,
    dtype: Any = None,
    device_map: Optional[str] = "cpu",
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = True,
    **kwargs,
) -> GemmaInterface:
    """Load gemma-4 (or other) via transformers AutoModelForCausalLM (dev version supports gemma4).

    Preferred for google/gemma-4-E2B-it , gemma-4-12B-it etc when mlx snapshots have arch skew.
    Sets HF cache per env (HF_HOME) if user wants storage drive cache.
    Returns GemmaInterface (methods dispatch to torch path).
    """
    _ensure_transformers()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import accelerate
    except ImportError:
        if device_map in ("cpu", "auto"):
            device_map = None

    if dtype is None:
        dtype = torch.bfloat16

    # Clean up arguments that are specific to MLX model loader
    kwargs.pop("strict", None)
    kwargs.pop("lazy", None)
    kwargs.pop("model_config", None)
    kwargs.pop("tokenizer_config", None)
    kwargs.pop("adapter_path", None)

    model = AutoModelForCausalLM.from_pretrained(
        path_or_repo,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
        **{k: v for k, v in kwargs.items() if v is not None},
    )
    tokenizer = AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=trust_remote_code)
    iface = GemmaInterface(model, tokenizer)
    iface._backend = "transformers"
    return iface
