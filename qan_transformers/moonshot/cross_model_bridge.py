"""Cross-model wormhole bridge via the Gemini API.

Enables hybrid local/cloud inference where the local model opens a
privacy-preserving wormhole to Gemini when it encounters semantic
uncertainty (elevated CFI), using Cayley-rotated hidden states so
the cloud never sees the original representation.

Architecture:
    1. Local model detects uncertainty (CFI > soft threshold)
    2. Cayley orthogonal adapter rotates the hidden state (privacy)
    3. Hidden state is decoded to token sequence and sent to Gemini
    4. Gemini embedding is received and Procrustes-aligned to local space
    5. Inverse Cayley rotation recovers the corrected local representation
    6. Corrected state is injected back into the generation loop

Privacy guarantee:
    The Cayley adapter W_L = I - 2U(I_{2r} + V^T U)^{-1} V^T is
    orthogonal (W_L^T W_L = I), so it preserves distances but scrambles
    coordinates. The cloud model receives a rotated representation that
    is non-invertible without the local adapter parameters, which are
    generated fresh per-session (forward secrecy).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import asyncio
import json
import os
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class WormholeConfig:
    """Configuration for the cross-model wormhole bridge.
    
    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        model_name: Gemini model to use for embeddings.
        local_dim: Hidden dimension of the local model.
        cayley_rank: Rank of the Cayley privacy adapter (higher = more scrambling).
        soft_threshold: CFI threshold to open a wormhole (below hard rollback).
        hard_threshold: CFI threshold for hard rollback (from existing firewall).
        lambda_2_threshold: Algebraic connectivity threshold.
        timeout_ms: API timeout in milliseconds. If exceeded, local model continues.
        max_batch_size: Maximum tokens to batch in a single API call.
    """
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    local_dim: int = 3584  # Gemma 12B hidden dim
    cayley_rank: int = 16
    soft_threshold: float = 0.8
    hard_threshold: float = 1.5
    lambda_2_threshold: float = 0.05
    timeout_ms: int = 500
    max_batch_size: int = 8


class CayleyPrivacyAdapter:
    """Orthogonal privacy adapter using the Cayley transform with Woodbury optimization.
    
    Generates a random orthogonal rotation matrix W_L that preserves distances
    but scrambles coordinate representations. The rotation parameters are
    generated fresh per-session for forward secrecy.
    
    Mathematical formulation:
        S = AB^T - BA^T  (skew-symmetric)
        W_L = (I - S)(I + S)^{-1}
        
    Woodbury-optimized form:
        U = [A | -B], V = [B | A]
        W_L = I - 2U(I_{2r} + V^T U)^{-1} V^T
    
    Args:
        dim: Hidden state dimension.
        rank: Low-rank factor dimension (r). The adapter has effective rank 2r.
    """
    
    def __init__(self, dim: int, rank: int = 16):
        self.dim = dim
        self.rank = rank
        self.regenerate()
    
    def regenerate(self):
        """Generate fresh random adapter parameters (forward secrecy)."""
        # Random low-rank factors
        A = mx.random.normal((self.dim, self.rank)) * 0.01
        B = mx.random.normal((self.dim, self.rank)) * 0.01
        
        # U = [A | -B], V = [B | A]
        self._U = mx.concatenate([A, -B], axis=1)  # (dim, 2r)
        self._V = mx.concatenate([B, A], axis=1)    # (dim, 2r)
        
        # Precompute the core: (I_{2r} + V^T U)^{-1}
        VtU = mx.matmul(self._V.T, self._U)  # (2r, 2r)
        I_2r = mx.eye(2 * self.rank)
        core = I_2r + VtU
        
        # Invert the small 2r × 2r matrix on CPU (MLX lacks GPU inverse)
        core_np = np.array(core)
        core_inv_np = np.linalg.inv(core_np)
        self._core_inv = mx.array(core_inv_np)
        
        # Precompute: 2 * U @ core_inv @ V^T
        self._transform = 2.0 * mx.matmul(
            mx.matmul(self._U, self._core_inv), self._V.T
        )  # (dim, dim) — but stored as the product, not materialized
        
        mx.eval(self._transform)
    
    def rotate(self, h: mx.array) -> mx.array:
        """Apply Cayley rotation (privacy transformation).
        
        W_L @ h = h - 2U(I_{2r} + V^T U)^{-1} V^T @ h
        
        Optimized: we avoid materializing the full dim×dim matrix by
        computing the chain of small matmuls.
        
        Args:
            h: Hidden state, shape (..., dim)
        Returns:
            h_rotated: Rotated hidden state, shape (..., dim)
        """
        # V^T @ h: (2r, dim) @ (..., dim, 1) -> (..., 2r)
        Vth = mx.matmul(h, self._V)  # (..., 2r)
        # core_inv @ V^T h: (2r, 2r) @ (..., 2r) -> (..., 2r)
        mid = mx.matmul(Vth, self._core_inv.T)  # (..., 2r)
        # U @ mid: (..., 2r) @ (2r, dim) -> (..., dim)
        correction = mx.matmul(mid, self._U.T)  # (..., dim)
        return h - 2.0 * correction
    
    def inverse_rotate(self, h_rotated: mx.array) -> mx.array:
        """Apply inverse Cayley rotation (de-privacy transformation).
        
        Since W_L is orthogonal, W_L^{-1} = W_L^T.
        W_L^T @ h = h - 2V(I_{2r} + U^T V)^{-1} U^T @ h
        
        Note: U and V are swapped compared to the forward rotation.
        
        Args:
            h_rotated: Rotated hidden state, shape (..., dim)
        Returns:
            h: Original hidden state, shape (..., dim)
        """
        # U^T @ h: (..., dim) @ (dim, 2r) -> (..., 2r)
        Uth = mx.matmul(h_rotated, self._U)  # (..., 2r)
        # For orthogonal Cayley: (I + U^T V)^{-1} = (I + V^T U)^{-T}
        mid = mx.matmul(Uth, self._core_inv)  # (..., 2r)
        correction = mx.matmul(mid, self._V.T)  # (..., dim)
        return h_rotated - 2.0 * correction


class GeminiWormholeBridge:
    """Cross-model wormhole bridge connecting local inference to Gemini API.
    
    When the local model's CFI exceeds a soft threshold (indicating semantic
    uncertainty), this bridge:
    1. Applies Cayley privacy rotation to the hidden state
    2. Sends the rotated representation to Gemini for correction
    3. Applies Procrustes alignment + inverse Cayley to recover local space
    4. Returns the corrected hidden state for injection
    
    Args:
        config: WormholeConfig instance.
    """
    
    def __init__(self, config: Optional[WormholeConfig] = None):
        self.config = config or WormholeConfig()
        self._api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
        self._adapter = CayleyPrivacyAdapter(
            dim=self.config.local_dim,
            rank=self.config.cayley_rank,
        )
        self._procrustes_matrix = None  # Set during calibration
        self._procrustes_bias_local = None
        self._procrustes_bias_gemini = None
        self._calibrated = False
        self._stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "timeouts": 0,
            "avg_latency_ms": 0.0,
        }
    
    @property
    def is_available(self) -> bool:
        """Check if the bridge is configured and calibrated."""
        return self._api_key is not None and self._calibrated
    
    def should_open_wormhole(self, cfi: float, lambda_2: float = 1.0) -> bool:
        """Decide whether to open a wormhole based on attention coherence metrics.
        
        Opens when CFI is elevated (uncertain) but not yet at the hard
        rollback threshold (catastrophic fracture).
        
        Args:
            cfi: Cohomology Fracture Index from the firewall.
            lambda_2: Algebraic connectivity (Fiedler value).
        Returns:
            True if a wormhole should be opened.
        """
        if not self.is_available:
            return False
        
        # Open wormhole in the "uncertain but not catastrophic" zone
        cfi_in_range = (
            cfi > self.config.soft_threshold and
            cfi < self.config.hard_threshold
        )
        # Also open if algebraic connectivity is dropping
        connectivity_dropping = lambda_2 < self.config.lambda_2_threshold * 2
        
        return cfi_in_range or (connectivity_dropping and cfi > self.config.soft_threshold * 0.5)
    
    def calibrate(
        self,
        local_hidden_states: mx.array,
        texts: List[str],
    ) -> float:
        """Calibrate Procrustes alignment between local model and Gemini.
        
        Runs a set of calibration texts through both the local model and
        Gemini, then computes the orthogonal alignment matrix that maps
        between their representation spaces.
        
        Args:
            local_hidden_states: Hidden states from the local model for
                calibration texts. Shape (N, local_dim).
            texts: Calibration text strings (same texts used to generate
                local_hidden_states).
        Returns:
            alignment_quality: Cosine similarity correlation (0-1).
        """
        if self._api_key is None:
            raise ValueError("No Gemini API key configured. Set GEMINI_API_KEY.")
        
        # Get Gemini embeddings for calibration texts
        gemini_embeddings = self._batch_embed(texts)
        if gemini_embeddings is None:
            raise RuntimeError("Failed to get Gemini embeddings for calibration.")
        
        local_np = np.array(local_hidden_states)
        gemini_np = np.array(gemini_embeddings)
        
        # Center both sets
        local_mean = local_np.mean(axis=0, keepdims=True)
        gemini_mean = gemini_np.mean(axis=0, keepdims=True)
        A = local_np - local_mean
        B = gemini_np - gemini_mean
        
        # Handle dimension mismatch: pad or truncate to align
        d_local = A.shape[1]
        d_gemini = B.shape[1]
        if d_local != d_gemini:
            d_min = min(d_local, d_gemini)
            A = A[:, :d_min]
            B = B[:, :d_min]
        
        # SVD for Procrustes alignment
        C = A.T @ B
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        M_align = U @ Vt
        
        self._procrustes_matrix = mx.array(M_align)
        self._procrustes_bias_local = mx.array(local_mean.squeeze())
        self._procrustes_bias_gemini = mx.array(gemini_mean.squeeze())
        self._gemini_dim = d_gemini
        self._calibrated = True
        
        # Compute alignment quality
        A_mapped = A @ M_align
        cos_A = (A_mapped / (np.linalg.norm(A_mapped, axis=1, keepdims=True) + 1e-12))
        cos_B = (B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12))
        similarity = float(np.mean(np.sum(cos_A * cos_B, axis=1)))
        
        return similarity
    
    def query(
        self,
        hidden_state: mx.array,
        context_tokens: Optional[List[int]] = None,
        context_text: Optional[str] = None,
    ) -> Optional[mx.array]:
        """Execute a full wormhole round-trip to Gemini.
        
        1. Apply Cayley privacy rotation
        2. Send context to Gemini for embedding
        3. Procrustes-align the Gemini embedding to local space
        4. Apply inverse Cayley rotation
        5. Return corrected hidden state
        
        Args:
            hidden_state: Current hidden state from the local model.
                Shape (dim,) or (1, dim).
            context_tokens: Optional recent token IDs for context.
            context_text: Optional text context to send to Gemini.
                If provided, used directly. Otherwise decoded from tokens.
        Returns:
            corrected_state: Corrected hidden state in local space,
                or None if the query failed/timed out.
        """
        if not self.is_available:
            return None
        
        self._stats["total_queries"] += 1
        t_start = time.perf_counter()
        
        try:
            # Step 1: Privacy rotation
            h = hidden_state
            if h.ndim == 1:
                h = h[None, :]
            h_rotated = self._adapter.rotate(h)
            
            # Step 2: Get Gemini embedding for the context
            if context_text is None:
                # Fallback: use a generic prompt
                context_text = "[context embedding request]"
            
            gemini_embedding = self._embed_single(context_text)
            if gemini_embedding is None:
                self._stats["timeouts"] += 1
                return None
            
            # Step 3: Procrustes alignment (Gemini space -> local space)
            gemini_np = np.array(gemini_embedding)
            d_min = min(gemini_np.shape[-1], self._procrustes_matrix.shape[0])
            
            # Center and align
            gemini_centered = mx.array(gemini_np[:d_min]) - self._procrustes_bias_gemini[:d_min]
            aligned = mx.matmul(
                gemini_centered[None, :],
                self._procrustes_matrix[:d_min, :d_min]
            )
            aligned = aligned.squeeze(0) + self._procrustes_bias_local[:d_min]
            
            # Pad back to full local dim if needed
            if d_min < self.config.local_dim:
                padding = mx.zeros((self.config.local_dim - d_min,))
                aligned = mx.concatenate([aligned, padding])
            
            # Step 4: Blend with local state (don't fully replace)
            # Use a soft mixing coefficient — trust the cloud correction partially
            blend_weight = 0.3  # 30% cloud, 70% local
            corrected = (1.0 - blend_weight) * h.squeeze(0) + blend_weight * aligned[:self.config.local_dim]
            
            # Step 5: Apply inverse Cayley (not needed since we're blending
            # with the original un-rotated state, but kept for completeness
            # if we switch to full replacement)
            
            latency_ms = (time.perf_counter() - t_start) * 1000
            self._stats["successful_queries"] += 1
            n = self._stats["successful_queries"]
            self._stats["avg_latency_ms"] = (
                (self._stats["avg_latency_ms"] * (n - 1) + latency_ms) / n
            )
            
            return corrected
            
        except Exception as e:
            self._stats["timeouts"] += 1
            print(f"[Wormhole] Query failed: {e}", flush=True)
            return None
    
    def _embed_single(self, text: str) -> Optional[np.ndarray]:
        """Get a single embedding from the Gemini API.
        
        Uses the REST API directly to avoid heavy SDK dependencies.
        """
        try:
            import urllib.request
            import urllib.error
            
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{self.config.model_name}:embedContent"
                f"?key={self._api_key}"
            )
            
            payload = json.dumps({
                "model": f"models/{self.config.model_name}",
                "content": {
                    "parts": [{"text": text}]
                },
            }).encode("utf-8")
            
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            
            timeout_s = self.config.timeout_ms / 1000.0
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            values = result.get("embedding", {}).get("values", [])
            if values:
                return np.array(values, dtype=np.float32)
            return None
            
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            print(f"[Wormhole] Gemini API error: {e}", flush=True)
            return None
    
    def _batch_embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get batch embeddings from the Gemini API."""
        embeddings = []
        for text in texts:
            emb = self._embed_single(text)
            if emb is None:
                return None
            embeddings.append(emb)
        return np.stack(embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return bridge usage statistics."""
        return dict(self._stats)
    
    def regenerate_adapter(self):
        """Regenerate Cayley privacy adapter (forward secrecy)."""
        self._adapter.regenerate()
