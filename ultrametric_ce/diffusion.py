"""Ultrametric diffusion skeleton (Vladimirov-style approx on finite p-adic tree).

Simple intra-ball weighted average + cross-level decay using p ** (delta_val * alpha).
Operates on dict of per-ball states: key=(depth, prefix) -> mx.array feature vector.
"""

from __future__ import annotations

from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

from ultrametric_ce.padic import valuation
from ultrametric_ce.floquet import FloquetScheduler, AdaptiveFloquetScheduler
from ultrametric_ce.wormhole_gate import LearnableWormholeGate


__all__ = ["UltrametricDiffusion"]


class UltrametricDiffusion(nn.Module):
    """Skeleton multi-scale diffusion / aggregation over tree balls.

    For toy: mixes states of balls at same depth with ultrametric weights,
    plus simple cross-level (parent) mixing. Then applies small linear per layer.
    """

    def __init__(
        self,
        p: int,
        depth: int,
        dim: int,
        num_layers: int = 1,
        alpha: float = 0.5,
        schwarzschild_warp: bool = True,
        r_s: float = 0.5,
        wormhole_gate: bool = True,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(p, int) or p < 2:
            raise ValueError("p must be int >=2")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be int >=1")
        self.p = p
        self.depth = depth
        self.dim = dim
        self.alpha = float(alpha)
        self.num_layers = int(num_layers)
        self.schwarzschild_warp = schwarzschild_warp
        self.r_s = r_s
        self.wormhole_gate = wormhole_gate
        self.epsilon = epsilon
        self.floquet = AdaptiveFloquetScheduler(omega_init=1.0) if wormhole_gate else None
        self.learned_gate = LearnableWormholeGate(dim=dim) if wormhole_gate else None

        # Per-layer linear mixer (real valued params). Use plain list (MLX tracks modules assigned to lists via __setattr__).
        self.mix_linears: list = []
        for _ in range(self.num_layers):
            self.mix_linears.append(nn.Linear(dim, dim))

    def forward(
        self, states: Dict[Tuple[int, int], mx.array]
    ) -> Dict[Tuple[int, int], mx.array]:
        """Mix active (and all pre-inited) ball states.

        Returns updated dict with same keys.
        """
        if not states:
            return {}

        current: Dict[Tuple[int, int], mx.array] = {k: v for k, v in states.items()}

        for layer_idx in range(self.num_layers):
            if not hasattr(self, "_W_cache"):
                self._W_cache = {}

            new_states: Dict[Tuple[int, int], mx.array] = {}

            # Group by depth for intra-ball (same-depth) mixing
            by_depth: Dict[int, list] = {}
            for (d, pref), vec in current.items():
                by_depth.setdefault(d, []).append((pref, vec))

            for d, items in by_depth.items():
                N = len(items)
                if N == 1:
                    pref, vec = items[0]
                    mixed = vec
                    if d > 0:
                        parent_pref = 0 if d == 1 else (pref % (self.p ** (d - 1)))
                        parent_key = (d - 1, parent_pref)
                        if parent_key in current:
                            cross_w = float(self.p ** (-1.0 * self.alpha))
                            mixed = mixed * (1.0 / (1.0 + cross_w)) + current[parent_key] * (cross_w / (1.0 + cross_w))
                    new_states[(d, pref)] = mixed
                    continue

                # Stack all vectors at this depth
                vecs = [item[1] for item in items]
                vecs_matrix = mx.stack(vecs)  # shape (N, dim)

                # Compute or retrieve base weight matrix
                import numpy as np
                prefs_tuple = tuple(item[0] for item in items)
                cache_key = (d, prefs_tuple)
                if cache_key not in self._W_cache:
                    W_np = np.zeros((N, N), dtype=np.float32)
                    for i in range(N):
                        pref_i = prefs_tuple[i]
                        for j in range(N):
                            pref_j = prefs_tuple[j]
                            if d == 0:
                                w = 1.0
                            else:
                                delta = int(pref_i) - int(pref_j)
                                v = valuation(delta, self.p)
                                if isinstance(v, float) and v == float("inf"):
                                    v = d
                                else:
                                    v = min(int(v), d)
                                w = float(self.p ** (v * self.alpha))
                            W_np[i, j] = w
                    self._W_cache[cache_key] = mx.array(W_np)

                W_base = self._W_cache[cache_key]
                W_mx = W_base

                # 1. Schwarzschild metric warping based on state distances to parent
                if self.schwarzschild_warp and d > 0:
                    parent_keys = [(d - 1, 0 if item[0] == 1 else (item[0] % (self.p ** (d - 1)))) for item in items]
                    parent_vecs = [current.get(pkey, mx.zeros((self.dim,))) for pkey in parent_keys]
                    parents_matrix = mx.stack(parent_vecs)  # shape (N, dim)
                    
                    # Distance of child states to parent state vector
                    dist_to_parent = mx.sum((vecs_matrix - parents_matrix) ** 2, axis=-1, keepdims=True)  # shape (N, 1)
                    
                    # Gravitational contraction factor
                    gravity_warp = 1.0 - (self.r_s / (self.r_s + dist_to_parent + dist_to_parent.T + 1e-6))
                    W_mx = W_mx * gravity_warp

                # 2. Einstein-Rosen Wormhole Shortcuts
                if self.wormhole_gate:
                    if self.learned_gate is not None:
                        gate_matrix = self.learned_gate.compute_gate_matrix(vecs_matrix)
                        # Only allow wormholes between tree-distant branches (W_base == 0)
                        tree_distant_mask = (W_base == 0.0)
                        gate_matrix = gate_matrix * tree_distant_mask
                    else:
                        # Fallback: hard cosine threshold
                        normed_vecs = vecs_matrix / (mx.linalg.norm(vecs_matrix, axis=-1, keepdims=True) + 1e-6)
                        cos_sim = mx.matmul(normed_vecs, normed_vecs.T)
                        gate_matrix = mx.where((cos_sim > 0.85) & (W_base == 0.0), cos_sim, 0.0)
                    
                    phase = self.floquet.get_phase() if self.floquet is not None else mx.array(1.0)
                    W_mx = W_mx + (self.epsilon * phase) * gate_matrix

                # Normalize rows dynamically after warping/shortcuts
                row_sums = mx.sum(W_mx, axis=1, keepdims=True)
                W_mx = mx.where(row_sums > 0, W_mx / (row_sums + 1e-9), W_mx)

                mixed_matrix = mx.matmul(W_mx, vecs_matrix)  # shape (N, dim)

                # Unpack and apply parent mixing
                for i in range(N):
                    pref, _ = items[i]
                    mixed = mixed_matrix[i]
                    if d > 0:
                        parent_pref = 0 if d == 1 else (pref % (self.p ** (d - 1)))
                        parent_key = (d - 1, parent_pref)
                        if parent_key in current:
                            cross_w = float(self.p ** (-1.0 * self.alpha))
                            mixed = mixed * (1.0 / (1.0 + cross_w)) + current[parent_key] * (cross_w / (1.0 + cross_w))
                    new_states[(d, pref)] = mixed

            # Apply layer linear (with residual for stability in skeleton)
            for key, mixed in new_states.items():
                lin = self.mix_linears[layer_idx]
                transformed = lin(mixed)
                # simple residual + tanh for bounded mixing (keeps toy stable)
                new_states[key] = transformed + mixed
                new_states[key] = mx.tanh(new_states[key])

            current = new_states

        return current

    def __call__(
        self, states: Dict[Tuple[int, int], mx.array]
    ) -> Dict[Tuple[int, int], mx.array]:
        return self.forward(states)
