"""Clopen digit routing heads (per-depth linears for digit-by-digit address prediction).

Skeleton: one small Linear per depth. Conditionality comes from feeding
the diffused state *at the candidate ball at that depth*.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn


__all__ = ["DigitHeads", "distillation_loss"]


class DigitHeads(nn.Module):
    """Per-depth heads for predicting the next p-adic digit (0..p-1) at each level.

    In the skeleton each head is a Linear(dim -> p). The 'conditional' aspect
    is realized in the caller (UCEModel) by selecting which ball's diffused
    state to pass to the head for a given prefix of a candidate address.
    """

    def __init__(self, p: int, depth: int, dim: int) -> None:
        super().__init__()
        if not isinstance(p, int) or p < 2:
            raise ValueError("p must be >=2")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be >=1")
        self.p = p
        self.depth = depth
        self.dim = dim

        # One head per depth level (0 = root/first digit ... depth-1). Plain list (MLX auto registers).
        self.heads: list = []
        for _ in range(depth):
            self.heads.append(nn.Linear(dim, p))

    def __call__(self, state_vec: mx.array, depth_idx: int) -> mx.array:
        """Return p logits for the digit choice at the given depth, using the state at that ball."""
        if not (0 <= depth_idx < self.depth):
            raise ValueError(f"depth_idx {depth_idx} out of range [0, {self.depth})")
        if state_vec.shape[-1] != self.dim:
            # allow (dim,) or (1,dim) etc; ensure vector
            state_vec = mx.reshape(state_vec, (self.dim,))
        return self.heads[depth_idx](state_vec)

    def forward(self, state_vec: mx.array, depth_idx: int) -> mx.array:
        return self(state_vec, depth_idx)


def distillation_loss(
    head_logits: mx.array, teacher_sub_dist: mx.array, temperature: float = 1.0
) -> mx.array:
    """Compute soft-label cross-entropy (distillation) loss for a digit head.

    head_logits: (p,) unnormalized logits from DigitHeads(..., depth_idx)
    teacher_sub_dist: (p,) target probability distribution over the p children
        (derived by restricting teacher logits to ball and grouping by child digit).
    temperature: softens the student logits for distillation (default 1.0 for hard-ish).

    Returns scalar loss (higher = worse match). Used in Phase 0 head warm-start.
    """
    if temperature != 1.0:
        head_logits = head_logits / temperature
    # MLX has no log_softmax; use logits - logsumexp (stable for 1D)
    log_probs = head_logits - mx.logsumexp(head_logits)
    # cross entropy with soft targets: -sum(target * logp)
    # (KL equiv up to entropy const of teacher)
    loss = -mx.sum(teacher_sub_dist * log_probs)
    return loss
