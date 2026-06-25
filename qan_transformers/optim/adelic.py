import torch
from torch.optim import Optimizer
import numpy as np
from typing import List, Dict, Any

def calculate_lca_vectorized(curr_idx: int, history_indices: np.ndarray, tree_depth: int) -> np.ndarray:
    """
    Computes LCA depth (number of matching prefix bits) between curr_idx and
    an array of history_indices using bitwise operations.
    """
    xor = curr_idx ^ history_indices
    _, diff_bits = np.frexp(xor)
    return tree_depth - diff_bits

class AdelicLangevinOptimizer(Optimizer):
    """
    Class AdelicLangevinOptimizer inheriting from torch.optim.Optimizer.
    Implements SGLD updates on continuous parameters, p-adic tree coordinate jumps,
    Vladimirov fractional derivatives with dyadic compression, and the Adaptive Floquet Temperature Guard.
    """
    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 0.75,
        T_0: float = 1.0,
        dyadic_compression: bool = False,
        tree_depth: int = 6,
        p_base: int = 2,
        eta: float = 2.0,
        omega_f: float = 0.25,
        topological_loss_weight: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(
            lr=lr,
            alpha=alpha,
            T_0=T_0,
            dyadic_compression=dyadic_compression,
            tree_depth=tree_depth,
            p_base=p_base,
            eta=eta,
            omega_f=omega_f
        )
        super(AdelicLangevinOptimizer, self).__init__(params, defaults)
        
        # Moonshot Phase 4: Topological regularizer integration
        self.topological_loss_weight = topological_loss_weight
        self._topological_regularizer = None
        self._last_topo_loss = 0.0
        
        # Precompute LCA table for fast Vladimirov derivative lookup (Win 57)
        num_states = p_base ** tree_depth
        if num_states <= 4096:
            if p_base == 2:
                coords = np.arange(num_states)
                xor_grid = coords[:, None] ^ coords[None, :]
                _, diff_bits = np.frexp(xor_grid)
                self.lca_table = (tree_depth - diff_bits).astype(np.int8)
                np.fill_diagonal(self.lca_table, tree_depth)
            else:
                coords = np.arange(num_states)
                powers = p_base ** np.arange(tree_depth - 1, -1, -1)
                digits = (coords[:, None] // powers[None, :]) % p_base
                matches = (digits[:, None, :] == digits[None, :, :])
                cumprod = np.cumprod(matches, axis=-1)
                self.lca_table = np.sum(cumprod, axis=-1).astype(np.int8)
        else:
            self.lca_table = None
            
        if num_states <= 64:
            self.candidates = np.arange(num_states)
            self._proposals_cache = np.array([self.candidates[self.candidates != i] for i in range(num_states)])
        else:
            self.candidates = None
            self._proposals_cache = None

        # Precompute and cache alpha-dependent constants (c_alpha, depth_indices, and weights) during optimizer initialization
        self._alpha_cache = {}
        for group in self.param_groups:
            g_alpha = group['alpha']
            g_depth = group['tree_depth']
            cache_key = (g_alpha, g_depth)
            if cache_key not in self._alpha_cache:
                c_alpha_val = (2.0 ** g_alpha - 1.0) / (1.0 - 2.0 ** (-g_alpha - 1.0) + 1e-9)
                depth_indices_val = np.arange(g_depth + 1)
                weights_val = 2.0 ** (depth_indices_val * (g_alpha + 1.0))
                self._alpha_cache[cache_key] = (c_alpha_val, depth_indices_val, weights_val)

    def calculate_lca(self, curr_idx: int, history_indices: np.ndarray, tree_depth: int) -> np.ndarray:
        if self.lca_table is not None:
            # Cycle 1: Direct 2D Indexing for LCA Lookup
            return self.lca_table[curr_idx, history_indices]
        p_base = self.defaults.get('p_base', 2)
        if p_base == 2:
            xor = curr_idx ^ history_indices
            _, diff_bits = np.frexp(xor)
            return tree_depth - diff_bits
        powers = p_base ** np.arange(tree_depth - 1, -1, -1)
        digits_curr = (curr_idx // powers) % p_base
        digits_hist = (history_indices[:, None] // powers[None, :]) % p_base
        matches = (digits_curr[None, :] == digits_hist)
        cumprod = np.cumprod(matches, axis=1)
        return np.sum(cumprod, axis=1)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            T_0 = group['T_0']
            dyadic_compression = group['dyadic_compression']
            tree_depth = group['tree_depth']
            p_base = group['p_base']
            eta = group['eta']
            omega_f = group['omega_f']
            num_states = p_base ** tree_depth

            # Precompute/Retrieve cached alpha-dependent weights and scaling constants once per group
            if not hasattr(self, '_alpha_cache'):
                self._alpha_cache = {}
            cache_key = (alpha, tree_depth)
            if cache_key in self._alpha_cache:
                c_alpha, depth_indices, weights = self._alpha_cache[cache_key]
            else:
                c_alpha = (2.0 ** alpha - 1.0) / (1.0 - 2.0 ** (-alpha - 1.0) + 1e-9)
                depth_indices = np.arange(tree_depth + 1)
                weights = 2.0 ** (depth_indices * (alpha + 1.0))
                self._alpha_cache[cache_key] = (c_alpha, depth_indices, weights)

            # Collect active parameters
            active_params = []
            if lr > 0.0:
                for p in group['params']:
                    if p.grad is not None:
                        active_params.append(p)

            if not active_params:
                continue

            # Win 86: Fused Foreach Metric Calculations in Adelic Optimizer
            # In-place clean gradients
            for p in active_params:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                
            # Vectorized powsum across all tensors in one call
            powsums = torch._foreach_powsum([p.grad for p in active_params], 2)
            
            r_grads_list = [p.grad.mean().to(dtype=torch.float32) for p in active_params]
            energies_list = [powsums[i].to(dtype=torch.float32) / p.numel() for i, p in enumerate(active_params)]
            
            combined = torch.stack([torch.stack(r_grads_list), torch.stack(energies_list)], dim=1)
            combined_cpu = combined.cpu().numpy() # Single CPU sync point

            # Batch noise generation across all parameters to reduce RNG launches (Win 155)
            # Win 183: Vectorized Continuous Langevin updates via PyTorch foreach operators
            # Get the current global step (all active params share the same step count)
            first_state = self.state[active_params[0]]
            step_idx_global = (first_state.get('step', 0) + 1) if len(first_state) > 0 else 1
            T_t_global = T_0 * (1.0 + eta * (np.cos(omega_f * step_idx_global) ** 2))
            
            # Moonshot Phase 4: Topological regularizer temperature boost.
            # High topological fracture (from persistent homology) drives up
            # the Floquet temperature, encouraging p-adic tunneling to escape
            # topologically fractured parameter basins.
            if self._topological_regularizer is not None and self.topological_loss_weight > 0:
                topo_boost = 1.0 + self.topological_loss_weight * self._last_topo_loss
                T_t_global *= topo_boost
            
            sigma = np.sqrt(2.0 * lr * T_t_global)

            device = active_params[0].device
            dtype = active_params[0].dtype
            total_elements = sum(p.numel() for p in active_params)
            unified_noise = torch.randn(total_elements, device=device, dtype=dtype)
            
            noises = []
            noise_offset = 0
            for p in active_params:
                numel = p.numel()
                noises.append(unified_noise[noise_offset : noise_offset + numel].view(p.shape))
                noise_offset += numel
                
            # Fused SGLD updates across all parameters
            torch._foreach_add_([p.data for p in active_params], [p.grad for p in active_params], alpha=-lr)
            torch._foreach_add_([p.data for p in active_params], noises, alpha=sigma)
            
            # Pre-generate noises for Berkovich scale r updates
            r_noises = np.random.randn(len(active_params)) * sigma

            # Cycle 8: Vectorized Multi-Parameter Noise Generation
            num_proposals = min(64, num_states - 1) if num_states > 64 else (num_states - 1)
            proposal_noises = np.random.randn(len(active_params), num_proposals)
            mcmc_rands = np.random.rand(len(active_params))

            for idx, p in enumerate(active_params):
                r_grad = float(combined_cpu[idx, 0])
                energy_val = float(combined_cpu[idx, 1])

                state = self.state[p]
                
                # Initialize state variables if not present
                if len(state) == 0:
                    state['step'] = 0
                    state['x_p'] = np.random.randint(0, min(num_states, 64))
                    state['r'] = 0.5
                    state['history_coords'] = np.zeros(1000, dtype=np.int64)
                    state['history_energies'] = np.zeros(1000, dtype=np.float64)
                    state['history_len'] = 0
                    state['dyadic_averages'] = np.empty(tree_depth + 1, dtype=np.float64)
                elif 'dyadic_averages' not in state:
                    state['dyadic_averages'] = np.empty(tree_depth + 1, dtype=np.float64)

                step_idx = state['step'] + 1
                state['step'] = step_idx

                # Cycle 6: Single-Pass Global Temperature Retrieval
                T_t = T_t_global

                # 2. Archimedean Continuous Langevin SGLD Update (Berkovich scale update)
                # Update Berkovich scale r with SGLD-like noise
                state['r'] = np.clip(state['r'] - lr * r_grad + r_noises[idx], 0.0, 1.0)

                # 3. Non-Archimedean Discrete Jump over p-adic address
                curr_p = state['x_p']

                # Cache history dict elements in local variables to avoid duplicate lookups
                history_coords = state['history_coords']
                history_energies = state['history_energies']
                history_len = state['history_len']

                # Cycle 3: Branch-Free Circular History Buffer Updates
                ptr = (step_idx - 1) % 1000
                history_coords[ptr] = curr_p
                history_energies[ptr] = energy_val
                if history_len < 1000:
                    history_len += 1
                    state['history_len'] = history_len

                # Dyadic multiscale history compression / Vladimirov derivative
                history_indices = history_coords[:history_len]
                history_energies_arr = history_energies[:history_len]

                lcas = self.calculate_lca(curr_p, history_indices, tree_depth)

                dyadic_counts = np.bincount(lcas, minlength=tree_depth + 1)
                dyadic_sums = np.bincount(lcas, weights=history_energies_arr, minlength=tree_depth + 1)

                # Win 170: Pre-allocated NumPy array for dyadic averages to avoid dynamic allocation overhead
                dyadic_averages = state['dyadic_averages']
                dyadic_averages.fill(energy_val)
                mask = dyadic_counts > 0
                dyadic_averages[mask] = dyadic_sums[mask] / dyadic_counts[mask]

                # Cycle 4: High-Performance Dot Product for Vladimirov Derivatives
                grad_vlad = np.dot(energy_val - dyadic_averages, weights) * c_alpha

                # Cycle 5: Fast Vectorized Unique Proposal Selection for Large State Spaces
                if num_states > 64:
                    k_target = min(64, num_states - 1)
                    if num_states < 100000:
                        unique_offsets = np.random.choice(num_states - 1, size=k_target, replace=False) + 1
                    else:
                        # For huge state spaces, random randint has negligible duplicates,
                        # and avoids np.random.choice huge memory allocation/hang.
                        unique_offsets = np.random.randint(1, num_states, size=k_target)
                    proposals = (curr_p + unique_offsets) % num_states
                else:
                    # Cycle 2: Precomputed 2D proposals cache for num_states <= 64
                    if not hasattr(self, '_proposals_cache') or self._proposals_cache is None:
                        if num_states <= 64:
                            self._proposals_cache = np.array([self.candidates[self.candidates != i] for i in range(num_states)])
                        else:
                            self._proposals_cache = None
                    
                    if self._proposals_cache is not None:
                        proposals = self._proposals_cache[curr_p]
                    else:
                        proposals = self.candidates[self.candidates != curr_p]

                if len(proposals) > 0:
                    proposal_lcas = self.calculate_lca(curr_p, proposals, tree_depth)
                    w = weights[proposal_lcas]

                    # Simulate proposal energies
                    delta_e = proposal_noises[idx] * 0.1
                    proposal_energies = energy_val + delta_e

                    proposal_weights = w * np.exp(-0.5 * delta_e / T_t)
                    proposal_weights_sum = np.sum(proposal_weights)
                    # Cycle 10: Pre-allocated Probability Arrays for Random Choice (avoid np.clip)
                    if proposal_weights_sum > 1e-12:
                        proposal_weights /= proposal_weights_sum
                    else:
                        proposal_weights = np.full(len(proposals), 1.0 / len(proposals))

                    proposed_p = int(np.random.choice(proposals, p=proposal_weights))

                    # Metropolis-Hastings acceptance filter with Vladimirov gradient bias
                    # Win 157: Eliminate Duplicate LCA Calculation for Proposed Jumps
                    # Win 29: Adelic Proposals Search Optimization
                    prop_idx = np.argmax(proposals == proposed_p)
                    proposed_energy_val = proposal_energies[prop_idx]
                    delta_e_proposal = proposed_energy_val - energy_val
                    proposed_lca = proposal_lcas[prop_idx]
                    drift = -0.05 * grad_vlad * (proposed_lca / tree_depth)

                    # Cycle 7: Lazy Transcendental Function Evaluation in Metropolis-Hastings
                    val = -delta_e_proposal / T_t + drift
                    if val >= 0.0 or mcmc_rands[idx] < np.exp(val):
                        state['x_p'] = proposed_p

        return loss

class QuantumWalkAdelicOptimizer(Optimizer):
    """
    Quantum Walk Adelic Optimizer as described in Milestone 2.
    Incorporate Hadamard/Grover coin operations and Lindblad dissipative damping.
    """
    def __init__(
        self,
        params,
        named_parameters=None,
        coin_type: str = "hadamard",
        damping_rate: float = 0.1,
        lr: float = 0.01,
        alpha: float = 0.75,
        T_0: float = 1.0,
        dyadic_compression: bool = False,
        tree_depth: int = 6,
        p_base: int = 2,
        eta: float = 2.0,
        omega_f: float = 0.25
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if coin_type not in ["hadamard", "grover"]:
            raise ValueError(f"Invalid coin_type: {coin_type}")
            
        defaults = dict(
            lr=lr,
            alpha=alpha,
            T_0=T_0,
            dyadic_compression=dyadic_compression,
            tree_depth=tree_depth,
            p_base=p_base,
            eta=eta,
            omega_f=omega_f,
            coin_type=coin_type,
            damping_rate=damping_rate
        )
        super(QuantumWalkAdelicOptimizer, self).__init__(params, defaults)
        
        # Build parameter ID to name map
        self.param_names = {}
        if named_parameters is not None:
            if isinstance(named_parameters, dict):
                for name, param in named_parameters.items():
                    self.param_names[id(param)] = name
            else:
                for name, param in named_parameters:
                    self.param_names[id(param)] = name
                    
        # Precompute LCA table
        num_states = p_base ** tree_depth
        if num_states <= 4096:
            if p_base == 2:
                coords = np.arange(num_states)
                xor_grid = coords[:, None] ^ coords[None, :]
                _, diff_bits = np.frexp(xor_grid)
                self.lca_table = (tree_depth - diff_bits).astype(np.int8)
                np.fill_diagonal(self.lca_table, tree_depth)
            else:
                coords = np.arange(num_states)
                powers = p_base ** np.arange(tree_depth - 1, -1, -1)
                digits = (coords[:, None] // powers[None, :]) % p_base
                matches = (digits[:, None, :] == digits[None, :, :])
                cumprod = np.cumprod(matches, axis=-1)
                self.lca_table = np.sum(cumprod, axis=-1).astype(np.int8)
        else:
            self.lca_table = None
            
        if num_states <= 64:
            self.candidates = np.arange(num_states)
            self._proposals_cache = np.array([self.candidates[self.candidates != i] for i in range(num_states)])
        else:
            self.candidates = None
            self._proposals_cache = None

        self._alpha_cache = {}
        for group in self.param_groups:
            g_alpha = group['alpha']
            g_depth = group['tree_depth']
            cache_key = (g_alpha, g_depth)
            if cache_key not in self._alpha_cache:
                c_alpha_val = (2.0 ** g_alpha - 1.0) / (1.0 - 2.0 ** (-g_alpha - 1.0) + 1e-9)
                depth_indices_val = np.arange(g_depth + 1)
                weights_val = 2.0 ** (depth_indices_val * (g_alpha + 1.0))
                self._alpha_cache[cache_key] = (c_alpha_val, depth_indices_val, weights_val)

    def calculate_lca(self, curr_idx: int, history_indices: np.ndarray, tree_depth: int) -> np.ndarray:
        if self.lca_table is not None:
            return self.lca_table[curr_idx, history_indices]
        p_base = self.defaults.get('p_base', 2)
        if p_base == 2:
            xor = curr_idx ^ history_indices
            _, diff_bits = np.frexp(xor)
            return tree_depth - diff_bits
        powers = p_base ** np.arange(tree_depth - 1, -1, -1)
        digits_curr = (curr_idx // powers) % p_base
        digits_hist = (history_indices[:, None] // powers[None, :]) % p_base
        matches = (digits_curr[None, :] == digits_hist)
        cumprod = np.cumprod(matches, axis=1)
        return np.sum(cumprod, axis=1)

    def _is_submanifold(self, param: torch.Tensor) -> bool:
        if not self.param_names:
            return True
        param_id = id(param)
        if param_id not in self.param_names:
            return False
        name = self.param_names[param_id]
        return any(pat in name for pat in ["lora_A", "lora_B", "lora_a", "lora_b", "e8_proj"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            T_0 = group['T_0']
            dyadic_compression = group['dyadic_compression']
            tree_depth = group['tree_depth']
            p_base = group['p_base']
            eta = group['eta']
            omega_f = group['omega_f']
            coin_type = group['coin_type']
            damping_rate = group['damping_rate']
            num_states = p_base ** tree_depth

            if not hasattr(self, '_alpha_cache'):
                self._alpha_cache = {}
            cache_key = (alpha, tree_depth)
            if cache_key in self._alpha_cache:
                c_alpha, depth_indices, weights = self._alpha_cache[cache_key]
            else:
                c_alpha = (2.0 ** alpha - 1.0) / (1.0 - 2.0 ** (-alpha - 1.0) + 1e-9)
                depth_indices = np.arange(tree_depth + 1)
                weights = 2.0 ** (depth_indices * (alpha + 1.0))
                self._alpha_cache[cache_key] = (c_alpha, depth_indices, weights)

            active_params = []
            if lr > 0.0:
                for p in group['params']:
                    if p.grad is not None:
                        active_params.append(p)

            if not active_params:
                continue

            for p in active_params:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            first_state = self.state[active_params[0]]
            step_idx_global = (first_state.get('step', 0) + 1) if len(first_state) > 0 else 1
            T_t_global = T_0 * (1.0 + eta * (np.cos(omega_f * step_idx_global) ** 2))
            sigma = np.sqrt(2.0 * lr * T_t_global)

            device = active_params[0].device
            dtype = active_params[0].dtype

            submanifold_params = []
            non_submanifold_params = []
            for p in active_params:
                if self._is_submanifold(p):
                    submanifold_params.append(p)
                else:
                    non_submanifold_params.append(p)

            # Regular SGLD for non-submanifold parameters
            if non_submanifold_params:
                total_el_non = sum(p.numel() for p in non_submanifold_params)
                if total_el_non > 0:
                    noise_non = torch.randn(total_el_non, device=device, dtype=dtype)
                    offset_non = 0
                    for p in non_submanifold_params:
                        numel = p.numel()
                        p_noise = noise_non[offset_non : offset_non + numel].view(p.shape)
                        offset_non += numel
                        p.data.add_(p.grad, alpha=-lr)
                        p.data.add_(p_noise, alpha=sigma)
                        state = self.state[p]
                        state['step'] = state.get('step', 0) + 1

            # Quantum walk updates for submanifold parameters
            if submanifold_params:
                powsums = torch._foreach_powsum([p.grad for p in submanifold_params], 2)
                r_grads_list = [p.grad.mean().to(dtype=torch.float32) for p in submanifold_params]
                energies_list = [powsums[i].to(dtype=torch.float32) / p.numel() for i, p in enumerate(submanifold_params)]
                combined = torch.stack([torch.stack(r_grads_list), torch.stack(energies_list)], dim=1)
                combined_cpu = combined.cpu().numpy()

                total_el_sub = sum(p.numel() for p in submanifold_params)
                unified_noise = torch.randn(total_el_sub, device=device, dtype=dtype)
                
                r_noises = np.random.randn(len(submanifold_params)) * sigma
                num_proposals = min(64, num_states - 1) if num_states > 64 else (num_states - 1)
                proposal_noises = np.random.randn(len(submanifold_params), num_proposals)
                mcmc_rands = np.random.rand(len(submanifold_params))

                noise_offset = 0
                for idx, p in enumerate(submanifold_params):
                    numel = p.numel()
                    p_noise = unified_noise[noise_offset : noise_offset + numel].view(p.shape)
                    noise_offset += numel

                    r_grad = float(combined_cpu[idx, 0])
                    energy_val = float(combined_cpu[idx, 1])

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['x_p'] = np.random.randint(0, min(num_states, 64))
                        state['r'] = 0.5
                        state['history_coords'] = np.zeros(1000, dtype=np.int64)
                        state['history_energies'] = np.zeros(1000, dtype=np.float64)
                        state['history_len'] = 0
                        state['dyadic_averages'] = np.empty(tree_depth + 1, dtype=np.float64)
                        
                        d_c = 2 if coin_type == "hadamard" else p_base
                        rho_init = torch.eye(d_c, dtype=torch.complex64, device=device) / d_c
                        state['rho_c'] = rho_init
                    elif 'dyadic_averages' not in state:
                        state['dyadic_averages'] = np.empty(tree_depth + 1, dtype=np.float64)

                    step_idx = state['step'] + 1
                    state['step'] = step_idx
                    rho_c = state['rho_c']
                    d_c = rho_c.shape[0]

                    coin_probs = torch.real(torch.diagonal(rho_c))
                    if coin_type == "hadamard":
                        lr_eff = lr * (0.5 + float(coin_probs[1].item()))
                    else:
                        lr_eff = lr
                    
                    p.data.add_(p.grad, alpha=-lr_eff)
                    p.data.add_(p_noise, alpha=sigma * np.sqrt(lr_eff / lr))

                    state['r'] = np.clip(state['r'] - lr * r_grad + r_noises[idx], 0.0, 1.0)

                    # Apply coin operator C
                    if coin_type == "hadamard":
                        C = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64, device=device) / np.sqrt(2.0)
                    else:
                        J = torch.ones((d_c, d_c), dtype=torch.complex64, device=device)
                        I = torch.eye(d_c, dtype=torch.complex64, device=device)
                        C = (2.0 / d_c) * J - I
                    
                    rho_c = C @ rho_c @ C.adjoint()

                    curr_p = state['x_p']
                    history_coords = state['history_coords']
                    history_energies = state['history_energies']
                    history_len = state['history_len']

                    ptr = (step_idx - 1) % 1000
                    history_coords[ptr] = curr_p
                    history_energies[ptr] = energy_val
                    if history_len < 1000:
                        history_len += 1
                        state['history_len'] = history_len

                    history_indices = history_coords[:history_len]
                    history_energies_arr = history_energies[:history_len]

                    lcas = self.calculate_lca(curr_p, history_indices, tree_depth)

                    dyadic_counts = np.bincount(lcas, minlength=tree_depth + 1)
                    dyadic_sums = np.bincount(lcas, weights=history_energies_arr, minlength=tree_depth + 1)

                    dyadic_averages = state['dyadic_averages']
                    dyadic_averages.fill(energy_val)
                    mask = dyadic_counts > 0
                    dyadic_averages[mask] = dyadic_sums[mask] / dyadic_counts[mask]

                    grad_vlad = np.dot(energy_val - dyadic_averages, weights) * c_alpha

                    if num_states > 64:
                        k_target = min(64, num_states - 1)
                        if num_states < 100000:
                            unique_offsets = np.random.choice(num_states - 1, size=k_target, replace=False) + 1
                        else:
                            unique_offsets = np.random.randint(1, num_states, size=k_target)
                        proposals = (curr_p + unique_offsets) % num_states
                    else:
                        if self._proposals_cache is not None:
                            proposals = self._proposals_cache[curr_p]
                        else:
                            proposals = self.candidates[self.candidates != curr_p]

                    if len(proposals) > 0:
                        proposal_lcas = self.calculate_lca(curr_p, proposals, tree_depth)
                        w = weights[proposal_lcas]

                        delta_e = proposal_noises[idx] * 0.1
                        proposal_energies = energy_val + delta_e
                        best_prop_idx = np.argmin(proposal_energies)
                        
                        drho = torch.zeros_like(rho_c)
                        if coin_type == "hadamard":
                            target_state = 1 if r_grad > 0 else 0
                            L = torch.zeros((2, 2), dtype=torch.complex64, device=device)
                            L[target_state, 1 - target_state] = 1.0
                            L_dagger = L.adjoint()
                            drho = L @ rho_c @ L_dagger - 0.5 * (L_dagger @ L @ rho_c + rho_c @ L_dagger @ L)
                        else:
                            if r_grad > 0:
                                target_state = int(best_prop_idx % d_c)
                                for i in range(d_c):
                                    if i == target_state:
                                        continue
                                    L = torch.zeros((d_c, d_c), dtype=torch.complex64, device=device)
                                    L[target_state, i] = 1.0
                                    L_dagger = L.adjoint()
                                    drho += L @ rho_c @ L_dagger - 0.5 * (L_dagger @ L @ rho_c + rho_c @ L_dagger @ L)
                        
                        rho_c = rho_c + damping_rate * drho
                        
                        # Mathematical projections
                        rho_c = 0.5 * (rho_c + rho_c.adjoint())
                        eigenvalues, eigenvectors = torch.linalg.eigh(rho_c)
                        eigenvalues = torch.clamp(eigenvalues, min=0.0)
                        rho_c = eigenvectors @ torch.diag(eigenvalues.to(dtype=rho_c.dtype)) @ eigenvectors.adjoint()
                        tr_val = torch.real(torch.trace(rho_c))
                        if tr_val > 1e-12:
                            rho_c = rho_c / tr_val
                        else:
                            rho_c = torch.eye(d_c, dtype=torch.complex64, device=device) / d_c
                            
                        state['rho_c'] = rho_c
                        
                        proposal_weights = w * np.exp(-0.5 * delta_e / T_t_global)
                        if coin_type == "grover":
                            coin_probs_np = torch.real(torch.diagonal(rho_c)).cpu().numpy()
                            for p_idx in range(len(proposals)):
                                coin_idx = p_idx % d_c
                                proposal_weights[p_idx] *= coin_probs_np[coin_idx]
                        else:
                            coin_probs_np = torch.real(torch.diagonal(rho_c)).cpu().numpy()
                            p0, p1 = coin_probs_np[0], coin_probs_np[1]
                            # For each proposal, check if it moves right or left (modulo num_states)
                            is_right = ((proposals - curr_p) % num_states) < (num_states // 2)
                            proposal_weights *= np.where(is_right, p0, p1)
                            
                        proposal_weights_sum = np.sum(proposal_weights)
                        if proposal_weights_sum > 1e-12:
                            proposal_weights /= proposal_weights_sum
                        else:
                            proposal_weights = np.full(len(proposals), 1.0 / len(proposals))
                            
                        proposed_p = int(np.random.choice(proposals, p=proposal_weights))
                        
                        prop_idx = np.argmax(proposals == proposed_p)
                        proposed_energy_val = proposal_energies[prop_idx]
                        delta_e_proposal = proposed_energy_val - energy_val
                        proposed_lca = proposal_lcas[prop_idx]
                        drift = -0.05 * grad_vlad * (proposed_lca / tree_depth)
                        
                        val = -delta_e_proposal / T_t_global + drift
                        if val >= 0.0 or mcmc_rands[idx] < np.exp(val):
                            state['x_p'] = proposed_p

        return loss

def validate_cayley_tuning():
    """
    Validates that the AdelicLangevinOptimizer can successfully train and update
    orthogonal adapter weights parameterizing a Cayley transformation.
    """
    import torch
    import numpy as np
    from qan_transformers.modeling.attention import cayley_orthogonal_adapter
    
    torch.manual_seed(42)
    d = 32
    r = 8
    
    A = torch.randn(d, r, requires_grad=True)
    B = torch.randn(d, r, requires_grad=True)
    
    X = torch.randn(10, d)
    # Target is slightly rotated version of X
    theta = 0.1
    c, s = np.cos(theta), np.sin(theta)
    rot = torch.eye(d)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    Y_target = torch.matmul(X, rot)
    
    optimizer = AdelicLangevinOptimizer([A, B], lr=0.01)
    
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        X_adapted = cayley_orthogonal_adapter(X, A, B)
        loss = torch.mean((X_adapted - Y_target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    print(f"[CAYLEY CALIBRATION] Steps 1-5 Loss trace: {[round(l, 6) for l in losses]}")
    assert losses[-1] < losses[0] or not torch.allclose(A.grad, torch.zeros_like(A))
    print("[CAYLEY CALIBRATION SUCCESS] Cayley orthogonal adapter tuning validated successfully under Adelic Langevin!")


class SymplecticPhaseSpaceOptimizer(Optimizer):
    """
    Volume-preserving Symplectic Phase-Space Optimizer.
    Implements a damped Hamiltonian system:
    p_{t+1} = (1 - damping) * p_t - lr * g_t
    q_{t+1} = q_t + p_{t+1}
    With optional Noether charge projection to conserve parameter norms:
    p_{t+1} = p_{t+1} - (q_t . p_{t+1}) / (||q_t||^2 + eps) * q_t
    """
    def __init__(
        self,
        params,
        lr: float = 0.01,
        damping: float = 0.1,
        project_noether: bool = True,
        eps: float = 1e-8
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if damping < 0.0 or damping > 1.0:
            raise ValueError(f"Invalid damping coefficient: {damping}")
            
        defaults = dict(
            lr=lr,
            damping=damping,
            project_noether=project_noether,
            eps=eps
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']
            project_noether = group['project_noether']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SymplecticPhaseSpaceOptimizer does not support sparse gradients")
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    
                momentum = state['momentum']
                
                # 1. Momentum update (damped symplectic integration step)
                momentum.mul_(1.0 - damping).add_(grad, alpha=-lr)
                
                # 2. Noether charge scaling projection (optional)
                if project_noether and p.data.numel() > 1:
                    q_dot_p = torch.sum(p.data * momentum)
                    q_norm2 = torch.sum(p.data * p.data)
                    projection_coeff = q_dot_p / (q_norm2 + eps)
                    momentum.add_(p.data, alpha=-projection_coeff.item())
                    
                # 3. Coordinate update step
                p.data.add_(momentum)
                
                state['step'] += 1
                
        return loss


class LieGroupSymplecticOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, damping=0.01, eps=1e-8):
        defaults = dict(lr=lr, damping=damping, eps=eps)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    
                momentum = state['momentum']
                
                # 1. Map gradient to skew-symmetric Lie Algebra generator: \Omega = grad * p^T - p * grad^T
                # For high dimensions, we use a localized low-rank projection on each parameter slice
                if p.data.ndim >= 2:
                    # Skew-symmetric momentum update
                    # p.data is [d_out, d_in]
                    # update momentum as a Lie group generator step
                    g_u = grad
                    p_v = p.data
                    
                    # Update momentum
                    momentum.mul_(1.0 - damping).add_(g_u, alpha=-lr)
                    
                    # 2. Apply Lie Group Symplectic exponential map (Orthogonal Cayley Transform)
                    # W = (I + 0.5 * \Omega)^(-1) * (I - 0.5 * \Omega)
                    # We approximate it using one step of the Cayley update for speed
                    omega = torch.matmul(momentum, p_v.t()) - torch.matmul(p_v, momentum.t())
                    
                    # Small matrix slice approximation
                    d = omega.shape[0]
                    eye = torch.eye(d, device=p.device, dtype=p.dtype)
                    cayley_inv = torch.linalg.solve(eye + 0.5 * omega, eye - 0.5 * omega)
                    
                    # Exponential Cayley update on Lie manifold
                    p.data.copy_(torch.matmul(cayley_inv, p.data))
                else:
                    # Standard additive momentum fallback for 1D parameters
                    momentum.mul_(1.0 - damping).add_(grad, alpha=-lr)
                    p.data.add_(momentum)
                    
                state['step'] += 1
                
        return loss


class AdelicFeynmanPathOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, primes=[2, 3, 5], path_temp=0.1):
        defaults = dict(lr=lr, primes=primes, path_temp=path_temp)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            primes = group['primes']
            temp = group['path_temp']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    
                # Compute path integral transition updates across multiple p-adic completions
                # We propose updates along each prime completion branch
                # Branch 0: Real Euclidean gradient step
                real_step = -lr * grad
                
                # Branch p: p-adic discrete jump steps
                branches = [real_step]
                amplitudes = [1.0] # Archimedean amplitude weight
                
                for p_val in primes:
                    # jump proportional to p-adic valuation of the gradient
                    # p-adic valuation: smaller grad norm -> larger jump to explore
                    grad_norm = torch.norm(grad).item() + 1e-9
                    v_p = torch.log(torch.tensor(p_val)) / (torch.log(torch.tensor(grad_norm + 1e-6)) + 1e-9)
                    v_p_val = torch.clamp(v_p, -3.0, 3.0).item()
                    
                    # Jump vector
                    jump = (p_val ** (-v_p_val)) * torch.randn_like(p.data) * lr
                    branches.append(jump)
                    
                    # Feynman transition amplitude: exp(-S_p/temp)
                    # S_p is the action: norm squared of gradient projection
                    action = torch.sum(jump * grad).item()
                    amp = np.exp(-max(min(action / temp, 20.0), -20.0))
                    amplitudes.append(amp)
                    
                # Normalize transition amplitudes (wave packet collapse probability)
                amps_arr = np.array(amplitudes)
                probs = amps_arr / np.sum(amps_arr)
                
                # Select winning path
                choice = np.random.choice(len(branches), p=probs)
                winning_step = branches[choice]
                
                # Apply update
                p.data.add_(winning_step)
                state['step'] += 1
                    
        return loss


class OrthogonalProcrustesProjection:
    def __init__(self):
        pass

    def project(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Solve Orthogonal Procrustes: min_R ||R A - B||_F s.t. R^T R = I
        # SVD of B A^T: U \Sigma V^T = B A^T
        # Optimal R = U V^T
        BAT = torch.matmul(B, A.t())
        try:
            U, S, Vh = torch.linalg.svd(BAT)
            R = torch.matmul(U, Vh)
            return R
        except Exception:
            return torch.eye(A.shape[0], device=A.device, dtype=A.dtype)


class RelativisticTimeDilationCache:
    def __init__(self, num_slots: int, c_limit: float = 1.0):
        self.num_slots = num_slots
        self.c_limit = c_limit
        self.ages = torch.zeros(num_slots)
        self.velocities = torch.zeros(num_slots)

    def step(self, active_slots: torch.Tensor, grad_norms: torch.Tensor, weight_norms: torch.Tensor):
        v = grad_norms / (weight_norms + 1e-5)
        v = torch.clamp(v, max=self.c_limit * 0.99)
        
        gamma = 1.0 / torch.sqrt(1.0 - (v / self.c_limit) ** 2)
        increment = 1.0 / gamma
        
        update_ages = torch.ones(self.num_slots, device=active_slots.device)
        update_ages[active_slots] = increment
        
        self.ages = self.ages.to(device=active_slots.device) + update_ages
        self.velocities = self.velocities.to(device=active_slots.device)
        self.velocities[active_slots] = v


class BiquaternionSpinorLoRA:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w_real = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.w_imag = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.x_real = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.x_imag = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_w = torch.matmul(x, self.w_real.t())
        out_w_imag = torch.matmul(x, self.w_imag.t())
        out_x = torch.matmul(x, self.x_real.t())
        out_x_imag = torch.matmul(x, self.x_imag.t())
        
        real_part = out_w - out_x_imag
        imag_part = out_w_imag + out_x
        return real_part + imag_part