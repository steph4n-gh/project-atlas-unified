import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftRipsComplex:
    """
    Differentiable relaxation of the Vietoris-Rips filtration.
    Computes birth/death times of H0 (connected components) and H1 (loops)
    directly within the PyTorch autograd graph.
    """
    def __init__(self, max_filtration: float = 2.0):
        self.max_filtration = max_filtration

    def forward(self, skeleton: torch.Tensor) -> tuple:
        """
        skeleton: [B, K, K] attention skeleton (Morse-contracted summits)
        Returns:
            h0_diagrams: list of (births, deaths) tensors per batch
            h1_diagrams: list of (births, deaths) tensors per batch
        """
        B, K, _ = skeleton.shape
        device = skeleton.device
        dtype = skeleton.dtype
        
        # Symmetrize
        W = 0.5 * (skeleton + skeleton.transpose(-1, -2))
        # Convert similarity to distance: d = 1 - w
        D = torch.clamp(1.0 - W, min=0.0, max=self.max_filtration)
        mask = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
        D = D.masked_fill(mask, 0.0)
        
        h0_diagrams = []
        h1_diagrams = []
        
        # Precompute edge list indices
        edges = []
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((i, j))
        edge_indices = torch.tensor(edges, device=device, dtype=torch.long) # [M, 2]
        
        for b in range(B):
            Db = D[b]
            # Edge weights for this batch
            edge_weights = Db[edge_indices[:, 0], edge_indices[:, 1]]
            
            # Sort edges by weight (differentiable sorting proxy)
            sorted_idx = torch.argsort(edge_weights)
            sorted_edges = edge_indices[sorted_idx]
            sorted_weights = edge_weights[sorted_idx]
            
            # Union-Find tracking
            parent = list(range(K))
            
            def find(x):
                curr = x
                while parent[curr] != curr:
                    parent[curr] = parent[parent[curr]]
                    curr = parent[curr]
                return curr
                
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx
                    return True
                return False
                
            h0_births = [torch.tensor(0.0, device=device, dtype=dtype)]
            h0_deaths = [torch.tensor(self.max_filtration, device=device, dtype=dtype)] # Infinite component
            
            h1_births = []
            h1_deaths = []
            
            for i in range(len(sorted_edges)):
                u, v = sorted_edges[i]
                w = sorted_weights[i]
                
                if union(u.item(), v.item()):
                    # H0 merge: a component dies at filtration w
                    h0_births.append(torch.tensor(0.0, device=device, dtype=dtype))
                    h0_deaths.append(w)
                else:
                    # Loop born at filtration w
                    h1_births.append(w)
                    h1_deaths.append(torch.tensor(self.max_filtration, device=device, dtype=dtype))
            
            # Pack diagrams
            h0_b_tensor = torch.stack(h0_births) if h0_births else torch.empty(0, device=device, dtype=dtype)
            h0_d_tensor = torch.stack(h0_deaths) if h0_deaths else torch.empty(0, device=device, dtype=dtype)
            h0_diagrams.append((h0_b_tensor, h0_d_tensor))
            
            h1_b_tensor = torch.stack(h1_births) if h1_births else torch.empty(0, device=device, dtype=dtype)
            h1_d_tensor = torch.stack(h1_deaths) if h1_deaths else torch.empty(0, device=device, dtype=dtype)
            h1_diagrams.append((h1_b_tensor, h1_d_tensor))
            
        return h0_diagrams, h1_diagrams


class PersistenceLandscapeVectorizer(nn.Module):
    """
    Vectorizes persistence diagrams into stable, differentiable persistence landscapes.
    """
    def __init__(self, num_landscapes: int = 5, resolution: int = 50, max_filtration: float = 2.0):
        super().__init__()
        self.num_landscapes = num_landscapes
        self.resolution = resolution
        self.max_filtration = max_filtration
        # Fixed grid of filtration values
        self.register_buffer("grid", torch.linspace(0.0, max_filtration, resolution))

    def forward(self, births: torch.Tensor, deaths: torch.Tensor) -> torch.Tensor:
        """
        births, deaths: Tensors of shape [N]
        Returns: [num_landscapes, resolution] landscape representation
        """
        device = births.device
        dtype = births.dtype
        
        if births.numel() == 0:
            return torch.zeros(self.num_landscapes, self.resolution, device=device, dtype=dtype)
            
        # Evaluate tent functions: max(0, min(t - b, d - t))
        t = self.grid.unsqueeze(0)  # [1, resolution]
        b = births.unsqueeze(1)     # [N, 1]
        d = deaths.unsqueeze(1)     # [N, 1]
        
        tents = torch.clamp(torch.minimum(t - b, d - t), min=0.0)  # [N, resolution]
        
        # Sort values at each grid point descending to construct the landscape layers
        landscapes = torch.sort(tents, dim=0, descending=True).values  # [N, resolution]
        
        # Format output to fixed shape [num_landscapes, resolution]
        if landscapes.shape[0] < self.num_landscapes:
            pad_size = self.num_landscapes - landscapes.shape[0]
            pad = torch.zeros(pad_size, self.resolution, device=device, dtype=dtype)
            landscapes = torch.cat([landscapes, pad], dim=0)
        else:
            landscapes = landscapes[:self.num_landscapes]
            
        return landscapes


class PersistentHomologyLoss(nn.Module):
    """
    Differentiable topological training loss using Soft Rips complex and Persistence Landscapes.
    """
    def __init__(self, h0_weight: float = 1.0, h1_weight: float = 0.5, ref_ema_decay: float = 0.99):
        super().__init__()
        self.h0_weight = h0_weight
        self.h1_weight = h1_weight
        self.ref_ema_decay = ref_ema_decay
        
        self.rips = SoftRipsComplex()
        self.vectorizer = PersistenceLandscapeVectorizer()
        
        # Reference landscapes EMA state
        self.register_buffer("ref_landscape_h0", torch.zeros(5, 50))
        self.register_buffer("ref_landscape_h1", torch.zeros(5, 50))
        self.register_buffer("has_ref", torch.tensor(False, dtype=torch.bool))

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        skeleton: [B, K, K] attention skeleton
        Returns: scalar loss
        """
        device = skeleton.device
        h0_diags, h1_diags = self.rips.forward(skeleton)
        
        B = len(h0_diags)
        l_h0 = []
        l_h1 = []
        
        for b in range(B):
            h0_b, h0_d = h0_diags[b]
            h1_b, h1_d = h1_diags[b]
            
            land_h0 = self.vectorizer(h0_b, h0_d)
            land_h1 = self.vectorizer(h1_b, h1_d)
            
            l_h0.append(land_h0)
            l_h1.append(land_h1)
            
        batch_landscape_h0 = torch.stack(l_h0).mean(dim=0)
        batch_landscape_h1 = torch.stack(l_h1).mean(dim=0)
        
        if not self.has_ref:
            # Initialize reference on first batch
            with torch.no_grad():
                self.ref_landscape_h0.copy_(batch_landscape_h0)
                self.ref_landscape_h1.copy_(batch_landscape_h1)
                self.has_ref.fill_(True)
                
        elif self.training:
            # Update reference using EMA during training
            with torch.no_grad():
                self.ref_landscape_h0.mul_(self.ref_ema_decay).add_(batch_landscape_h0 * (1.0 - self.ref_ema_decay))
                self.ref_landscape_h1.mul_(self.ref_ema_decay).add_(batch_landscape_h1 * (1.0 - self.ref_ema_decay))
                
        # Differentiable Wasserstein-1 distance (L1 norm between landscapes)
        loss_h0 = F.l1_loss(batch_landscape_h0, self.ref_landscape_h0)
        loss_h1 = F.l1_loss(batch_landscape_h1, self.ref_landscape_h1)
        
        total_loss = self.h0_weight * loss_h0 + self.h1_weight * loss_h1
        return total_loss
