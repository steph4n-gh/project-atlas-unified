# Mathematical Specifications & Derivations

This document provides a mathematically rigorous formulation and derivation of the core algorithms implemented in **Project Atlas (QAN-ATLAS)**. 

---

## 1. Concentric Icosian Shell Mapping ($E_8$ Projection)

The standard $E_8$ Gosset root lattice contains 240 root vectors in $\mathbb{R}^8$ at norm squared equal to $2$. These roots are projected into $\mathbb{R}^3$ to create a coordinate-sparse attention grid while preserving icosahedral rotational and inversion symmetries.

### 1.1 The Golden Ratio Projection
The projection matrix $P \in \mathbb{R}^{8 \times 3}$ is constructed as a product of two projections:
$$P = P_{8 \to 4} \cdot P_{4 \to 3}$$

where $P_{8 \to 4} \in \mathbb{R}^{8 \times 4}$ maps the 8D space into a 4D space using the golden ratio $\phi = \frac{1+\sqrt{5}}{2}$ to embed the icosahedral symmetries, scaled by $s = \frac{1}{\sqrt{1 + \phi^2}}$:
$$P_{8 \to 4} = s \begin{bmatrix} 
\phi & 0 & 0 & 0 \\
0 & \phi & 0 & 0 \\
0 & 0 & \phi & 0 \\
0 & 0 & 0 & \phi \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 
\end{bmatrix}$$

and $P_{4 \to 3} \in \mathbb{R}^{4 \times 3}$ projects the 4D space into 3D by dropping the first coordinate:
$$P_{4 \to 3} = \begin{bmatrix} 
0 & 0 & 0 \\
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 
\end{bmatrix}$$

### 1.2 Concentration of Shells
When the 240 root coordinates $X_{E8} \in \mathbb{R}^{240 \times 8}$ are projected via $Y = X_{E8} \cdot P$, their Euclidean norms $\|y_i\|_2$ cluster into exactly 5 discrete shells:
*   **Shell 0**: $r = 0.0$ (2 points)
*   **Shell 1**: $r = \frac{1}{2}\sqrt{10 - 2\sqrt{5}} \approx 0.5878$ (30 points)
*   **Shell 2**: $r = \frac{\sqrt{3}}{2} \approx 0.8660$ (64 points)
*   **Shell 3**: $r = \frac{1}{2}\sqrt{10 + 2\sqrt{5}} \approx 0.9511$ (64 points)
*   **Shell 4**: $r = 1.0$ (80 points)

*Code Reference*: Generated dynamically in [e8_projection.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/math/e8_projection.py#L7-L98) (`generate_dynamic_e8_coordinates`).

---

## 2. p-Adic Tree Coordinate Routing & E8 Integration (UCE)

The **Ultrametric Cognitive Engine (UCE)** imposes a tree structure over the continuous attention space to enable efficient routing and sequence sorting.

### 2.1 continuous-to-p-adic Mapping
The input hidden state $X \in \mathbb{R}^{B \times S \times D}$ is projected into continuous coordinates $\mathbf{c} \in (0,1)^3$:
$$\mathbf{z} = X W_c + b_c, \quad W_c \in \mathbb{R}^{D \times 3}, \quad b_c \in \mathbb{R}^3$$
$$\mathbf{c} = \sigma(\mathbf{z})$$

To build an ultrametric tree, we extract digits $d_{i, k}$ at depth levels $k \in \{1, 2, \dots, \text{depth}\}$ using prime bases $p_0=2$, $p_1=3$, and $p_2=5$ (corresponding to prime factors of base-30).
Let $r_i^{(0)} = c_i$. For each depth level $k$:
$$d_{i, k} = \lfloor r_i^{(k-1)} \cdot p_i \rfloor$$
$$d_{i, k} = \operatorname{clamp}(d_{i, k}, 0, p_i - 1)$$
$$r_i^{(k)} = r_i^{(k-1)} \cdot p_i - d_{i, k}$$

These base-$p_i$ digits are interleaved at each level $k$ to yield a base-30 Morton code index:
$$d_{30, k} = d_{0, k} + 2 d_{1, k} + 6 d_{2, k}$$
$$M(X) = \sum_{k=1}^{\text{depth}} d_{30, k} \cdot 30^{\text{depth} - k}$$

The Morton code $M(X)$ uniquely positions the token within a hierarchical $p$-adic tree.

### 2.2 2-Adic Database Pruning
To retrieve stored KV states without brute-force searching the entire E8 coordinate-sparse database, the query's quantized coordinate $\mathbf{x}_{\text{quant}} \in \mathbb{R}^8$ is expanded using the 240 Shell 1 roots $\mathbf{r}_j$:
$$\mathbf{x}_{\text{cand}, j} = \mathbf{x}_{\text{quant}} + \mathbf{r}_j$$

We map coordinates $\mathbf{x} \in \mathbb{R}^8$ to dyadic coset representatives $\mathbb{F}_2^8$ at level 1:
$$\mathbf{y} = \lfloor 2 \mathbf{x} \rceil \pmod 2 \in \{0, 1\}^8$$
This dyadic coset is packed into an 8-bit integer coset ID:
$$\operatorname{Coset}(\mathbf{x}) = \sum_{n=0}^{7} y_n \cdot 2^n$$

The database lookup is pruned to candidates matching the query's candidate cosets:
$$\operatorname{Coset}(\mathbf{x}_{\text{db}}) \in \{ \operatorname{Coset}(\mathbf{x}_{\text{cand}, j}) \}_j$$

*Code Reference*: Digit extraction and Morton coding are implemented in [attention.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/modeling/attention.py#L1045-L1089) (`UltrametricAttention.forward`). The 2-adic database lookup pruning is implemented in [e8_swap.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/math/e8_swap.py#L756-L800) (`AdelicMemorySwapGridDB._swap_in`).

---

## 3. Closed-Form Orthogonal Procrustes Alignment

To align key-value representations between draft and target models (e.g., Gemma 2B and Gemma 9B) without gradient descent, we solve the orthogonal Procrustes problem.

### 3.1 Derivation
Given centered source hidden states $A \in \mathbb{R}^{N \times D_1}$ and centered target hidden states $B \in \mathbb{R}^{N \times D_2}$:
$$A = X_{\text{src}} - \mu_{\text{src}}, \quad B = X_{\text{tgt}} - \mu_{\text{tgt}}$$

We seek an orthogonal alignment matrix $M_{\text{align}} \in \mathbb{R}^{D_1 \times D_2}$ that minimizes the Frobenius norm of the mapping error:
$$\min_{M^T M = I} \| A M - B \|_F^2$$

Expanding the Frobenius norm:
$$\| A M - B \|_F^2 = \operatorname{Tr}((AM - B)^T(AM - B)) = \operatorname{Tr}(M^T A^T A M) - 2\operatorname{Tr}(M^T A^T B) + \operatorname{Tr}(B^T B)$$

Since $M$ is orthogonal, $\operatorname{Tr}(M^T A^T A M) = \operatorname{Tr}(A^T A)$, which is constant. Minimizing the error is equivalent to maximizing:
$$\max_{M^T M = I} \operatorname{Tr}(M^T A^T B)$$

Compute the Singular Value Decomposition (SVD) of the cross-covariance matrix $C = A^T B \in \mathbb{R}^{D_1 \times D_2}$:
$$C = U \Sigma V^T$$

Substituting into the trace term:
$$\operatorname{Tr}(M^T A^T B) = \operatorname{Tr}(M^T U \Sigma V^T) = \operatorname{Tr}(V^T M^T U \Sigma)$$

Let $Z = V^T M^T U$. Since $U, V, M$ are orthogonal, $Z$ is also orthogonal, meaning its diagonal elements satisfy $z_{ii} \le 1$. Because $\Sigma$ is diagonal with non-negative singular values $\sigma_i \ge 0$:
$$\operatorname{Tr}(Z \Sigma) = \sum_i z_{ii} \sigma_i \le \sum_i \sigma_i$$

Equality is achieved if and only if $Z = I$, which implies:
$$V^T M^T U = I \implies M^T = V U^T \implies M_{\text{align}} = U V^T$$

*Code Reference*: Implemented in [procrustes.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/math/procrustes.py#L6-L49) (`compute_procrustes_alignment`).

---

## 4. Woodbury-Optimized Cayley Layer Adapters

To bind multiple layers to a single swap database without VRAM blowouts, each layer uses a residual adapter $W_L = I + AB^T$ ($A, B \in \mathbb{R}^{D \times r}$) that preserves relative distances.

### 4.1 Skew-Symmetric Cayley Mapping
To guarantee that the adapter is strictly orthogonal ($W_L^T W_L = I$) and does not warp distances, we parameterize it using the Cayley transform of a skew-symmetric matrix $S$:
$$W_L = (I - S)(I + S)^{-1}$$

To enforce low-rank structure (rank $2r$), we construct $S$ using factor matrices $A, B \in \mathbb{R}^{D \times r}$ ($r=16$):
$$S = AB^T - BA^T$$

Since $S^T = (AB^T - BA^T)^T = BA^T - AB^T = -S$, $S$ is skew-symmetric by construction.

### 4.2 Woodbury Matrix Identity Optimization
Directly computing $(I + S)^{-1}$ involves inverting a $D \times D$ matrix, which has $O(D^3)$ computational complexity. For $D=2048$, this is too slow for real-time inference. We optimize this using the Woodbury matrix identity:
$$(I_D + U V^T)^{-1} = I_D - U(I_{2r} + V^T U)^{-1} V^T$$

We rewrite the skew-symmetric matrix $S = AB^T - BA^T$ as a product of two low-rank matrices $U, V \in \mathbb{R}^{D \times 2r}$:
$$U = [A \mid -B], \quad V = [B \mid A]$$
Checking the product:
$$U V^T = \begin{bmatrix} A & -B \end{bmatrix} \begin{bmatrix} B^T \\ A^T \end{bmatrix} = AB^T - BA^T = S$$

We substitute this factorization into the Cayley transform:
$$W_L = (I_D - U V^T)(I_D + U V^T)^{-1}$$

Applying the Woodbury identity to the inverse term:
$$(I_D + U V^T)^{-1} = I_D - U (I_{2r} + V^T U)^{-1} V^T$$

Now, expand the full adapter product:
$$W_L = (I_D - U V^T) \left[ I_D - U (I_{2r} + V^T U)^{-1} V^T \right]$$
$$W_L = I_D - U V^T - U (I_{2r} + V^T U)^{-1} V^T + U V^T U (I_{2r} + V^T U)^{-1} V^T$$

Factor out $U$ and $V^T$:
$$W_L = I_D - U \left[ I_{2r} + (I_{2r} - V^T U) (I_{2r} + V^T U)^{-1} \right] V^T$$

Let $M = V^T U \in \mathbb{R}^{2r \times 2r}$. The term inside brackets simplifies as:
$$I_{2r} + (I_{2r} - M)(I_{2r} + M)^{-1} = (I_{2r} + M)(I_{2r} + M)^{-1} + (I_{2r} - M)(I_{2r} + M)^{-1}$$
$$= \left[ (I_{2r} + M) + (I_{2r} - M) \right] (I_{2r} + M)^{-1} = 2 (I_{2r} + M)^{-1}$$

Substituting back yields the final Woodbury Cayley adapter equation:
$$W_L = I_D - 2 U (I_{2r} + V^T U)^{-1} V^T$$

This requires inverting a matrix of size $2r \times 2r$ ($32 \times 32$ for rank $r=16$), reducing complexity from $O(D^3)$ to $O(D \cdot r^2 + r^3)$, which is extremely fast and numerically stable.

*Code Reference*: Implemented in [attention.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/modeling/attention.py#L46-L92) (`cayley_orthogonal_adapter`).

---

## 5. Graph Laplacian & Fiedler Vector Context Bisection

The Čech Cohomology firewall calculates structural fracture boundaries in the attention matrix using Spectral Graph Theory.

### 5.1 Symmetric Normalized Laplacian
Given the attention skeleton matrix $A_{\text{skeleton}} \in \mathbb{R}^{K \times K}$ representing interactions between the top $K$ critical summits:
$$W = \frac{1}{2}(A_{\text{skeleton}} + A_{\text{skeleton}}^T)$$

The diagonal degree matrix $D \in \mathbb{R}^{K \times K}$ has elements:
$$d_{ii} = \sum_j W_{ij}$$

The graph Laplacian $L \in \mathbb{R}^{K \times K}$ is:
$$L = D - W$$

The second smallest eigenvalue $\lambda_2$ of $L$ is the **algebraic connectivity** of the graph. If $\lambda_2 < \tau$, the attention graph has fractured into disconnected components (indicating potential hallucination or adversarial steering).

### 5.2 Fiedler Vector Bisection
The eigenvector $v_2$ corresponding to $\lambda_2$ is the **Fiedler vector**. The signs of $v_2$ partition the vertices into two maximally disconnected subgraphs:
$$G_{\text{pos}} = \{ i \mid v_2[i] \ge 0 \}, \quad G_{\text{neg}} = \{ i \mid v_2[i] < 0 \}$$

The boundary index of the split is determined by finding the boundary coordinate:
$$\text{boundary} = \begin{cases} 
\min(G_{\text{neg}}) & \text{if } \min(G_{\text{pos}}) < \min(G_{\text{neg}}) \\
\min(G_{\text{pos}}) & \text{otherwise}
\end{cases}$$

This boundary represents the exact sequence position where the semantic fracture occurred, allowing targeted token generation rollbacks.

*Code Reference*: Implemented in [cohomology.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/firewall/cohomology.py#L23-L247) (`check_obstruction`).

---

## 6. Adelic Langevin Optimization

To guarantee training stability on coordinate-sparse grids, weight parameters are optimized by combining continuous (Archimedean) gradients with $p$-adic tree-space leaps.

### 6.1 $p$-Adic Vladimirov Fractional Derivative
The Adelic Langevin update integrates non-Archimedean optimization. The Vladimirov fractional derivative of a function $f$ over the $p$-adic field $\mathbb{Q}_p$ is:
$$\left(D^\alpha f\right)(x) = \frac{p^\alpha - 1}{1 - p^{-\alpha-1}} \int_{\mathbb{Q}_p} \frac{f(x) - f(y)}{\|x - y\|_p^{\alpha + 1}} dy$$

For dyadic multiscale history compression ($\alpha = 1$, $p = 2$), the gradient updates perform discrete tunneling steps, enabling parameters to leap out of narrow local minima without loss divergence.

*Code Reference*: Implemented in [adelic.py](file:///Volumes/Storage/project_atlas_unified/qan_transformers/optim/adelic.py#L103-L308) (`AdelicLangevinOptimizer.step`).
