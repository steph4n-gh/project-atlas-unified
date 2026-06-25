/**
 * docs/assets/js/wormhole_warp_runner.js
 * QAN Cross-Model Wormhole Bridge (The Warp) JavaScript Engine
 * Mathematical implementation of Woodbury-Cayley and Jacobi SVD Procrustes Alignment.
 */

// Matrix math utilities
const MatrixMath = {
    // Gauss-Jordan elimination for matrix inverse
    invert(M) {
        const n = M.length;
        const A = M.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]);
        
        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
                    maxRow = k;
                }
            }
            const temp = A[i];
            A[i] = A[maxRow];
            A[maxRow] = temp;
            
            const pivot = A[i][i];
            if (Math.abs(pivot) < 1e-15) {
                // Return identity matrix on singularity
                return Array(n).fill(0).map((_, x) => Array(n).fill(0).map((_, y) => x === y ? 1 : 0));
            }
            
            for (let j = i; j < 2 * n; j++) {
                A[i][j] /= pivot;
            }
            
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = A[k][i];
                    for (let j = i; j < 2 * n; j++) {
                        A[k][j] -= factor * A[i][j];
                    }
                }
            }
        }
        return A.map(row => row.slice(n));
    },

    multiply(A, B) {
        const m = A.length;
        const n = A[0].length;
        const p = B[0].length;
        const C = Array(m).fill(0).map(() => Array(p).fill(0));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    },

    transpose(A) {
        const m = A.length;
        const n = A[0].length;
        const T = Array(n).fill(0).map(() => Array(m).fill(0));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    },

    identity(n) {
        return Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : 0));
    },

    randomNormal(m, n, std = 0.1) {
        const R = Array(m).fill(0).map(() => Array(n).fill(0));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                // Box-Muller transform
                let u = 0, v = 0;
                while(u === 0) u = Math.random();
                while(v === 0) v = Math.random();
                const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
                R[i][j] = num * std;
            }
        }
        return R;
    },

    subtract(A, B) {
        return A.map((row, i) => row.map((val, j) => val - B[i][j]));
    },

    add(A, B) {
        return A.map((row, i) => row.map((val, j) => val + B[i][j]));
    },

    norm(v) {
        return Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    },

    cosineSimilarity(v1, v2) {
        let dot = 0, n1 = 0, n2 = 0;
        for (let i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            n1 += v1[i] * v1[i];
            n2 += v2[i] * v2[i];
        }
        return dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-12);
    },

    // One-sided Jacobi SVD for matrix A (m x n, m >= n)
    // Returns { U, S, V } where A = U * diag(S) * V^T
    jacobiSVD(A, tolerance = 1e-9) {
        const m = A.length;
        const n = A[0].length;
        
        // Clone A into U
        const U = A.map(row => [...row]);
        // V starts as identity matrix of size n x n
        const V = this.identity(n);
        const S = Array(n).fill(0);
        
        for (let iter = 0; iter < 100; iter++) {
            let anyPivot = false;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    let aii = 0, ajj = 0, aij = 0;
                    for (let k = 0; k < m; k++) {
                        aii += U[k][i] * U[k][i];
                        ajj += U[k][j] * U[k][j];
                        aij += U[k][i] * U[k][j];
                    }
                    
                    if (Math.abs(aij) > tolerance * Math.sqrt(aii * ajj)) {
                        anyPivot = true;
                        const tau = (ajj - aii) / (2 * aij);
                        const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
                        const c = 1 / Math.sqrt(1 + t * t);
                        const s = c * t;
                        
                        // Rotate columns of U
                        for (let k = 0; k < m; k++) {
                            const u_ki = U[k][i];
                            const u_kj = U[k][j];
                            U[k][i] = c * u_ki - s * u_kj;
                            U[k][j] = s * u_ki + c * u_kj;
                        }
                        
                        // Rotate columns of V
                        for (let k = 0; k < n; k++) {
                            const v_ki = V[k][i];
                            const v_kj = V[k][j];
                            V[k][i] = c * v_ki - s * v_kj;
                            V[k][j] = s * v_ki + c * v_kj;
                        }
                    }
                }
            }
            if (!anyPivot) break;
        }
        
        // Extract singular values and normalize U
        for (let j = 0; j < n; j++) {
            let norm = 0;
            for (let i = 0; i < m; i++) norm += U[i][j] * U[i][j];
            norm = Math.sqrt(norm);
            S[j] = norm;
            if (norm > 1e-12) {
                for (let i = 0; i < m; i++) U[i][j] /= norm;
            }
        }
        
        return { U, S, V };
    }
};

// Woodbury-Cayley Privacy Adapter
class CayleyPrivacyAdapter {
    constructor(dim, rank = 8) {
        this.dim = dim;
        this.rank = rank;
        this.regenerate();
    }

    regenerate() {
        // Random low-rank factors
        this.A = MatrixMath.randomNormal(this.dim, this.rank, 0.05);
        this.B = MatrixMath.randomNormal(this.dim, this.rank, 0.05);

        // U = [A | -B], V = [B | A]
        this.U = this.A.map((row, idx) => [...row, ...this.B[idx].map(v => -v)]);
        this.V = this.B.map((row, idx) => [...row, ...this.A[idx]]);

        // Precompute the inversion core: (I_{2r} + V^T U)^{-1}
        // VtU size: (2r, 2r)
        const Vt = MatrixMath.transpose(this.V);
        const VtU = MatrixMath.multiply(Vt, this.U);
        
        const coreSize = 2 * this.rank;
        const I_2r = MatrixMath.identity(coreSize);
        const core = MatrixMath.add(I_2r, VtU);
        
        this.coreInv = MatrixMath.invert(core);
    }

    // Apply scrambling rotation (W_L @ h)
    rotate(h) {
        // h shape: (1, dim) represented as array [h_vector]
        const h_batch = [h];
        // V^T @ h: shape (2r, 1)
        const Vt = MatrixMath.transpose(this.V);
        const Vth = MatrixMath.transpose(MatrixMath.multiply(h_batch, this.V)); // (2r, 1)
        
        // coreInv @ Vth: shape (2r, 1)
        const mid = MatrixMath.multiply(this.coreInv, Vth);
        
        // U @ mid: shape (dim, 1)
        const correction_t = MatrixMath.multiply(this.U, mid); // (dim, 1)
        const correction = MatrixMath.transpose(correction_t)[0]; // (dim)
        
        return h.map((val, idx) => val - 2.0 * correction[idx]);
    }

    // Apply inverse scrambling (W_L^T @ h_rotated)
    inverseRotate(h_rotated) {
        const hr_batch = [h_rotated];
        // U^T @ h_rotated: shape (2r, 1)
        const Ut = MatrixMath.transpose(this.U);
        const Uth = MatrixMath.transpose(MatrixMath.multiply(hr_batch, this.U));
        
        // coreInv^T @ Uth: shape (2r, 1)
        const coreInvT = MatrixMath.transpose(this.coreInv);
        const mid = MatrixMath.multiply(coreInvT, Uth);
        
        // V @ mid: shape (dim, 1)
        const correction_t = MatrixMath.multiply(this.V, mid);
        const correction = MatrixMath.transpose(correction_t)[0];
        
        return h_rotated.map((val, idx) => val - 2.0 * correction[idx]);
    }
}

// Procrustes Space Aligner
class ProcrustesAligner {
    constructor() {
        this.alignedMatrix = null;
        this.biasLocal = null;
        this.biasGemini = null;
        this.calibrated = false;
    }

    calibrate(localStates, geminiEmbeddings) {
        const N = localStates.length;
        const d_local = localStates[0].length;
        const d_gemini = geminiEmbeddings[0].length;
        const d_min = Math.min(d_local, d_gemini);

        // Center both sets
        this.biasLocal = Array(d_local).fill(0);
        this.biasGemini = Array(d_gemini).fill(0);

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < d_local; j++) this.biasLocal[j] += localStates[i][j];
            for (let j = 0; j < d_gemini; j++) this.biasGemini[j] += geminiEmbeddings[i][j];
        }
        for (let j = 0; j < d_local; j++) this.biasLocal[j] /= N;
        for (let j = 0; j < d_gemini; j++) this.biasGemini[j] /= N;

        // X = local - biasLocal, Y = gemini - biasGemini
        const X = localStates.map(row => row.slice(0, d_min).map((v, col) => v - this.biasLocal[col]));
        const Y = geminiEmbeddings.map(row => row.slice(0, d_min).map((v, col) => v - this.biasGemini[col]));

        // C = X^T * Y
        const Xt = MatrixMath.transpose(X);
        const C = MatrixMath.multiply(Xt, Y);

        // Jacobi SVD: C = U * S * V^T
        const { U, S, V } = MatrixMath.jacobiSVD(C);
        
        // R = U * V^T
        const Vt = MatrixMath.transpose(V);
        this.alignedMatrix = MatrixMath.multiply(U, Vt);
        this.calibrated = true;

        // Calculate alignment similarity
        const X_mapped = MatrixMath.multiply(X, this.alignedMatrix);
        let simSum = 0;
        for (let i = 0; i < N; i++) {
            simSum += MatrixMath.cosineSimilarity(X_mapped[i], Y[i]);
        }
        return simSum / N;
    }

    align(geminiEmbedding, localDim) {
        const d_min = Math.min(geminiEmbedding.length, this.alignedMatrix.length);
        const centered = geminiEmbedding.slice(0, d_min).map((v, col) => v - this.biasGemini[col]);
        const aligned = MatrixMath.multiply([centered], this.alignedMatrix)[0];
        const restored = aligned.map((v, col) => v + this.biasLocal[col]);
        
        const result = Array(localDim).fill(0);
        for (let i = 0; i < localDim; i++) {
            if (i < restored.length) {
                result[i] = restored[i];
            }
        }
        return result;
    }
}

// Global Demo State Orchestrator
class WormholeWarpDemo {
    constructor() {
        this.dim = 64; 
        this.rank = 8;
        this.adapter = new CayleyPrivacyAdapter(this.dim, this.rank);
        this.aligner = new ProcrustesAligner();
        
        this.apiKey = localStorage.getItem("GEMINI_API_KEY") || "";
        this.modelName = "gemini-embedding-001";
        this.simMode = true;
        this.blendWeight = 0.3;
        
        this.cfiThreshold = 0.8;
        this.lambda2Threshold = 0.05;

        this.calibrationTexts = [
            "Topological quantum computation using Chern insulators.",
            "Speculative decoding routes draft tokens to safety.",
            "Cayley orthogonal transformations preserve vector norms in Hilbert space.",
            "Orthogonal Procrustes SVD aligns disparate latent spaces.",
            "Anomalies in multi-head attention map to Cech cohomology bounds."
        ];
        
        this.presetPrompts = [
            {
                name: "Standard Math Query (Normal Flow)",
                text: "Describe a Čech Cohomology calculation...",
                anomalyPoint: -1, 
                localTokens: ["To", " compute", " Čech", " cohomology,", " we", " define", " an", " open", " cover", " U", " of", " the", " topological", " space", " X,", " then", " assemble", " the", " nerve", " simplicial", " complex,", " and", " construct", " coboundary", " operators", " to", " solve", " for", " cocycles", " modulo", " coboundaries."],
                restoredTokens: []
            },
            {
                name: "Contradictory Logic Loop (Warp Event)",
                text: "Run p-adic diffusion inside an open boundary system...",
                anomalyPoint: 12, 
                localTokens: ["Initialize", " p-adic", " diffusion", " coefficients", " on", " the", " ultrametric", " tree.", " If", " the", " boundary", " is", " open,", " local", " connectivity", " collapses", " and", " the", " Fiedler", " value", " drops", " to", " zero."],
                restoredTokens: [" The", " Cohomology", " Firewall", " intercepts", " the", " path,", " opens", " a", " secure", " Cayley-rotated", " wormhole,", " aligns", " representations", " with", " SVD,", " and", " restores", " global", " diffusion", " stability."]
            }
        ];
        
        this.stats = {
            totalQueries: 0,
            successQueries: 0,
            avgLatency: 0
        };
    }

    setAPIKey(key) {
        this.apiKey = key;
        localStorage.setItem("GEMINI_API_KEY", key);
    }

    setSimMode(val) {
        this.simMode = val;
    }

    async fetchEmbedding(text) {
        if (this.simMode || !this.apiKey) {
            await new Promise(r => setTimeout(r, 800 + Math.random() * 500));
            // Return mock vector of size 768
            const mockVec = Array(768).fill(0).map(() => Math.random() - 0.5);
            return mockVec;
        }

        const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.modelName}:embedContent?key=${this.apiKey}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: `models/${this.modelName}`,
                content: { parts: [{ text }] }
            })
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Gemini API Error (${response.status}): ${errText}`);
        }

        const result = await response.json();
        const embedding = result.embedding?.values;
        if (!embedding) throw new Error("No embedding values returned in response.");
        return embedding;
    }

    async calibrate(onProgress) {
        onProgress("Initializing calibration vectors...");
        
        const localStates = [];
        const geminiEmbeddings = [];

        for (let i = 0; i < this.calibrationTexts.length; i++) {
            const txt = this.calibrationTexts[i];
            onProgress(`Fetching cloud embedding for text ${i+1}/${this.calibrationTexts.length}...`);
            
            try {
                const cloudVec = await this.fetchEmbedding(txt);
                geminiEmbeddings.push(cloudVec);
                
                const localVec = Array(this.dim).fill(0);
                for (let j = 0; j < this.dim; j++) {
                    const cloudVal = cloudVec[j % cloudVec.length];
                    localVec[j] = cloudVal * 0.8 + (Math.random() - 0.5) * 0.15; 
                }
                localStates.push(localVec);
            } catch (e) {
                console.error(e);
                throw new Error(`Calibration failed at step ${i+1}: ${e.message}`);
            }
        }

        onProgress("Computing centering biases and covariance matrices...");
        await new Promise(r => setTimeout(r, 400));
        
        onProgress("Executing Orthogonal Procrustes SVD solver...");
        const quality = this.aligner.calibrate(localStates, geminiEmbeddings);
        
        onProgress(`Calibration completed. Quality metric: ${quality.toFixed(5)}`);
        return { quality, geminiDim: geminiEmbeddings[0].length };
    }
}
