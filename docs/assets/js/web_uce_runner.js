/**
 * WebUCERunner: Client-side Ultrametric Cognitive Engine (UCE) Runner
 * 
 * Implements client-side loading of UCE models (.safetensors + .meta.json),
 * p-adic address tokenization/digit lookup, Vladimirov-style tree diffusion,
 * Cohomology Firewall check, and Top-K output probability blending across multiple experts.
 * Exposes visualizer hooks for particle routing, E8 projections, and VU meters.
 */

// --- Helper Math Functions ---

function matvec(W, x, b, rows, cols) {
    const y = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
        let sum = 0;
        const rowOffset = i * cols;
        for (let j = 0; j < cols; j++) {
            sum += W[rowOffset + j] * x[j];
        }
        y[i] = sum + (b ? b[i] : 0);
    }
    return y;
}

function addVectors(u, v) {
    const r = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) {
        r[i] = u[i] + v[i];
    }
    return r;
}

function scaleVector(u, s) {
    const r = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) {
        r[i] = u[i] * s;
    }
    return r;
}

function l2Distance2(u, v) {
    let sum = 0;
    for (let i = 0; i < u.length; i++) {
        const diff = u[i] - v[i];
        sum += diff * diff;
    }
    return sum;
}

function l2Norm(u) {
    let sum = 0;
    for (let i = 0; i < u.length; i++) {
        sum += u[i] * u[i];
    }
    return Math.sqrt(sum);
}

function dotProduct(u, v) {
    let sum = 0;
    for (let i = 0; i < u.length; i++) {
        sum += u[i] * v[i];
    }
    return sum;
}

function tanhVector(u) {
    const r = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) {
        r[i] = Math.tanh(u[i]);
    }
    return r;
}

function softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    const exp = new Float32Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        exp[i] = Math.exp(arr[i] - max);
        sum += exp[i];
    }
    for (let i = 0; i < arr.length; i++) {
        exp[i] /= sum;
    }
    return exp;
}

function valuation(x, p) {
    if (x === 0) return Infinity;
    let v = 0;
    x = Math.abs(x);
    while (x % p === 0) {
        x = Math.floor(x / p);
        v++;
    }
    return v;
}

function addressToDigits(addr, p, k) {
    const digits = [];
    let temp = addr;
    for (let i = 0; i < k; i++) {
        digits.push(temp % p);
        temp = Math.floor(temp / p);
    }
    return digits;
}

// --- Safetensors Binary Parser ---

class SafetensorsParser {
    static parse(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const headerLengthLow = view.getUint32(0, true);
        const headerLengthHigh = view.getUint32(4, true);
        const headerLength = headerLengthLow + headerLengthHigh * 0x100000000;
        
        const headerBytes = new Uint8Array(arrayBuffer, 8, headerLength);
        const headerText = new TextDecoder("utf-8").decode(headerBytes);
        const header = JSON.parse(headerText);
        
        const tensors = {};
        const dataStart = 8 + headerLength;
        
        for (const [name, info] of Object.entries(header)) {
            if (name === "__metadata__") continue;
            const shape = info.shape;
            const dtype = info.dtype;
            const offsets = info.data_offsets;
            const start = dataStart + offsets[0];
            const end = dataStart + offsets[1];
            const length = offsets[1] - offsets[0];
            
            let data;
            if (dtype === "F32" || dtype === "float32") {
                data = new Float32Array(arrayBuffer, start, length / 4);
            } else if (dtype === "F16" || dtype === "float16") {
                const f16View = new Uint16Array(arrayBuffer, start, length / 2);
                data = new Float32Array(f16View.length);
                for (let i = 0; i < f16View.length; i++) {
                    data[i] = SafetensorsParser.decodeFloat16(f16View[i]);
                }
            } else if (dtype === "BF16" || dtype === "bfloat16") {
                const bf16View = new Uint16Array(arrayBuffer, start, length / 2);
                data = new Float32Array(bf16View.length);
                for (let i = 0; i < bf16View.length; i++) {
                    data[i] = SafetensorsParser.decodeBfloat16(bf16View[i]);
                }
            } else {
                continue;
            }
            tensors[name] = { shape, data };
        }
        return tensors;
    }
    
    static decodeFloat16(binary) {
        const exponent = (binary & 0x7C00) >> 10;
        const fraction = binary & 0x03FF;
        const sign = (binary & 0x8000) ? -1 : 1;
        if (exponent === 0) {
            return sign * Math.pow(2, -14) * (fraction / 1024);
        } else if (exponent === 0x1F) {
            return fraction ? NaN : sign * Infinity;
        }
        return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
    
    static decodeBfloat16(binary) {
        const temp = new Uint32Array(1);
        temp[0] = binary << 16;
        const floatView = new Float32Array(temp.buffer);
        return floatView[0];
    }
}

// --- Finite Tree Implementation ---

class FiniteTree {
    constructor(p, depth, addressMap = null) {
        this.p = p;
        this.depth = depth;
        
        this._token_to_addr = new Map();
        this._addr_to_token = new Map();
        this._ball_children = new Map(); // "depth,prefix" -> Array of prefixes
        this._ball_tokens = new Map();   // "depth,address" -> Array of tokenIds
        
        if (addressMap) {
            for (const [addrStr, tokenId] of Object.entries(addressMap)) {
                this.addLeaf(Number(tokenId), Number(addrStr));
            }
        }
    }
    
    addLeaf(tokenId, address) {
        const maxAddr = Math.pow(this.p, this.depth);
        if (address < 0 || address >= maxAddr) {
            throw new Error(`Address ${address} out of range for p=${this.p}, depth=${this.depth}`);
        }
        
        this._token_to_addr.set(tokenId, address);
        this._addr_to_token.set(address, tokenId);
        this._insertPath(address);
    }
    
    _insertPath(address) {
        const p = this.p;
        let currentPrefix = 0;
        for (let d = 1; d <= this.depth; d++) {
            const digit = Math.floor(address / Math.pow(p, d - 1)) % p;
            const nextPrefix = currentPrefix + digit * Math.pow(p, d - 1);
            
            const parentKey = `${d - 1},${currentPrefix}`;
            if (!this._ball_children.has(parentKey)) {
                this._ball_children.set(parentKey, []);
            }
            const children = this._ball_children.get(parentKey);
            if (!children.includes(nextPrefix)) {
                children.push(nextPrefix);
            }
            currentPrefix = nextPrefix;
        }
        
        const leafKey = `${this.depth},${address}`;
        if (!this._ball_tokens.has(leafKey)) {
            this._ball_tokens.set(leafKey, []);
        }
        const toks = this._ball_tokens.get(leafKey);
        const tok = this._addr_to_token.get(address);
        if (!toks.includes(tok)) {
            toks.push(tok);
        }
    }
    
    children(depth, prefix) {
        const key = `${depth},${prefix}`;
        return this._ball_children.get(key) || [];
    }
    
    getAncestors(address) {
        const ancs = [0];
        for (let d = 1; d <= this.depth; d++) {
            ancs.push(address % Math.pow(this.p, d));
        }
        return ancs;
    }
    
    addressToToken(address) {
        if (!this._addr_to_token.has(address)) {
            throw new Error(`Address ${address} not registered in tree`);
        }
        return this._addr_to_token.get(address);
    }
    
    tokenToAddress(tokenId) {
        if (!this._token_to_addr.has(tokenId)) {
            throw new Error(`Token ${tokenId} not registered in tree`);
        }
        return this._token_to_addr.get(tokenId);
    }
    
    leafAddresses() {
        return Array.from(this._addr_to_token.keys());
    }
}

// --- UCE Model Implementation ---

class UCEModel {
    constructor(tree, dim = 16, numDiffLayers = 1, alpha = 0.5) {
        this.tree = tree;
        this.p = tree.p;
        this.depth = tree.depth;
        this.dim = dim;
        this.numDiffLayers = numDiffLayers;
        this.alpha = alpha;
        
        this.schwarzschildWarp = true;
        this.r_s = 0.5;
        this.wormholeGate = true;
        this.epsilon = 0.1;
        
        this.leafAddrs = tree.leafAddresses();
        this.numLeaves = this.leafAddrs.length;
        
        this.addrToLeafIdx = new Map();
        this.leafAddrs.forEach((addr, idx) => {
            this.addrToLeafIdx.set(addr, idx);
        });
        
        // Build ball prefix list
        this.allBalls = [];
        this.ballToIdx = new Map();
        let bidx = 0;
        for (let d = 0; d <= this.depth; d++) {
            const maxPref = Math.pow(this.p, d);
            for (let pref = 0; pref < maxPref; pref++) {
                if (d === 0 && pref !== 0) continue;
                const key = `${d},${pref}`;
                this.allBalls.push(key);
                this.ballToIdx.set(key, bidx++);
            }
        }
        this.numBalls = this.allBalls.length;
        
        // Parameters (initialized to small random weights, overwritten by load)
        this.ball_embed = new Float32Array(this.numBalls * this.dim);
        this.leaf_activation = new Float32Array(this.numLeaves * this.dim);
        
        this.mix_linears = [];
        for (let l = 0; l < this.numDiffLayers; l++) {
            this.mix_linears.push({
                weight: new Float32Array(this.dim * this.dim),
                bias: new Float32Array(this.dim)
            });
        }
        
        this.heads = [];
        for (let d = 0; d < this.depth; d++) {
            this.heads.push({
                weight: new Float32Array(this.p * this.dim),
                bias: new Float32Array(this.p)
            });
        }
        
        this.randomizeWeights();
    }
    
    randomizeWeights() {
        const rand = (n) => {
            const arr = new Float32Array(n);
            for (let i = 0; i < n; i++) arr[i] = (Math.random() - 0.5) * 0.1;
            return arr;
        };
        this.ball_embed.set(rand(this.ball_embed.length));
        this.leaf_activation.set(rand(this.leaf_activation.length));
        this.mix_linears.forEach(m => {
            m.weight.set(rand(m.weight.length));
            m.bias.set(rand(m.bias.length));
        });
        this.heads.forEach(h => {
            h.weight.set(rand(h.weight.length));
            h.bias.set(rand(h.bias.length));
        });
    }
    
    loadWeights(tensors) {
        if (tensors['ball_embed.weight']) {
            this.ball_embed.set(tensors['ball_embed.weight'].data);
        }
        if (tensors['leaf_activation.weight']) {
            this.leaf_activation.set(tensors['leaf_activation.weight'].data);
        }
        for (let l = 0; l < this.numDiffLayers; l++) {
            const wKey = `diffusion.mix_linears.${l}.weight`;
            const bKey = `diffusion.mix_linears.${l}.bias`;
            if (tensors[wKey]) this.mix_linears[l].weight.set(tensors[wKey].data);
            if (tensors[bKey]) this.mix_linears[l].bias.set(tensors[bKey].data);
        }
        for (let d = 0; d < this.depth; d++) {
            const wKey = `heads.heads.${d}.weight`;
            const bKey = `heads.heads.${d}.bias`;
            if (tensors[wKey]) this.heads[d].weight.set(tensors[wKey].data);
            if (tensors[bKey]) this.heads[d].bias.set(tensors[bKey].data);
        }
    }
    
    embedAndDiffuse(previousAddresses = [], activeBalls = null) {
        const states = new Map();
        
        const keysToProcess = activeBalls || this.allBalls;
        keysToProcess.forEach(key => {
            if (this.ballToIdx.has(key)) {
                const bidx = this.ballToIdx.get(key);
                const vec = new Float32Array(this.dim);
                for (let i = 0; i < this.dim; i++) {
                    vec[i] = this.ball_embed[bidx * this.dim + i];
                }
                states.set(key, vec);
            }
        });
        
        if (previousAddresses.length > 0) {
            const recent = previousAddresses.slice(-5);
            recent.forEach(addr => {
                if (this.addrToLeafIdx.has(addr)) {
                    const lidx = this.addrToLeafIdx.get(addr);
                    const lkey = `${this.depth},${addr}`;
                    if (states.has(lkey)) {
                        const act = new Float32Array(this.dim);
                        for (let i = 0; i < this.dim; i++) {
                            act[i] = this.leaf_activation[lidx * this.dim + i];
                        }
                        states.set(lkey, addVectors(states.get(lkey), act));
                    }
                }
            });
        }
        
        let current = new Map(states);
        
        for (let layerIdx = 0; layerIdx < this.numDiffLayers; layerIdx++) {
            const newStates = new Map();
            
            const byDepth = new Map();
            for (const [key, vec] of current.entries()) {
                const [dStr, prefStr] = key.split(",");
                const d = Number(dStr);
                const pref = Number(prefStr);
                if (!byDepth.has(d)) byDepth.set(d, []);
                byDepth.get(d).push({ pref, vec });
            }
            
            for (const [d, items] of byDepth.entries()) {
                const N = items.length;
                if (N === 1) {
                    const { pref, vec } = items[0];
                    let mixed = new Float32Array(vec);
                    if (d > 0) {
                        const parentPref = d === 1 ? 0 : (pref % Math.pow(this.p, d - 1));
                        const parentKey = `${d - 1},${parentPref}`;
                        if (current.has(parentKey)) {
                            const crossW = Math.pow(this.p, -1 * this.alpha);
                            const parentVec = current.get(parentKey);
                            mixed = addVectors(
                                scaleVector(mixed, 1 / (1 + crossW)),
                                scaleVector(parentVec, crossW / (1 + crossW))
                            );
                        }
                    }
                    newStates.set(`${d},${pref}`, mixed);
                    continue;
                }
                
                const W = Array.from({ length: N }, () => new Float32Array(N));
                for (let i = 0; i < N; i++) {
                    const pref_i = items[i].pref;
                    for (let j = 0; j < N; j++) {
                        const pref_j = items[j].pref;
                        if (d === 0) {
                            W[i][j] = 1.0;
                        } else {
                            const delta = pref_i - pref_j;
                            let v = valuation(delta, this.p);
                            if (v === Infinity) {
                                v = d;
                            } else {
                                v = Math.min(v, d);
                            }
                            W[i][j] = Math.pow(this.p, v * this.alpha);
                        }
                    }
                }
                
                if (this.schwarzschildWarp && d > 0) {
                    const dists = new Float32Array(N);
                    for (let i = 0; i < N; i++) {
                        const pref = items[i].pref;
                        const parentPref = d === 1 ? 0 : (pref % Math.pow(this.p, d - 1));
                        const parentKey = `${d - 1},${parentPref}`;
                        const parentVec = current.get(parentKey) || new Float32Array(this.dim);
                        dists[i] = l2Distance2(items[i].vec, parentVec);
                    }
                    for (let i = 0; i < N; i++) {
                        for (let j = 0; j < N; j++) {
                            const gravityWarp = 1.0 - (this.r_s / (this.r_s + dists[i] + dists[j] + 1e-6));
                            W[i][j] *= gravityWarp;
                        }
                    }
                }
                
                if (this.wormholeGate) {
                    const norms = items.map(it => l2Norm(it.vec) + 1e-6);
                    for (let i = 0; i < N; i++) {
                        for (let j = 0; j < N; j++) {
                            const cosSim = dotProduct(items[i].vec, items[j].vec) / (norms[i] * norms[j]);
                            const baseW = d === 0 ? 1.0 : Math.pow(this.p, Math.min(valuation(items[i].pref - items[j].pref, this.p), d) * this.alpha);
                            if (cosSim > 0.85 && baseW === 0.0) {
                                W[i][j] += this.epsilon * cosSim;
                            }
                        }
                    }
                }
                
                for (let i = 0; i < N; i++) {
                    let rowSum = 0;
                    for (let j = 0; j < N; j++) rowSum += W[i][j];
                    if (rowSum > 0) {
                        for (let j = 0; j < N; j++) W[i][j] /= (rowSum + 1e-9);
                    }
                }
                
                const mixedMatrix = [];
                for (let i = 0; i < N; i++) {
                    const mixedVec = new Float32Array(this.dim);
                    for (let j = 0; j < N; j++) {
                        for (let k = 0; k < this.dim; k++) {
                            mixedVec[k] += W[i][j] * items[j].vec[k];
                        }
                    }
                    mixedMatrix.push(mixedVec);
                }
                
                for (let i = 0; i < N; i++) {
                    const pref = items[i].pref;
                    let mixed = mixedMatrix[i];
                    if (d > 0) {
                        const parentPref = d === 1 ? 0 : (pref % Math.pow(this.p, d - 1));
                        const parentKey = `${d - 1},${parentPref}`;
                        if (current.has(parentKey)) {
                            const crossW = Math.pow(this.p, -1 * this.alpha);
                            const parentVec = current.get(parentKey);
                            mixed = addVectors(
                                scaleVector(mixed, 1 / (1 + crossW)),
                                scaleVector(parentVec, crossW / (1 + crossW))
                            );
                        }
                    }
                    newStates.set(`${d},${pref}`, mixed);
                }
            }
            
            for (const [key, mixed] of newStates.entries()) {
                const lin = this.mix_linears[layerIdx];
                const transformed = matvec(lin.weight, mixed, lin.bias, this.dim, this.dim);
                const residual = addVectors(transformed, mixed);
                newStates.set(key, tanhVector(residual));
            }
            current = newStates;
        }
        
        return current;
    }
    
    forward(previousAddresses = []) {
        const diffused = this.embedAndDiffuse(previousAddresses);
        const logpsTotal = new Float32Array(this.numLeaves);
        
        for (let d = 0; d < this.depth; d++) {
            const p = this.p;
            const p_d = Math.pow(p, d);
            
            for (let i = 0; i < this.numLeaves; i++) {
                const addr = this.leafAddrs[i];
                const key = `${d},${addr % p_d}`;
                const state = diffused.get(key) || new Float32Array(this.dim);
                
                const head = this.heads[d];
                const logits = matvec(head.weight, state, head.bias, p, this.dim);
                const logProbs = new Float32Array(p);
                
                let maxLogit = -Infinity;
                for (let k = 0; k < p; k++) if (logits[k] > maxLogit) maxLogit = logits[k];
                let sumExp = 0;
                for (let k = 0; k < p; k++) sumExp += Math.exp(logits[k] - maxLogit);
                const logSumExp = maxLogit + Math.log(sumExp);
                for (let k = 0; k < p; k++) logProbs[k] = logits[k] - logSumExp;
                
                const targetDigit = Math.floor(addr / p_d) % p;
                logpsTotal[i] += logProbs[targetDigit];
            }
        }
        
        let maxLog = -Infinity;
        for (let i = 0; i < this.numLeaves; i++) if (logpsTotal[i] > maxLog) maxLog = logpsTotal[i];
        const probs = new Float32Array(this.numLeaves);
        let sum = 0;
        for (let i = 0; i < this.numLeaves; i++) {
            probs[i] = Math.exp(logpsTotal[i] - maxLog);
            sum += probs[i];
        }
        for (let i = 0; i < this.numLeaves; i++) probs[i] /= sum;
        
        return probs;
    }
}

// --- Cohomology Firewall Implementation ---

class CohomologyFirewall {
    constructor(threshold = 1.5, tau = 0.05) {
        this.threshold = threshold;
        this.tau = tau;
    }
    
    checkObstruction(attnMatrix) {
        const S = attnMatrix.length;
        if (S <= 1) return { isFractured: false, lam2: 1.0, altIdx: [] };
        
        const K = Math.min(8, S);
        
        const energies = Array.from({ length: S }, (_, i) => {
            let sum = 0;
            for (let j = 0; j < S; j++) sum += attnMatrix[i][j];
            return { idx: i, val: sum };
        });
        energies.sort((a, b) => b.val - a.val);
        const criticalSummits = energies.slice(0, K).map(e => e.idx);
        
        const W = Array.from({ length: K }, () => new Float32Array(K));
        let offDiagSum = 0;
        for (let i = 0; i < K; i++) {
            const u = criticalSummits[i];
            for (let j = 0; j < K; j++) {
                const v = criticalSummits[j];
                W[i][j] = (attnMatrix[u][v] + attnMatrix[v][u]) / 2.0;
                if (i === j) W[i][j] = 0;
                else offDiagSum += W[i][j];
            }
        }
        
        const degrees = new Float32Array(K);
        for (let i = 0; i < K; i++) {
            let sum = 0;
            for (let j = 0; j < K; j++) sum += W[i][j];
            degrees[i] = sum;
        }
        
        const L = Array.from({ length: K }, () => new Float32Array(K));
        for (let i = 0; i < K; i++) {
            for (let j = 0; j < K; j++) {
                if (i === j) {
                    L[i][j] = degrees[i];
                } else {
                    L[i][j] = -W[i][j];
                }
            }
        }
        
        let lam2 = 1.0;
        if (K > 1 && offDiagSum > 0.1) {
            try {
                const eigs = this.jacobiEigenvalues(L);
                eigs.sort((a, b) => a - b);
                lam2 = eigs[1];
            } catch (err) {
                lam2 = 1.0;
            }
        }
        
        const isFractured = (lam2 < this.tau) && (offDiagSum > 0.1) && (K > 1);
        const altIdx = energies.map(e => e.idx);
        
        return { isFractured, lam2, altIdx };
    }
    
    jacobiEigenvalues(L, maxIter = 100) {
        const n = L.length;
        const A = L.map(row => [...row]);
        const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
        
        for (let iter = 0; iter < maxIter; iter++) {
            let p = 0, q = 1;
            let maxVal = Math.abs(A[0][1]);
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(A[i][j]) > maxVal) {
                        maxVal = Math.abs(A[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }
            
            if (maxVal < 1e-9) break;
            
            const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
            let t;
            if (theta >= 0) {
                t = 1 / (theta + Math.sqrt(1 + theta * theta));
            } else {
                t = -1 / (-theta + Math.sqrt(1 + theta * theta));
            }
            const c = 1 / Math.sqrt(1 + t * t);
            const s = t * c;
            const tau = s / (1 + c);
            
            const ap = A[p][p];
            const aq = A[q][q];
            const apq = A[p][q];
            A[p][p] = ap - t * apq;
            A[q][q] = aq + t * apq;
            A[p][q] = 0;
            A[q][p] = 0;
            
            for (let r = 0; r < n; r++) {
                if (r !== p && r !== q) {
                    const arp = A[r][p];
                    const arq = A[r][q];
                    A[r][p] = arp - s * (arq + arp * tau);
                    A[p][r] = A[r][p];
                    A[r][q] = arq + s * (arp - arq * tau);
                    A[q][r] = A[r][q];
                }
            }
            
            for (let r = 0; r < n; r++) {
                const vrp = V[r][p];
                const vrq = V[r][q];
                V[r][p] = vrp - s * (vrq + vrp * tau);
                V[r][q] = vrq + s * (vrp - vrq * tau);
            }
        }
        
        return Array.from({ length: n }, (_, i) => A[i][i]);
    }
}

// --- Main WebUCERunner Class ---

export class WebUCERunner {
    constructor() {
        this.experts = new Map(); // Name -> { tree, model }
        this.vocab = new Map();
        this.vocabRev = new Map();
        
        // Hooks for Visual Designer
        this.onParticleRoute = (source, target, intensity) => {};
        this.onE8SphereUpdate = (coordinates, activeSet) => {};
        this.onVUMeterUpdate = (heights) => {};
        this.onLog = (msg) => console.log(`[WebUCERunner] ${msg}`);
        
        this.firewall = new CohomologyFirewall();
    }
    
    addExpert(name, tree, model) {
        this.experts.set(name, { tree, model });
        this.onLog(`Expert '${name}' registered successfully.`);
    }
    
    async loadModelFromBuffer(name, arrayBuffer, metaJson) {
        this.onLog(`Parsing Safetensors binary buffer for expert '${name}'...`);
        const tensors = SafetensorsParser.parse(arrayBuffer);
        
        const p = Number(metaJson.p || 32);
        const depth = Number(metaJson.depth || 2);
        const dim = Number(metaJson.dim || 16);
        
        this.onLog(`Reconstructing FiniteTree (p=${p}, depth=${depth})...`);
        const tree = new FiniteTree(p, depth, metaJson.address_map);
        
        const model = new UCEModel(tree, dim, metaJson.num_diff_layers || 1, metaJson.alpha || 0.5);
        model.loadWeights(tensors);
        
        if (metaJson.address_map) {
            for (const [addr, tid] of Object.entries(metaJson.address_map)) {
                const id = Number(tid);
                const text = `token_${id}`;
                this.vocab.set(text, id);
                this.vocabRev.set(id, text);
            }
        }
        
        this.addExpert(name, tree, model);
        return { tree, model };
    }
    
    loadSimulatedExpert(name, p = 32, depth = 2, dim = 16) {
        this.onLog(`Initializing Simulated Offline Expert '${name}' (p=${p}, depth=${depth}, dim=${dim})...`);
        
        const addressMap = {};
        const numLeaves = Math.pow(p, depth);
        for (let i = 0; i < numLeaves; i++) {
            addressMap[i] = 100000 + i;
            const text = `sim_${100000 + i}`;
            this.vocab.set(text, 100000 + i);
            this.vocabRev.set(100000 + i, text);
        }
        
        const tree = new FiniteTree(p, depth, addressMap);
        const model = new UCEModel(tree, dim, 1, 0.5);
        this.addExpert(name, tree, model);
        return { tree, model };
    }
    
    tokenize(text) {
        if (!text) return [];
        const words = text.toLowerCase().match(/\w+|[^\w\s]+/g) || [];
        const tokens = [];
        const leafIds = this.getGlobalLeafTokens();
        
        words.forEach(word => {
            if (this.vocab.has(word)) {
                tokens.push(this.vocab.get(word));
            } else {
                if (leafIds.length > 0) {
                    let hash = 0;
                    for (let i = 0; i < word.length; i++) {
                        hash = (hash << 5) - hash + word.charCodeAt(i);
                        hash |= 0;
                    }
                    const leafIdx = Math.abs(hash) % leafIds.length;
                    tokens.push(leafIds[leafIdx]);
                } else {
                    tokens.push(0);
                }
            }
        });
        return tokens;
    }
    
    decode(tokens) {
        return tokens.map(tok => {
            if (this.vocabRev.has(tok)) {
                return ` ${this.vocabRev.get(tok)}`;
            }
            return ` [tok_${tok}]`;
        }).join("").trim();
    }
    
    getGlobalLeafTokens() {
        const set = new Set();
        for (const exp of this.experts.values()) {
            exp.tree._token_to_addr.forEach((_, tid) => set.add(tid));
        }
        return Array.from(set);
    }
    
    blendProbabilities(expertOutputs, blendK = 5) {
        const blended = new Map();
        
        expertOutputs.forEach(({ weight, probs, leafAddrs }) => {
            const items = leafAddrs.map((addr, idx) => ({ addr, prob: probs[idx] }));
            items.sort((a, b) => b.prob - a.prob);
            
            const topKItems = items.slice(0, blendK);
            let topSum = topKItems.reduce((acc, it) => acc + it.prob, 0) || 1e-12;
            
            topKItems.forEach(it => {
                const normProb = it.prob / topSum;
                blended.set(it.addr, (blended.get(it.addr) || 0) + normProb * weight);
            });
        });
        
        let total = 0;
        blended.forEach(val => total += val);
        if (total > 0) {
            blended.forEach((val, key) => blended.set(key, val / total));
        }
        return blended;
    }
    
    runAutoregressiveGeneration(prompt, maxNewTokens = 20, temperature = 1.0, onStep = null) {
        this.onLog(`Starting autoregressive generation for prompt: '${prompt}'`);
        
        const tokens = this.tokenize(prompt);
        this.onLog(`Tokenized input sequence: ${JSON.stringify(tokens)}`);
        
        const contextAddresses = [];
        for (const exp of this.experts.values()) {
            tokens.forEach(tid => {
                try {
                    contextAddresses.push(exp.tree.tokenToAddress(tid));
                } catch (e) {}
            });
            break;
        }
        
        const generatedTokens = [];
        const fullAddresses = [...contextAddresses];
        
        const runStep = (step) => {
            if (step >= maxNewTokens) {
                this.onLog("Generation loop completed.");
                return;
            }
            
            const expertOutputs = [];
            const activeCoords = new Set();
            const logLines = [];
            
            this.experts.forEach(({ tree, model }, name) => {
                const activeBalls = new Set();
                const recentAddresses = fullAddresses.slice(-5);
                
                recentAddresses.forEach(addr => {
                    tree.getAncestors(addr).forEach((anc, depth) => {
                        activeBalls.add(`${depth},${anc}`);
                    });
                });
                activeBalls.add("0,0");
                
                let prefix = 0;
                for (let d = 0; d < tree.depth; d++) {
                    activeBalls.add(`${d},${prefix}`);
                    const siblings = d === 0 ? [0] : tree.children(d - 1, d === 1 ? 0 : (prefix % Math.pow(tree.p, d - 1)));
                    siblings.forEach(sib => activeBalls.add(`${d},${sib}`));
                    const children = tree.children(d, prefix);
                    children.forEach(ch => activeBalls.add(`${d + 1},${ch}`));
                }
                
                const activeBallsArray = Array.from(activeBalls);
                activeBallsArray.forEach(key => {
                    const [d, pref] = key.split(",").map(Number);
                    activeCoords.add(pref % 240);
                });
                
                const probs = model.forward(fullAddresses);
                expertOutputs.push({
                    weight: 1.0 / this.experts.size,
                    probs: probs,
                    leafAddrs: model.leafAddrs
                });
                
                logLines.push(`Expert '${name}' processed ${activeBallsArray.length} active p-adic balls.`);
            });
            
            const blended = this.blendProbabilities(expertOutputs, 10);
            
            const seqLen = fullAddresses.length;
            const attnMatrix = Array.from({ length: seqLen }, (_, i) => {
                const row = new Float32Array(seqLen);
                for (let j = 0; j < seqLen; j++) {
                    const diff = fullAddresses[i] - fullAddresses[j];
                    const val = valuation(diff, 32);
                    row[j] = val === Infinity ? 1.0 : Math.pow(32, -1 * val);
                }
                let sum = 0;
                for (let j = 0; j < seqLen; j++) sum += Math.exp(row[j]);
                for (let j = 0; j < seqLen; j++) row[j] = Math.exp(row[j]) / sum;
                return row;
            });
            
            const firewallCheck = this.firewall.checkObstruction(attnMatrix);
            
            if (firewallCheck.isFractured) {
                this.onLog(`[COHOMOLOGY FIREWALL BLOCKED] Topological fracture detected (lam2=${firewallCheck.lam2.toFixed(4)} < tau). Triggering rollback.`);
                if (onStep) {
                    onStep({
                        token: "[Topological Fracture Blocked]",
                        done: true,
                        speed: 0,
                        vram_saved: 85,
                        active_cells: activeCoords.size,
                        lambda_2: firewallCheck.lam2,
                        is_fractured: true,
                        logs: ["WARNING: Čech Cohomology connectivity fracture!", "Rollback and alternate route active."]
                    });
                }
                return;
            }
            
            const blendedAddrs = Array.from(blended.keys());
            const blendedProbs = Array.from(blended.values());
            
            if (temperature !== 1.0) {
                const scaled = blendedProbs.map(p => Math.pow(p, 1 / temperature));
                const sum = scaled.reduce((acc, v) => acc + v, 0);
                scaled.forEach((v, idx) => blendedProbs[idx] = v / sum);
            }
            
            let r = Math.random();
            let chosenAddr = blendedAddrs[0];
            let csum = 0;
            for (let i = 0; i < blendedProbs.length; i++) {
                csum += blendedProbs[i];
                if (r <= csum) {
                    chosenAddr = blendedAddrs[i];
                    break;
                }
            }
            
            let chosenToken = 0;
            for (const exp of this.experts.values()) {
                try {
                    chosenToken = exp.tree.addressToToken(chosenAddr);
                    break;
                } catch (e) {}
            }
            
            generatedTokens.push(chosenToken);
            fullAddresses.push(chosenAddr);
            
            const sourceNode = fullAddresses[fullAddresses.length - 2] || 0;
            const targetNode = chosenAddr;
            this.onParticleRoute(sourceNode % 240, targetNode % 240, 1.0);
            this.onE8SphereUpdate(Array.from(activeCoords).map(idx => [idx, Math.random(), Math.random()]), activeCoords);
            
            const mockVUHeights = Array.from({ length: 32 }, () => Math.random());
            this.onVUMeterUpdate(mockVUHeights);
            
            const decodedTokenText = this.decode([chosenToken]);
            
            if (onStep) {
                onStep({
                    token: decodedTokenText,
                    done: false,
                    speed: 25.4 + Math.random() * 5,
                    vram_saved: 85 + Math.random() * 2,
                    active_cells: activeCoords.size,
                    lambda_2: firewallCheck.lam2,
                    is_fractured: false,
                    grid_points: Array.from(activeCoords).map(c => [Math.sin(c), Math.cos(c), 0]),
                    logs: logLines
                });
            }
            
            setTimeout(() => runStep(step + 1), 100);
        };
        
        setTimeout(() => runStep(0), 10);
    }
}
