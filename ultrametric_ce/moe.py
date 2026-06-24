"""UCE Mixture of Experts (MoE) Router.

Dynamically classifies prompt domains, lazy-loads the target UCE expert model, 
and coordinates memory-mapped dynamic weight swapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from ultrametric_ce.inference import load_model_and_tree, generate
from ultrametric_ce.model import UCEModel, WeightManager
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.distillation import text_to_address_sequence, addresses_to_text

class UCEMoeRouter:
    """Orchestrates 8 developer experts under a unified gating router with dynamic weight paging."""

    def __init__(
        self,
        moe_dir: str | Path,
        gemma_model_id: str = "google/gemma-4-E2B-it",
        dim: int = 16,
        max_vram_bytes: int = 256 * 1024 * 1024, # Cap VRAM per expert weight manager
        active_experts_k: int = 1,
    ) -> None:
        self.moe_dir = Path(moe_dir)
        self.gemma_model_id = gemma_model_id
        self.dim = dim
        self.max_vram_bytes = max_vram_bytes
        self.active_experts_k = active_experts_k

        # Domain map mapping class prediction index to expert name
        self.domain_map = {
            0: "python_coder",
            1: "web_stack",
            2: "rust_systems",
            3: "database_sql",
            4: "devops_infra",
            5: "ml_tensors",
            6: "markdown_config",
            7: "gateway_router"
        }

        # Lazy tokenizer cache
        self.tokenizer = None
        self._init_tokenizer()

        # Loaded experts cache: expert_name -> (tree, model)
        self.experts: Dict[str, Tuple[FiniteTree, UCEModel]] = {}

        # Load Gateway Router immediately
        self.gateway_name = "gateway_router"
        self._load_expert(self.gateway_name)

    def _init_tokenizer(self) -> None:
        """Loads the tokenizer only, avoiding full model memory load."""
        try:
            from ultrametric_ce.gemma_interface import load_gemma_tokenizer, find_local_gemma_on_storage
            gm = self.gemma_model_id
            resolved = find_local_gemma_on_storage(gm)
            if resolved:
                gm = resolved
            self.tokenizer = load_gemma_tokenizer(gm)
        except Exception as e:
            print(f"[warn] Failed to load tokenizer: {e}")

    def _load_expert(self, name: str) -> Tuple[FiniteTree, UCEModel]:
        """Lazy-loads an expert model and tree from disk using a memory-mapped WeightManager."""
        if name in self.experts:
            return self.experts[name]

        ckpt_path = self.moe_dir / f"uce_{name}.safetensors"
        meta_path = ckpt_path.with_suffix(".meta.json")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Expert checkpoint not found: {ckpt_path}")

        # Extract number of balls to configure weight manager paging
        meta = json.loads(meta_path.read_text())
        p = int(meta["p"])
        depth = int(meta["depth"])
        
        # Calculate total balls in the tree: sum_{d=0}^depth p^d
        num_balls = sum(p ** d for d in range(depth + 1))

        # Instantiate weight manager for lazy memory-mapped paging
        wm = WeightManager(
            weight_file_path=str(ckpt_path),
            num_balls=num_balls,
            dim=self.dim,
            dtype=mx.float16,
            max_vram_bytes=self.max_vram_bytes,
            use_mmap=True
        )

        tree, model = load_model_and_tree(ckpt_path, meta_path=meta_path, weight_manager=wm)
        self.experts[name] = (tree, model)
        return tree, model

    def route_prompt(
        self,
        prompt: str,
        k: int | None = None,
        return_weights: bool = False
    ) -> str | List[Tuple[str, float]]:
        """Uses the gateway router expert to classify the prompt domain."""
        import re
        if k is None:
            k = self.active_experts_k
        k = max(1, min(k, len(self.domain_map)))

        lower_prompt = prompt.lower()
        
        # 1. High-precision regex word-boundary keyword check corresponding to domain vocabularies
        domain_keywords = {
            "database_sql": [
                r"\bsql\b", r"\bdatabase\b", r"\bdb\b", r"\bselect\b", r"\binsert\b", 
                r"\bdelete\b", r"\bupdate\b", r"\bjoin\b", r"\bwhere\b", r"\bquery\b", 
                r"\btable\b", r"\bpostgres\b", r"\bmysql\b", r"\bsqlite\b", r"\bcreate\b"
            ],
            "devops_infra": [
                r"\bdocker\b", r"\bkubernetes\b", r"\bkubectl\b", r"\bkube\b", r"\bdevops\b", 
                r"\bport\b", r"\bip\b", r"\binfra\b", r"\byaml\b", r"\bdeployment\b", 
                r"\bpod\b", r"\bservice\b", r"\bconfigmap\b", r"\bcontainer\b", r"\bport-forward\b"
            ],
            "web_stack": [
                r"\breact\b", r"\bhtml\b", r"\bcss\b", r"\bjs\b", r"\bjavascript\b", 
                r"\btypescript\b", r"\bts\b", r"\bweb\b", r"\bcomponent\b", r"\bstate\b", 
                r"\bprops\b", r"\burl\b", r"\blink\b", r"\bstylesheet\b", r"\bdom\b"
            ],
            "rust_systems": [
                r"\brust\b", r"\bcargo\b", r"\bfn\b", r"\bimpl\b", r"\bstruct\b", 
                r"\benum\b", r"\btrait\b", r"\bunsafe\b", r"\bmut\b", r"\bborrow\b", 
                r"\blifetime\b", r"\bsystems\b"
            ],
            "ml_tensors": [
                r"\btensor\b", r"\bnumpy\b", r"\bpytorch\b", r"\btensorflow\b", r"\bml\b", 
                r"\bmnist\b", r"\bregression\b", r"\bweight\b", r"\bbias\b", r"\bdim\b", 
                r"\breshape\b", r"\btranspose\b", r"\bmatrix\b", r"\bvector\b"
            ],
            "markdown_config": [
                r"\bmarkdown\b", r"\bmd\b", r"\bjson\b", r"\bconfig\b", r"\bconfiguration\b", 
                r"\byaml\b", r"\bschema\b", r"\bmetadata\b", r"\btags\b", r"\bfootnote\b"
            ],
            "python_coder": [
                r"\bpython\b", r"\bcoder\b", r"\bpy\b", r"\bscript\b", r"\bdef\b", 
                r"\bclass\b", r"\bimport\b", r"\bself\b", r"\blambda\b", r"\bpip\b"
            ],
            "gateway_router": [
                r"\bgateway\b", r"\brouter\b", r"\broute\b", r"\brouting\b", r"\bgating\b", 
                r"\bsubspace\b", r"\bclassification\b", r"\bclassify\b"
            ]
        }
        
        for domain, patterns in domain_keywords.items():
            for pat in patterns:
                if re.search(pat, lower_prompt):
                    if return_weights:
                        return [(domain, 1.0)]
                    return domain

        # 2. Fallback to gateway model forward pass
        gtree, gmodel = self.experts[self.gateway_name]
        
        if self.tokenizer is None:
            expert_name = self.domain_map.get(len(prompt) % 8, "python_coder")
            if return_weights:
                return [(expert_name, 1.0)]
            return expert_name

        addrs = text_to_address_sequence(prompt, self.tokenizer, gtree)
        if not addrs:
            expert_name = self.domain_map.get(len(prompt) % 8, "python_coder")
            if return_weights:
                return [(expert_name, 1.0)]
            return expert_name

        probs = gmodel.forward(addrs)
        
        # Calculate domain marginal probabilities across the gateway tree's leaves
        domain_probs = {d: 0.0 for d in self.domain_map.values()}
        probs_list = probs.tolist()
        for idx, p_val in enumerate(probs_list):
            domain_name = self.domain_map.get(idx % 8, "python_coder")
            domain_probs[domain_name] += p_val

        # Sort and select Top-K
        sorted_domains = sorted(domain_probs.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_domains[:k]
        
        # Normalize weights
        total_w = sum(w for _, w in top_k) or 1e-12
        top_k_normalized = [(name, w / total_w) for name, w in top_k]

        if return_weights:
            return top_k_normalized
        return top_k_normalized[0][0]


    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        k: int | None = None
    ) -> str:
        """Routes prompt to active experts, pages weights, and runs UCE generation (hard or soft top-k)."""
        if k is None:
            k = self.active_experts_k

        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer must be loaded to run MoE generation.")

        # Hard gating optimization path for single expert
        if k == 1:
            expert_name = self.route_prompt(prompt, k=1, return_weights=False)
            if verbose:
                print(f"[MoE Router] Selected Expert: '{expert_name}' for prompt: {prompt!r}")

            tree, model = self._load_expert(expert_name)
            prompt_addrs = text_to_address_sequence(prompt, self.tokenizer, tree)
            if not prompt_addrs:
                prompt_addrs = [tree.leaf_addresses()[0]]

            new_addrs = generate(
                model,
                tree,
                prompt_addresses=prompt_addrs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
                verbose=False
            )

            full_addrs = prompt_addrs + new_addrs
            decoded_text = addresses_to_text(full_addrs, self.tokenizer, tree)
            return decoded_text

        # Soft gating Top-K blended path
        expert_weights = self.route_prompt(prompt, k=k, return_weights=True)
        if verbose:
            print(f"[MoE Router] Top-{k} Blended Gating:")
            for name, w in expert_weights:
                print(f"  - {name}: {w*100:.1f}%")

        active_instances = []
        for name, w in expert_weights:
            tree, model = self._load_expert(name)
            prompt_addrs = text_to_address_sequence(prompt, self.tokenizer, tree)
            if not prompt_addrs:
                prompt_addrs = [tree.leaf_addresses()[0]]
            active_instances.append({
                "name": name,
                "weight": w,
                "tree": tree,
                "model": model,
                "context": list(prompt_addrs),
                "vocab": {tree.address_to_token(a): a for a in tree.leaf_addresses()}
            })

        if seed is not None:
            mx.random.seed(seed)

        generated_tokens = []

        for step in range(max_new_tokens):
            global_probs = {}
            
            # Aggregate probabilities over token IDs across all active experts
            for inst in active_instances:
                model = inst["model"]
                tree = inst["tree"]
                ctx = inst["context"]
                w = inst["weight"]
                
                probs = model.forward(ctx)
                probs_list = probs.tolist()
                
                for leaf_idx, leaf_prob in enumerate(probs_list):
                    try:
                        addr = tree.leaf_addresses()[leaf_idx]
                        tok_id = tree.address_to_token(addr)
                        global_probs[tok_id] = global_probs.get(tok_id, 0.0) + w * leaf_prob
                    except (KeyError, IndexError):
                        pass

            if not global_probs:
                break

            token_ids = list(global_probs.keys())
            probs_list = [global_probs[tid] for tid in token_ids]
            sum_p = sum(probs_list) or 1e-12
            normalized_probs = [p_val / sum_p for p_val in probs_list]

            if temperature != 1.0:
                logits = [mx.log(mx.array(max(pp, 1e-12))) / temperature for pp in normalized_probs]
                logits_arr = mx.stack(logits)
            else:
                logits_arr = mx.log(mx.array(normalized_probs) + 1e-12)

            sub_idx = int(mx.random.categorical(logits_arr).item())
            chosen_tok = token_ids[sub_idx]
            generated_tokens.append(chosen_tok)

            # Update context for each active expert
            for inst in active_instances:
                tree = inst["tree"]
                vocab = inst["vocab"]
                if chosen_tok in vocab:
                    inst["context"].append(vocab[chosen_tok])
                else:
                    inst["context"].append(tree.leaf_addresses()[0])

        decoded_text = self.tokenizer.decode(generated_tokens)
        if verbose:
            print(f"[MoE Router] Generated: {decoded_text!r}")
        return decoded_text

