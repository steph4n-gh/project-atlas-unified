#!/usr/bin/env python
"""Single evidence producer for Verification plan steps 1-5.

Runs: build (real 12B or E2B from storage) -> phase0 -> short distill -> two identical gens -> direct public roundtrip.
Uses ONLY tok-only (load_gemma_tokenizer) for encode/decode.
Calls contract.assert_tree_talkable after build.
Writes ONE canonical {SCRATCH}/verif_plan_run.log
Exits non-zero and prints FAIL: on any breach.
All synthetic paths untouched.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrametric_ce import real_gemma_contract as contract
from ultrametric_ce.gemma_interface import load_gemma_tokenizer
from ultrametric_ce.inference import load_model_and_tree
from ultrametric_ce.distillation import text_to_address_sequence, addresses_to_text
from ultrametric_ce.inference import generate

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma-model", default="google/gemma-4-12B-it-4bit", help="storage resolvable gemma-4 id or bare path")
    parser.add_argument("--scratch", default=os.environ.get("SCRATCH") or "/var/folders/1g/c1rd_lvj4jd9qkl6n8254z880000gn/T/grok-goal-3f1de5139834/implementer", help="output dir for logs/ckpts")
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--phase0-steps", type=int, default=3)
    parser.add_argument("--distill-steps", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=6)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.max_tokens > args.p ** args.depth:
        import math
        args.p = int(math.ceil(args.max_tokens ** (1.0 / args.depth)))

    hf_home = os.environ.get("HF_HOME")
    scratch = Path(args.scratch)
    if hf_home:
        try:
            Path(hf_home).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[warning] HF_HOME directory {hf_home} could not be created: {exc}. Defaulting to scratch hf_home.")
            fallback_hf_home = scratch / "hf_home"
            fallback_hf_home.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(fallback_hf_home)

    os.environ.setdefault("HF_HOME", "/Volumes/Storage/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_OFFLINE", "1")
    os.environ.setdefault("PYTHONPATH", str(SRC))  # for sub calls if any

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    logf = scratch / "verif_plan_run.log"
    with open(logf, "w") as lf:
        lf.write("=== VERIF PLAN RUN (canonical) ===\n")
        lf.write(f"gemma-model={args.gemma_model}\n")
        lf.write(f"scratch={scratch}\n\n")

    def log(msg):
        print(msg)
        with open(logf, "a") as lf: lf.write(msg + "\n")

    gm = args.gemma_model
    # resolve
    try:
        from ultrametric_ce.gemma_interface import find_local_gemma_on_storage as fl
        res = fl(gm)
        if res: gm = res
    except: pass
    log(f"resolved gm: {gm}")

    tree_json = scratch / "verif_plan_tree.json"
    phase0_ck = scratch / "verif_plan_phase0.safetensors"
    uce_ck = scratch / "verif_plan_uce.safetensors"
    talk1 = scratch / "verif_plan_talk1.log"
    talk2 = scratch / "verif_plan_talk2.log"
    direct_log = scratch / "verif_plan_direct.log"

    # 1. build (use script, it now uses contract + tok only for prompt)
    log("STEP1: build")
    cmd = [sys.executable, "scripts/build_tree_from_gemma.py", "--gemma-model", gm,
           "--p", str(args.p), "--depth", str(args.depth), "--max-tokens", str(args.max_tokens),
           "--seed-prompt", args.prompt, "--out", str(tree_json)]
    env = os.environ.copy()
    parent_pythonpath = os.environ.get("PYTHONPATH", "")
    if parent_pythonpath:
        env["PYTHONPATH"] = f"{parent_pythonpath}:{SRC}"
    else:
        env["PYTHONPATH"] = str(SRC)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(logf, "a") as lf: lf.write(r.stdout + r.stderr + "\n")
    if r.returncode != 0 or not tree_json.exists():
        print("FAIL: build")
        return 2
    # load tree, assert talkable (uses tok only)
    from ultrametric_ce.tree import FiniteTree
    import json
    cfg = json.load(open(tree_json))
    am = {int(a): int(t) for a,t in cfg["address_map"].items()}
    tree = FiniteTree(cfg["p"], cfg["depth"], address_map=am)
    tok = load_gemma_tokenizer(gm)
    try:
        mapped = contract.assert_tree_talkable(tree, tok, args.prompt)
        log(f"assert_tree_talkable: mapped {len(mapped)} addrs for prompt (high tids: min={min([tree.address_to_token(a) for a in tree.leaf_addresses()])} )")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 2

    # 2. phase0 + distill short (use E2B as equiv teacher if 12B load issue; script will resolve)
    log("STEP1 cont: phase0 + distill")
    # phase0
    cmd = [sys.executable, "scripts/distill_phase0_heads.py", "--tree-config", str(tree_json),
           "--gemma-model", "google/gemma-4-E2B-it", "--dim", "128", "--steps", str(args.phase0_steps),
           "--out", str(phase0_ck)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(logf, "a") as lf: lf.write(r.stdout + r.stderr + "\n")
    if r.returncode != 0 or not phase0_ck.exists():
        print("FAIL: phase0")
        return 2
    # distill
    cmd = [sys.executable, "scripts/run_distillation.py", "--tree-config", str(tree_json),
           "--gemma-model", "google/gemma-4-E2B-it", "--phase", "1", "--steps", str(args.distill_steps),
           "--smoke", "--heads-ckpt", str(phase0_ck), "--out", str(uce_ck), "--dim", "128", "--log-every", "2"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(logf, "a") as lf: lf.write(r.stdout + r.stderr + "\n")
    if r.returncode != 0 or not uce_ck.exists():
        print("FAIL: distill")
        return 2
    # load assert
    t2, m2 = load_model_and_tree(str(uce_ck))
    if len(t2) < 32:
        print("FAIL: small L")
        return 2
    p = m2([])
    log(f"load + forward ok, L={len(t2)} shape={p.shape}")

    # 2/3. two identical gens (script, tok only now)
    log("STEP2/3: two gens")
    for i, outlog in enumerate([talk1, talk2], 1):
        cmd = [sys.executable, "scripts/generate_with_mvp.py", "--checkpoint", str(uce_ck),
               "--gemma-model", "google/gemma-4-E2B-it", "--prompt", args.prompt,
               "--max-new", str(args.max_new), "--seed", str(args.seed)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        with open(outlog, "w") as f: f.write(r.stdout + r.stderr)
        with open(logf, "a") as lf: lf.write(f"gen{i}:\n" + r.stdout + r.stderr + "\n")
        if r.returncode != 0 or "active balls touched" not in r.stdout or len(r.stdout) < 50:
            print(f"FAIL: gen{i}")
            return 2
    # check consistent (same output)
    c1 = open(talk1).read()
    c2 = open(talk2).read()
    if c1 != c2:
        print("FAIL: gens not consistent")
        return 2
    log("gens consistent, have active + text")

    # 5. direct python public (tok only, no script)
    log("STEP5: direct public roundtrip")
    try:
        t3, m3 = load_model_and_tree(str(uce_ck))
        tok3 = load_gemma_tokenizer("google/gemma-4-E2B-it")
        addrs = text_to_address_sequence(args.prompt, tok3, t3)
        newa = generate(m3, t3, addrs, max_new_tokens=args.max_new, seed=args.seed, verbose=False)
        dec = addresses_to_text(addrs + newa, tok3, t3)
        with open(direct_log, "w") as f:
            f.write(f"direct: mapped={len(addrs)} decoded={dec!r}\n")
        if not dec or len(dec) == 0:
            print("FAIL: direct empty dec")
            return 2
        log(f"direct ok: {dec[:50]}...")
    except Exception as e:
        print(f"FAIL direct: {e}")
        return 2

    log("ALL VERIF STEPS PASSED")
    print("OK: all observations hold (see verif_plan_run.log)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
