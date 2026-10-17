"""Static parity check: train/train_2_v0.py (ASL only) vs train/train_2_v5.py (entropy / Markov prior).

The pieces that must be identical between the two scripts (augmentation, ASL loss, early stopping, EMA / SWA,
evaluation, seeding) are compared as normalized ASTs, and every argparse flag the two scripts share must have the
same default. Reads the files only; imports nothing from them.

    python -I scripts/check_v0_v5_parity.py                       # v5 from the working tree
    python -I scripts/check_v0_v5_parity.py --v5-ref v5-pre-moe-removal   # v5 from a git ref
"""
import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNCS = ["stable_seed_from_key", "deterministic_random_resized_crop", "deterministic_brightness_contrast",
         "deterministic_aug", "deterministic_val_resize", "AsymmetricLoss", "EarlyStopping", "preprocess_batch",
         "seed_everything", "SeededSampleShuffle"]
TRAINER_METHODS = ["_save_json", "_ema_update", "_ema_state_dict", "_save_best", "_snapshot_ema_cpu", "_save_swa",
                   "evaluate"]
CONSTANTS = ["SAMPLE_SHUFFLE_BUFSIZE", "DEFAULT_STAGE1_CKPT", "SIZE", "AUG_BASE_SEED"]
V0_ONLY_FLAGS = {"--resume"}


def load(path_or_ref: str, is_ref: bool) -> ast.Module:
    if is_ref:
        src = subprocess.run(["git", "show", f"{path_or_ref}:train/train_2_v5.py"], capture_output=True, text=True,
                             check=True, cwd=ROOT).stdout
    else:
        src = Path(path_or_ref).read_text()
    return ast.parse(src)


def top(tree, name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return n
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n
    return None


def method(tree, name):
    trainer = top(tree, "Trainer")
    for n in trainer.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def dump(n):
    return ast.dump(n, include_attributes=False)


def flag_defaults(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            name = ast.literal_eval(n.args[0])
            d = None
            for k in n.keywords:
                if k.arg == "default":
                    d = ast.unparse(k.value)
                if k.arg == "action" and ast.literal_eval(k.value) == "store_true" and not any(
                        x.arg == "default" for x in n.keywords):
                    d = "False"
            out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-ref", default=None, help="git ref to read train_2_v5.py from (default: working tree)")
    args = ap.parse_args()
    v0 = load(str(ROOT / "train/train_2_v0.py"), False)
    v5 = load(args.v5_ref or str(ROOT / "train/train_2_v5.py"), bool(args.v5_ref))
    bad = []

    for name in FUNCS + CONSTANTS:
        a, b = top(v0, name), top(v5, name)
        if a is None or b is None or dump(a) != dump(b):
            bad.append(f"differs or missing: {name}")
    for name in TRAINER_METHODS:
        a, b = method(v0, name), method(v5, name)
        if a is None or b is None or dump(a) != dump(b):
            bad.append(f"differs or missing: Trainer.{name}")

    d0, d5 = flag_defaults(v0), flag_defaults(v5)
    for k in sorted(set(d0) & set(d5)):
        if d0[k] != d5[k]:
            bad.append(f"default differs for {k}: v0={d0[k]} v5={d5[k]}")
    for k in sorted(set(d0) - set(d5) - V0_ONLY_FLAGS):
        bad.append(f"flag only in v0: {k}")

    print(f"checked {len(FUNCS)} functions/classes, {len(TRAINER_METHODS)} Trainer methods, {len(CONSTANTS)} constants, "
          f"{len(set(d0) & set(d5))} shared flags (v5 = {args.v5_ref or 'working tree'})")
    if bad:
        print("\n".join("FAIL " + b for b in bad))
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()
