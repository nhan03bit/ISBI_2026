"""Assemble a Task-1 submission CSV from evaluate/predict_task1.py per-model dumps (2026-10-17, CPU).

predict_task1.py writes, next to its CSV, <out>.per_model/<name>.npy (template row order, label_info class order) and
<out>.manifest.json. This averages the named members' probabilities (uniform mean, as predict_task1 and
evaluate_tta do), maps them to the template's columns by name, and writes a new CSV plus its own manifest, so an
ensemble can change without a GPU rerun.

    python -I evaluate/assemble_task1.py --run /data/psytp7/isbi_task1/pred/candidates_area.csv \
        --out /data/psytp7/isbi_task1/pred/E5_area.csv \
        push__img768_dp01_seed42__model_swa@768_flip_area.npy ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("members", nargs="+", help="per-model dump file names (or a unique substring of one)")
    ap.add_argument("--run", type=Path, required=True, action="append",
                    help="a predict_task1.py output CSV; repeat to draw members from several runs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; pick a new --out")

    runs = []
    for r in args.run:
        man = json.loads(r.with_suffix(".manifest.json").read_text())
        if sha256(Path(man["out"])) != man["out_sha256"]:
            raise ValueError(f"{man['out']} changed since its manifest was written")
        runs.append((r, man))
    tpl_sha = {m["template_sha256"] for _, m in runs}
    orders = {tuple(m["model_class_order"]) for _, m in runs}
    if len(tpl_sha) != 1 or len(orders) != 1:
        raise ValueError("runs used different templates or class orders")
    man0 = runs[0][1]
    tpl = pd.read_csv(man0["template"], dtype={"ImageID": str})
    if sha256(Path(man0["template"])) != man0["template_sha256"]:
        raise ValueError("template changed since the runs")
    model_names = man0["model_class_order"]
    class_cols = man0["template_class_order"]
    perm = [model_names.index(c) for c in class_cols]

    picked = []
    for m in args.members:
        hits = [(r, c) for r, man in runs for c in man["checkpoints"] if m == c["dump"] or m in c["dump"]]
        if len(hits) != 1:
            raise ValueError(f"member {m!r} matches {len(hits)} dumps: {[c['dump'] for _, c in hits]}")
        picked.append(hits[0])
    arrs = [np.load(r.with_suffix(".per_model") / c["dump"]) for r, c in picked]
    if any(a.shape != (len(tpl), 30) or np.isnan(a).any() for a in arrs):
        raise ValueError("a member dump has the wrong shape or NaNs")
    probs = np.mean(arrs, 0)[:, perm]

    out = tpl[["ImageID"]].copy()
    for j, c in enumerate(class_cols):
        out[c] = probs[:, j]
    out = out[list(tpl.columns)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.8g")
    manifest = {"generated": datetime.now().isoformat(timespec="seconds"), "out": str(args.out.resolve()),
                "out_sha256": sha256(args.out), "rows": len(out), "template": man0["template"],
                "template_sha256": man0["template_sha256"], "ensemble": "uniform mean of member probabilities",
                "members": [{**c, "run": str(r)} for r, c in picked],
                "interp": sorted({m["interp"] for _, m in runs}),
                "channel_mean": sorted({bool(m.get("channel_mean", False)) for _, m in runs})}
    with open(args.out.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} ({len(out)} rows, {len(picked)} members: {[c['dump'] for _, c in picked]})")


if __name__ == "__main__":
    main()
