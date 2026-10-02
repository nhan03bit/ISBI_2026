"""Pre-flight checks for the v5 patient-prior arm (docs/plan/2026-09-30-markov-ab-training-plan.md).

CPU only. Fails with an AssertionError on the first problem.
  1. priors.pt: shard class order; val r == the B-late log_ratio of analysis/patient_markov_check
     (same vector for B-early and B-late); train priors use train-fold history only.
  2. ConvNeXt2Prior at init == ConvNeXt2 exactly (Stage-1 weights, eval mode, random priors);
     zero offset for r = 0; the conditioner gets gradients.
  3. Checkpoint loader dispatch: v3 -> ConvNeXt2, v4 -> ConvNeXt2Markov, v5 -> ConvNeXt2Prior.
  4. train_2_v5 mapper on real shard samples: 3-tuple, r matches priors.pt, dropout deterministic.

    srun -c 4 --mem=32G -p general --time=0:30:00 python analysis/check_prior_conditioning.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "train"))
import markov_check as mc  # noqa: E402
import patient_prior as pp  # noqa: E402
from class_groups import load_train_prevalence  # noqa: E402

STAGE1 = ROOT / "train/checkpoint3/Model_20260119_062652/model_best.pth"
V3 = ROOT / "train/checkpoint_Triplet_3/push/img768_dp01_seed42/Model_run/model_swa.pth"
V4 = ROOT / "train/checkpoint_Triplet_3/markov/img768_mk1_seed42/Model_run/model_swa.pth"


def check_priors() -> dict:
    pri = torch.load(pp.PRIORS, weights_only=False)
    classes = list(torch.load(pp.LABEL_INFO, weights_only=False)["class_names"])
    assert list(pri["class_names"]) == classes, "priors.pt class order != label_info.pt"
    _, prev, _ = load_train_prevalence(classes)
    tr, va_full = pp.load_folds(classes)
    sg, _ = pp.study_table(tr, va_full)

    # val r == B-late log_ratio, in the aligned (dumped-probs) order of patient_markov_check
    va = va_full.iloc[np.load(mc.KEY_ORDER)].reset_index(drop=True)
    q = pp.lookup_prior(va["pid"].to_numpy(), va["date"].to_numpy(), pp.date_states(sg))
    prior, has = pp.prior_bits(q)
    log_ratio = pp.prior_log_ratio(prior, has, pri["a"], pri["b"], prev)
    idx = {n: i for i, n in enumerate(pri["val"]["fnames"])}
    r_val = pri["val"]["r"][[idx[f] for f in va["filename"]]]
    assert np.allclose(r_val[:, :30], log_ratio, atol=1e-5), "val r != B-late log_ratio"
    assert (r_val[:, 30] == has).all()
    print(f"[1] val r == B-late log_ratio for all {len(va)} aligned rows (max |d| "
          f"{np.abs(r_val[:, :30] - log_ratio).max():.2e}); has_prior {has.mean():.3f}")

    # train priors must come from train-fold history only: where full history would give a
    # different (val-only, more recent) prior, the stored r must be the train-only one
    q_full = pp.lookup_prior(tr["pid"].to_numpy(), tr["date"].to_numpy(), pp.date_states(sg))
    q_tr = pp.lookup_prior(tr["pid"].to_numpy(), tr["date"].to_numpy(), pp.date_states(sg[sg["in_tr"]]))
    differ = (q_full["prior_date"].fillna(pd.Timestamp(0)) != q_tr["prior_date"].fillna(pd.Timestamp(0))).to_numpy()
    p_tr, h_tr = pp.prior_bits(q_tr)
    r_tr_expect = pp.prior_log_ratio(p_tr, h_tr, pri["a"], pri["b"], prev)
    assert list(pri["train"]["fnames"]) == tr["filename"].tolist()
    assert np.allclose(pri["train"]["r"][:, :30], r_tr_expect, atol=1e-5), "train r != train-only history"
    assert (pri["train"]["r"][:, 30] == h_tr).all()
    print(f"    train r uses train-fold history only: {int(differ.sum())} train images would get a "
          f"different (val-study) prior under full history, all stored with the train-only prior; "
          f"has_prior {h_tr.mean():.3f}")
    return pri


def check_model(pri: dict) -> None:
    from convnext import ConvNeXt2
    from markov_layer import CONVNEXT2_KW
    from prior_conditioning import ConvNeXt2Prior

    torch.manual_seed(0)
    sd = torch.load(STAGE1, map_location="cpu")
    base, cond = ConvNeXt2(**CONVNEXT2_KW, num_classes=30), ConvNeXt2Prior(**CONVNEXT2_KW, num_classes=30)
    base.load_state_dict(sd, strict=False)
    res = cond.load_state_dict(sd, strict=False)
    assert all(k.startswith("prior_cond.") for k in res.missing_keys) and res.missing_keys, res.missing_keys
    assert not res.unexpected_keys, res.unexpected_keys
    base.eval(); cond.eval()
    x = torch.rand(3, 3, 256, 256)
    r = torch.from_numpy(pri["val"]["r"][np.flatnonzero(pri["val"]["has"])[:3]])
    with torch.no_grad():
        lb, _ = base(x)
        lc, _ = cond(x, prior=r)
        ln, _ = cond(x)
    d = (lb - lc).abs().max().item()
    assert d == 0.0 and (lb - ln).abs().max().item() == 0.0, f"not identity at init: max |d| {d}"
    print(f"[2] ConvNeXt2Prior(init, prior) == ConvNeXt2 exactly (max |d| {d}); missing keys "
          f"{sorted(res.missing_keys)}")

    cond.train()
    r_mix = torch.cat([r[:2], torch.zeros(1, r.size(1))])
    logits, _ = cond(x, prior=r_mix)
    logits.sum().backward()
    gu, gw = cond.prior_cond.u.grad, cond.prior_cond.W.weight.grad
    assert gu is not None and gu.abs().sum() > 0 and gw.abs().sum() > 0, "conditioner gets no gradient"
    with torch.no_grad():
        cond.prior_cond.u.normal_(); cond.prior_cond.W.weight.normal_()
        off = cond.prior_cond(r_mix)
    assert off[2].abs().max().item() == 0.0 and off[0].abs().max().item() > 0, "r = 0 must give 0 offset"
    print(f"    grad |u| {gu.abs().sum():.3e}, |W| {gw.abs().sum():.3e}; offset exactly 0 for r = 0 "
          f"even with random u, W")


def check_loader_dispatch() -> None:
    from prior_conditioning import ConvNeXt2Prior, load_model_for_state_dict
    from markov_layer import CONVNEXT2_KW
    kinds = {}
    for name, sd in (("v3", torch.load(V3, map_location="cpu")), ("v4", torch.load(V4, map_location="cpu")),
                     ("v5", ConvNeXt2Prior(**CONVNEXT2_KW, num_classes=30).state_dict())):
        m, _ = load_model_for_state_dict(sd)
        kinds[name] = type(m).__name__
    assert kinds == {"v3": "ConvNeXt2", "v4": "ConvNeXt2Markov", "v5": "ConvNeXt2Prior"}, kinds
    print(f"[3] loader dispatch: {kinds}")


def check_mapper(pri: dict) -> None:
    import webdataset as wds
    import train_2_v5 as v5
    lookup = {n: pri[s]["r"][i] for s in ("train", "val") for i, n in enumerate(pri[s]["fnames"])}
    for split, is_train in (("train", True), ("val", False)):
        shard = sorted(Path(f"/data/psytp7/wds_shards_{split}_raw").glob("shard_*.tar"))[0]
        it = iter(wds.WebDataset("file:" + shard.as_posix(), shardshuffle=False))
        n_has = 0
        for _ in range(64):
            s = next(it)
            out = v5.decode_and_transform(s, is_train=is_train, epoch=0, size=128,
                                          prior_lookup=lookup, prior_dropout=0.0)
            assert out is not None and len(out) == 3, "mapper must return (x, y, r)"
            x, y, r = out
            want = lookup[s["fname"].decode()]
            assert np.array_equal(r.numpy(), want), "mapper prior != priors.pt"
            assert torch.equal(y, torch.load(io.BytesIO(s["cls"]))), "labels changed"
            n_has += int(r[-1].item())
            if is_train:  # dropout must be deterministic per (key, epoch)
                a1 = v5.decode_and_transform(s, is_train=True, epoch=3, size=128, prior_lookup=lookup,
                                             prior_dropout=0.5)[2]
                a2 = v5.decode_and_transform(s, is_train=True, epoch=3, size=128, prior_lookup=lookup,
                                             prior_dropout=0.5)[2]
                assert torch.equal(a1, a2), "prior dropout is not deterministic"
        print(f"[4] mapper on 64 {split} samples of {shard.name}: 3-tuple, r == priors.pt, labels intact, "
              f"has_prior {n_has}/64" + (", dropout deterministic" if is_train else ""))


if __name__ == "__main__":
    torch.set_num_threads(4)
    pri = check_priors()
    check_model(pri)
    check_loader_dispatch()
    check_mapper(pri)
    print("\nALL PRE-FLIGHT CHECKS PASSED")
