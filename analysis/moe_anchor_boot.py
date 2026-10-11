import sys, json, time
from pathlib import Path
import numpy as np, torch
from torchmetrics.classification import (MultilabelAveragePrecision, MultilabelAUROC,
    MultilabelF1Score, BinaryCalibrationError)
O = Path("/home/psytp7/ISBI_2026/analysis/out")
AB, OR, RR, W01, W1 = O/"probs_ab_flip", O/"probs_v5_orig_recheck_flip", O/"probs_v5_rerun_flip", O/"probs_w01_flip", O/"probs_w1_flip"
L = np.load(AB/"labels.npy").astype(np.uint8)
for d in (OR, RR, W01, W1):
    assert (np.load(d/"labels.npy") == L).all(), d
# key order identical?
k0 = (AB/"keys.txt").read_text().split("\n")
for d in (OR, RR): assert (d/"keys.txt").read_text().split("\n") == k0
print("w01/w1 have no keys.txt:", (W01/"keys.txt").exists(), (W1/"keys.txt").exists())
N, C = L.shape
ld = lambda d,f: np.load(d/f).astype(np.float32)
f = lambda s,seed,tag: f"prior__img768_qprior_seed{seed}{s}__model_swa@768_flip_prior-{tag}.npy"
M = {}
for tag in ("with","zero"):
    M[f"o42{tag}"] = ld(OR, f("",42,tag)); M[f"o1024{tag}"] = ld(OR, f("",1024,tag))
    M[f"r42{tag}"] = ld(RR, f("_rerun",42,tag)); M[f"r1024{tag}"] = ld(RR, f("_rerun",1024,tag))
v42 = ld(W01,"push__img768_dp01_seed42__model_swa@768_flip.npy"); v1024 = ld(W01,"push__img768_dp01_seed1024__model_swa@768_flip.npy")
v86_1024 = ld(W01,"push__img1024_dp01_seed86__model_swa@1024_flip.npy"); v86_640 = ld(W01,"res_sweep__img640_seed86__model_best@640_flip.npy")
v86_896 = ld(W01,"push__img896_dp01_seed86__model_swa@896_flip.npy")
# sanity: w1 members same as w01
for fn in ["push__img1024_dp01_seed86__model_swa@1024_flip.npy","push__img768_dp01_seed1024__model_swa@768_flip.npy","push__img768_dp01_seed42__model_swa@768_flip.npy","push__img896_dp01_seed86__model_swa@896_flip.npy"]:
    print("w1 vs w01 max|d|", fn, float(np.abs(ld(W1,fn)-ld(W01,fn)).max()))
arms = {
 "M4 WITH": (M["o42with"]+M["o1024with"]+M["r42with"]+M["r1024with"])/4,
 "M4 ZERO": (M["o42zero"]+M["o1024zero"]+M["r42zero"]+M["r1024zero"])/4,
 "Morig2 WITH": (M["o42with"]+M["o1024with"])/2, "Morig2 ZERO": (M["o42zero"]+M["o1024zero"])/2,
 "Mrerun2 WITH": (M["r42with"]+M["r1024with"])/2, "Mrerun2 ZERO": (M["r42zero"]+M["r1024zero"])/2,
 "TM2 (s42+s1024, 768)": (v42+v1024)/2,
 "TM4 fixed (w1)": (v42+v1024+v86_1024+v86_896)/4,
 "TM greedy3": (v42+v86_1024+v86_640)/3,
}
Lt = torch.from_numpy(L).long()
def tm_metrics(P):
    Pt = torch.from_numpy(np.ascontiguousarray(P))
    return dict(
      mAP=float(MultilabelAveragePrecision(num_labels=C)(Pt, Lt)),
      mAUC=float(MultilabelAUROC(num_labels=C)(Pt, Lt)),
      mF1=float(MultilabelF1Score(num_labels=C, average="macro", threshold=0.5)(Pt, Lt)),
      mECE=float(BinaryCalibrationError(n_bins=15, norm="l1")(Pt.flatten(), Lt.flatten())))
point = {}
for k,P in arms.items():
    point[k] = tm_metrics(P); print(f"{k:24s}", {a:round(b,4) for a,b in point[k].items()})

# ---------- fast weighted metrics
class Pre:
    def __init__(s, P):
        s.P = P
        s.order = np.argsort(P, axis=0, kind="stable")        # ascending
        s.grp = []
        for c in range(C):
            ps = P[s.order[:,c], c]
            new = np.r_[True, ps[1:] != ps[:-1]]
            s.grp.append(np.cumsum(new)-1)
        s.pred = (P >= 0.5)
        edges = np.linspace(0,1,16)
        s.bin = np.clip(np.searchsorted(edges, P, side="left")-1, 0, 14)   # tm: bucketize(right=True)? verified numerically below
def auc_ap(pre, w):
    auc = np.full(C, np.nan); ap = np.full(C, np.nan)
    for c in range(C):
        o = pre.order[:,c]; y = L[o,c]; ww = w[o]
        wp = ww*y; wn = ww-wp
        Wp, Wn = wp.sum(), wn.sum()
        if Wp==0 or Wn==0: continue
        g = pre.grp[c]; ng = g[-1]+1
        pg = np.bincount(g, wp, ng); nggr = np.bincount(g, wn, ng)
        cumn_before = np.cumsum(nggr)-nggr
        auc[c] = (pg*(cumn_before+0.5*nggr)).sum()/(Wp*Wn)
        # AP at distinct thresholds: descending score, tied groups merged
        pr = pg[::-1]; nr = nggr[::-1]
        tp = np.cumsum(pr); fp = np.cumsum(nr)
        prec = tp/np.maximum(tp+fp,1e-12)
        ap[c] = (pr*prec).sum()/Wp
    return np.nanmean(auc), np.nanmean(ap)
def f1_ece(pre, w):
    W = w.sum()
    f1 = np.zeros(C)
    for c in range(C):
        pr = pre.pred[:,c]; y = L[:,c].astype(bool)
        tp = (w*(pr&y)).sum(); fp = (w*(pr&~y)).sum(); fn = (w*(~pr&y)).sum()
        d = 2*tp+fp+fn
        f1[c] = 2*tp/d if d>0 else 0.0
    # L1 ECE over flattened
    ww = np.repeat(w[:,None], C, 1).ravel()
    b = pre.bin.ravel(); conf = pre.P.ravel(); acc = L.ravel()
    sb_conf = np.bincount(b, ww*conf, 15); sb_acc = np.bincount(b, ww*acc, 15)
    return f1.mean(), np.abs(sb_acc-sb_conf).sum()/(W*C)
PRE = {k: Pre(P) for k,P in arms.items()}
def allm(k, w):
    a,p = auc_ap(PRE[k], w); f,e = f1_ece(PRE[k], w)
    return dict(mAP=p, mAUC=a, mF1=f, mECE=e)
w1 = np.ones(N)
for k in arms:
    m = allm(k, w1)
    print(f"fast {k:24s}", {a:round(b,5) for a,b in m.items()}, "maxdiff vs torchmetrics", max(abs(m[a]-point[k][a]) for a in m))

pairs = [
 ("M4 WITH","TM2 (s42+s1024, 768)"), ("M4 ZERO","TM2 (s42+s1024, 768)"),
 ("Morig2 WITH","TM2 (s42+s1024, 768)"), ("Morig2 ZERO","TM2 (s42+s1024, 768)"),
 ("Mrerun2 WITH","TM2 (s42+s1024, 768)"), ("Mrerun2 ZERO","TM2 (s42+s1024, 768)"),
 ("M4 WITH","TM4 fixed (w1)"), ("M4 ZERO","TM4 fixed (w1)"),
 ("M4 WITH","TM greedy3"), ("M4 ZERO","TM greedy3"),
 ("TM4 fixed (w1)","TM2 (s42+s1024, 768)"),
 ("M4 WITH","M4 ZERO"),
]
B = int(sys.argv[1]); rng = np.random.default_rng(2026)
W = [np.bincount(rng.integers(0,N,N), minlength=N).astype(np.float64) for _ in range(B)]
out = {"point": point, "B": B, "pairs": {}}
t=time.time()
cache = {}
def bm(k):
    if k not in cache: cache[k] = [allm(k,w) for w in W]
    return cache[k]
for a,b in pairs:
    A, Bb = bm(a), bm(b)
    r = {}
    for met in ("mAP","mAUC","mF1","mECE"):
        d = np.array([x[met]-y[met] for x,y in zip(A,Bb)])
        r[met] = dict(point=point[a][met]-point[b][met], ci=[float(np.percentile(d,2.5)), float(np.percentile(d,97.5))], frac_le0=float((d<=0).mean()))
    out["pairs"][f"{a} - {b}"] = r
    print(f"{a} - {b}  ({time.time()-t:.0f}s)")
    for met,v in r.items(): print(f"   {met}: {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
json.dump(out, open(sys.argv[2],"w"), indent=1)
