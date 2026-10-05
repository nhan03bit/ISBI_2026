import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/psytp7/ISBI_2026")
sys.path.insert(0, str(ROOT/"analysis")); sys.path.insert(0, str(ROOT/"evaluate"))
import markov_check as mc, patient_markov_check as pmc, patient_prior as pp
from class_groups import frequency_groups, load_train_prevalence
from ens_select import macro_map

OUT = Path(sys.argv[1])
O = ROOT/"analysis/out"
AB, OR, RR, W01 = O/"probs_ab_flip", O/"probs_v5_orig_recheck_flip", O/"probs_v5_rerun_flip", O/"probs_w01_flip"
keys = (AB/"keys.txt").read_text().split("\n")
for d in (OR, RR):
    assert (d/"keys.txt").read_text().split("\n") == keys
L = np.load(AB/"labels.npy").astype(np.uint8)
for d in (OR, RR, W01):
    assert (np.load(d/"labels.npy") == L).all(), d
classes = list(torch.load(pp.LABEL_INFO, weights_only=False)["class_names"])
_, prev, _ = load_train_prevalence(classes)
grp = frequency_groups(prev)
gname = np.where(grp["head"], "head", np.where(grp["tail"], "tail", "medium"))
tr, va_full = pp.load_folds(classes)
sg, _ = pp.study_table(tr, va_full)
pos = {f:i for i,f in enumerate(va_full["filename"])}
va = va_full.iloc[[pos[k] for k in keys]].reset_index(drop=True)
assert (pp.decode(va["code"].to_numpy()) == L).all()
order = np.load(mc.KEY_ORDER)
print("keys == val_key_order:", va_full["filename"].iloc[order].tolist() == keys)
pri = torch.load(pp.PRIORS, weights_only=False)
ridx = {n:i for i,n in enumerate(pri["val"]["fnames"])}
r = pri["val"]["r"][[ridx[k] for k in keys]]
has = r[:,30] == 1
tr_pids = set(tr["pid"]); tr_sids = set(tr["sid"])
frontal = va["proj"].isin(mc.FRONTAL).to_numpy()
p_unseen = ~va["pid"].isin(tr_pids).to_numpy()
s_in = va["sid"].isin(tr_sids).to_numpy()
pid = va["pid"].to_numpy()
subsets = {"all": np.ones(len(va),bool), "frontal": frontal, "has_prior": has, "no_prior": ~has,
           "has_prior&frontal": has&frontal, "patient_unseen": p_unseen, "frontal&patient_unseen": frontal&p_unseen,
           "patient_seen": ~p_unseen, "frontal&patient_seen": frontal&~p_unseen}
print({k:int(v.sum()) for k,v in subsets.items()})

# ---- load arms
def ld(d, f): return np.load(d/f)
M = {}
M["o42w"]=ld(OR,"prior__img768_qprior_seed42__model_swa@768_flip_prior-with.npy")
M["o1024w"]=ld(OR,"prior__img768_qprior_seed1024__model_swa@768_flip_prior-with.npy")
M["o42z"]=ld(OR,"prior__img768_qprior_seed42__model_swa@768_flip_prior-zero.npy")
M["o1024z"]=ld(OR,"prior__img768_qprior_seed1024__model_swa@768_flip_prior-zero.npy")
M["r42w"]=ld(RR,"prior__img768_qprior_seed42_rerun__model_swa@768_flip_prior-with.npy")
M["r1024w"]=ld(RR,"prior__img768_qprior_seed1024_rerun__model_swa@768_flip_prior-with.npy")
M["r42z"]=ld(RR,"prior__img768_qprior_seed42_rerun__model_swa@768_flip_prior-zero.npy")
M["r1024z"]=ld(RR,"prior__img768_qprior_seed1024_rerun__model_swa@768_flip_prior-zero.npy")
M["v3_42"]=ld(AB,"push__img768_dp01_seed42__model_swa@768_flip.npy")
M["v3_1024"]=ld(AB,"push__img768_dp01_seed1024__model_swa@768_flip.npy")
# orig dumps identical to 2026-10-01 dumps?
for k,f in [("o42w","prior__img768_qprior_seed42__model_swa@768_flip_prior-with.npy"),("o1024z","prior__img768_qprior_seed1024__model_swa@768_flip_prior-zero.npy")]:
    print("orig recheck vs 10-01 dump max|d|", k, float(np.abs(M[k]-ld(AB,f)).max()))
# v3 greedy-3 members from probs_w01_flip
g3 = [ld(W01,"push__img768_dp01_seed42__model_swa@768_flip.npy"), ld(W01,"push__img1024_dp01_seed86__model_swa@1024_flip.npy"), ld(W01,"res_sweep__img640_seed86__model_best@640_flip.npy")]
print("v3 s42 w01 vs ab max|d|", float(np.abs(g3[0]-M["v3_42"]).max()))
ens = {
 "v5 orig WITH": (M["o42w"]+M["o1024w"])/2, "v5 orig ZERO": (M["o42z"]+M["o1024z"])/2,
 "v5 rerun WITH": (M["r42w"]+M["r1024w"])/2, "v5 rerun ZERO": (M["r42z"]+M["r1024z"])/2,
 "v5 4-model WITH": (M["o42w"]+M["o1024w"]+M["r42w"]+M["r1024w"])/4,
 "v5 4-model ZERO": (M["o42z"]+M["o1024z"]+M["r42z"]+M["r1024z"])/4,
 "v3 matched 2-seed": (M["v3_42"]+M["v3_1024"])/2,
 "v3 greedy-3": sum(g3)/3,
}
Lt = torch.from_numpy(L).long()
res = {"subset_n": {k:int(v.sum()) for k,v in subsets.items()}, "mAP": {}}
print("\nfull-val mAP (recomputed)")
for k,P in ens.items(): print(f"  {k:22s} {macro_map(np.ascontiguousarray(P), Lt):.4f}")
for k in ["o42w","o1024w","o42z","o1024z","r42w","r1024w","r42z","r1024z","v3_42","v3_1024"]:
    print(f"  single {k:8s} {macro_map(np.ascontiguousarray(M[k]), Lt):.4f}")

# ---- mAP by subset
allarms = {**{k:v for k,v in ens.items()}}
print("\nmAP by subset")
print(f"{'arm':22s}" + "".join(f"{s[:16]:>18s}" for s in subsets))
for k,P in allarms.items():
    row=[]
    for s,m in subsets.items():
        row.append(pmc.subset_map(P, L, np.where(m)[0])[0])
    res["mAP"][k]=dict(zip(subsets,row))
    print(f"{k:22s}" + "".join(f"{x:18.4f}" for x in row))

# ---- fast weighted-AP bootstrap
N=len(L)
def prep(P):
    o = np.argsort(-P, axis=0, kind="stable")  # [N,C]
    return o
def ap_w(order, w):
    # returns per-class AP given row weights w (N,), classes with no positive -> nan
    C = L.shape[1]; out = np.full(C, np.nan)
    for c in range(C):
        oc = order[:,c]; wy = w[oc]*L[oc,c]; ww = w[oc]
        tp = np.cumsum(wy); n = np.cumsum(ww)
        tot = wy.sum()
        if tot == 0: continue
        out[c] = (wy * (tp/np.maximum(n,1e-12))).sum()/tot
    return out
ORD = {}
def get_order(name):
    if name not in ORD: ORD[name]=prep(ens[name])
    return ORD[name]
# sanity vs torchmetrics
for nm in ("v5 rerun WITH","v3 matched 2-seed"):
    a = ap_w(get_order(nm), np.ones(N)); print("sanity fastAP", nm, np.nanmean(a), macro_map(np.ascontiguousarray(ens[nm]), Lt))

def boot(a, b, mask, B=1000, cluster=True, seed=0, groups=None):
    """paired bootstrap of macro-AP difference a-b on rows in mask"""
    rng = np.random.default_rng(seed)
    oa, ob = get_order(a), get_order(b)
    rows = np.where(mask)[0]
    if cluster:
        import pandas as pd
        s = pd.Series(rows, index=pid[rows]); gl=[g.to_numpy() for _,g in s.groupby(level=0)]
        k=len(gl)
    vals=[]; per=[]
    for _ in range(B):
        w = np.zeros(N)
        if cluster:
            idx = rng.integers(0,k,k)
            cnt = np.bincount(idx, minlength=k)
            for j in np.nonzero(cnt)[0]: w[gl[j]] += cnt[j]
        else:
            idx = rng.integers(0,len(rows),len(rows)); np.add.at(w, rows[idx], 1)
        apa, apb = ap_w(oa,w), ap_w(ob,w)
        vals.append(np.nanmean(apa)-np.nanmean(apb))
    vals=np.array(vals)
    wp = mask.astype(float)
    pa, pb = ap_w(oa,wp), ap_w(ob,wp)
    return {"point": float(np.nanmean(pa)-np.nanmean(pb)), "boot_mean": float(vals.mean()),
            "ci95":[float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))],
            "p_le0": float((vals<=0).mean())}, pa-pb

pairs = [
 ("v5 rerun WITH","v5 orig WITH"), ("v5 rerun ZERO","v5 orig ZERO"),
 ("v5 rerun WITH","v3 matched 2-seed"), ("v5 rerun ZERO","v3 matched 2-seed"),
 ("v5 orig WITH","v3 matched 2-seed"), ("v5 orig ZERO","v3 matched 2-seed"),
 ("v5 4-model WITH","v3 matched 2-seed"), ("v5 4-model ZERO","v3 matched 2-seed"),
 ("v5 rerun ZERO","v3 greedy-3"), ("v5 orig ZERO","v3 greedy-3"), ("v5 4-model ZERO","v3 greedy-3"),
 ("v5 rerun WITH","v5 rerun ZERO"), ("v5 orig WITH","v5 orig ZERO"),
]
mask_sets = {"all": subsets["all"], "has_prior&frontal": subsets["has_prior&frontal"], "no_prior": subsets["no_prior"],
             "frontal&patient_unseen": subsets["frontal&patient_unseen"], "frontal": subsets["frontal"]}
res["boot"] = {}; per_class = {}
B = int(sys.argv[2]) if len(sys.argv)>2 else 300
print(f"\npaired bootstrap B={B}; patient-clustered (PC) and image-level (IM)")
for a,b in pairs:
    for mn, m in mask_sets.items():
        for cl in (True, False):
            if mn not in ("all",) and not cl: continue
            e, d = boot(a,b,m,B=B,cluster=cl,seed=7)
            key=f"{a} - {b} | {mn} | {'PC' if cl else 'IM'}"
            res["boot"][key]=e
            print(f"{key:80s} {e['point']:+.4f} [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]")
            if mn=="all" and cl: per_class[f"{a} - {b}"]=d
res["per_class"]={"classes":classes,"group":gname.tolist(),"train_prev":prev.tolist(),"d":{k:[None if np.isnan(x) else float(x) for x in v] for k,v in per_class.items()}}
# group summaries
print("\nper-class dAP by group (full val)")
for k,d in per_class.items():
    s=[]
    for g in ("head","medium","tail"):
        m=gname==g; s.append(f"{g} {np.nanmean(d[m]):+.4f} ({int((d[m]>0).sum())}/{int(m.sum())} up)")
    print(f"  {k:48s} "+" | ".join(s))
json.dump(res, open(OUT,"w"), indent=1)
