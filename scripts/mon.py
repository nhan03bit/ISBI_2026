#!/home/psytp7/ISBI_2026/.venv/bin/python3
"""CLI monitor for ISBI 2026 jobs: running, pending and finished.

Parses the SLURM .out logs under ~/logs for the per-epoch
    Epoch NN/MM | train_loss=.. val_loss=.. | mAP=.. mAUC=.. mF1=.. mECE=..
lines, live tqdm progress, SWA / eval-only results and (Stage-2 v4) Markov gate
magnitudes. Job state comes from `squeue` (running/pending) and `sacct` (final state:
done / FAIL / OOM / TIME / CANC), falling back to the log when accounting has none.

    scripts/mon.py                      # every isbi2026_* job
    scripts/mon.py 'isbi2026_v3reg_*'   # custom glob (matched under $LOGDIR)
    scripts/mon.py --active             # running + pending only
    scripts/mon.py --failed             # FAIL / OOM / TIME / CANC / dead only
    scripts/mon.py --mk                 # Stage-2 v4 Markov-layer jobs (isbi2026_v4markov_*)
    scripts/mon.py --epochs             # also print each job's full epoch curve
    scripts/mon.py --all-runs           # keep reruns (default: newest job per arm+seed)
    watch -n 20 scripts/mon.py --active # live refresh
"""
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

LOGDIR = os.environ.get("LOGDIR", "/home/psytp7/logs")
FLAGS = ("--epochs", "--all-runs", "--mk", "--active", "--failed", "--all")

args = sys.argv[1:]
show_curve = "--epochs" in args
all_runs = "--all-runs" in args
mk_only = "--mk" in args
only_active = "--active" in args
only_failed = "--failed" in args
pos = [a for a in args if a not in FLAGS]
pat = "isbi2026_v4markov_*" if mk_only else "isbi2026_*"   # --all == default
if pos:
    pat = pos[0]

# Markov jobs (scripts/submit_stage2_v4_markov.sh) are matched to this v3 run:
# push/img768_dp01_seed42, same recipe and seed without the layer.
MK_BASELINE = ("push img768_dp01_s42", "isbi2026_v3push_img768_dp01_s42_49000.out")

EP = re.compile(
    r"Epoch (\d+)/(\d+) \| train_loss=([\d.]+) val_loss=([\d.]+) \| "
    r"mAP=([\d.]+) mAUC=([\d.]+) mF1=([\d.]+) mECE=([\d.]+)"
)
TQDM = re.compile(
    r"(Train|Eval)\s+(\d+)/(\d+):\s+(\d+)it \[([\d:]+),\s+([\d.]+)(it/s|s/it)\]"
)
BEST_ES = re.compile(r"Best (?:mAP|mAUC|mF1|mECE|val_loss)=([\d.]+) at epoch (\d+)")
BEST_DONE = re.compile(r"Done\. Best epoch: (\d+) \| Best \w+: ([\d.]+)")
SWA = re.compile(r"SWA weights -> .*? mAP=([\d.]+)")
EVAL = re.compile(r"val_mAP\"?:\s+([\d.]+)")          # evaluate*.py: `val_mAP: x` / `"val_mAP": x`
# train_2_v4.py logs one of these per epoch, just before the "Epoch NN/MM" line.
MK = re.compile(r"markov: steps=(\d+) \|gate\| mean=([\d.]+) max=([\d.]+)")
MK_ARGS = re.compile(r"Markov label-refine layer: steps=(\d+) learn_transition=(\w+)")

SACCT_SHORT = {"COMPLETED": "done", "RUNNING": "R", "PENDING": "PD", "FAILED": "FAIL",
               "OUT_OF_MEMORY": "OOM", "TIMEOUT": "TIME", "CANCELLED": "CANC",
               "NODE_FAIL": "NODEFAIL", "PREEMPTED": "PREEMPT", "REQUEUED": "REQUEUE"}
FAILED_STATES = {"FAIL", "OOM", "TIME", "CANC", "NODEFAIL", "PREEMPT", "dead"}
STATE_RANK = {"R": 0, "CG": 0, "PD": 1}
# squeue pending reasons, shortened for the NODE column.
PD_REASON = {"QOSMaxMemoryPerUser": "QOS-mem", "QOSMaxGRESPerUser": "QOS-gpu",
             "QOSMaxCpuPerUserLimit": "QOS-cpu", "QOSMaxJobsPerUserLimit": "QOS-jobs",
             "ReqNodeNotAvail": "node-n/a", "Dependency": "dependency"}

# --- squeue: live state ------------------------------------------------------
state, where, qname = {}, {}, {}
try:
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%t|%R|%j"],
                         capture_output=True, text=True, timeout=15).stdout
    for ln in out.splitlines():
        p = ln.split("|")
        if len(p) >= 4:
            state[p[0]], where[p[0]], qname[p[0]] = p[1], p[2], p[3]
except Exception as e:  # noqa: BLE001
    print(f"(squeue unavailable: {e})", file=sys.stderr)


def parse_name(b):
    """job name / log basename (no .out, optional trailing _<jobid>) -> (arm, jobid, seed).
    A _smoke suffix after the seed stays on the arm so smoke and full runs never share a row."""
    ms = re.match(r"isbi2026_v3seed_s(\d+)(?:_(\d+))?$", b)
    if ms:
        return f"seed{ms.group(1)}", ms.group(2), ms.group(1)
    m = re.match(r"isbi2026_(?:v3s2_)?(.+?)_s(\d+)(_smoke)?(?:_(\d+))?$", b)
    if m:
        return m.group(1) + (m.group(3) or ""), m.group(4), m.group(2)
    m = re.match(r"(?:isbi2026_)?(.+?)(?:_(\d+))?$", b)
    return m.group(1), m.group(2), None


def fmt_elapsed(s):
    """sacct Elapsed '[D-]HH:MM:SS' -> compact '1d02h', '5h03m', '12m'."""
    if not s:
        return "-"
    d, _, hms = s.rpartition("-")
    h, m, _sec = (int(x) for x in hms.split(":"))
    d = int(d) if d else 0
    if d:
        return f"{d}d{h:02d}h"
    return f"{h}h{m:02d}m" if h else f"{m}m"


# --- parse logs ---------------------------------------------------------------
files = sorted(glob.glob(f"{LOGDIR}/{pat}_*.out")) or sorted(glob.glob(f"{LOGDIR}/{pat}.out"))
rows = []
for f in files:
    arm, jid, seed = parse_name(os.path.basename(f)[:-4])
    if jid is None:
        continue
    txt = open(f, errors="replace").read().replace("\r", "\n")

    epochs = [(int(e[0]), int(e[1]), float(e[2]), float(e[3]),
               float(e[4]), float(e[5]), float(e[6]), float(e[7])) for e in EP.findall(txt)]
    best_map = best_ep = None
    for ep in epochs:
        if best_map is None or ep[4] > best_map:
            best_map, best_ep = ep[4], ep[0]
    md, me = BEST_DONE.search(txt), BEST_ES.search(txt)
    if md:
        best_ep, best_map = int(md.group(1)), float(md.group(2))
    elif me:
        best_map, best_ep = float(me.group(1)), int(me.group(2))
    sw = SWA.search(txt)
    ev = EVAL.findall(txt) if not epochs else []

    tq = None
    for tq in TQDM.finditer(txt):
        pass

    note = []
    if "Device: cpu" in txt:
        note.append("!!CPU-FALLBACK")
    if "CUDA unknown error" in txt:
        note.append("cuda-init-fail")
    for k in ("Traceback (most recent call last)", "slurmstepd: error", "srun: error",
              "CANCELLED", "OUT_OF_MEMORY", "CUDA out of memory"):
        if k in txt:
            note.append("ERR")
            break
    if "Early stopping triggered" in txt:
        note.append("early-stop")
    if ev:
        note.append("eval")

    # Markov layer (v4): gate magnitudes per epoch; ~0 means the layer is unused.
    mk_eps = [(int(a), float(b), float(c)) for a, b, c in MK.findall(txt)]
    mk_cfg = MK_ARGS.search(txt)
    mk = None
    if mk_cfg:
        mk = f"K{mk_cfg.group(1)}{'' if mk_cfg.group(2) == 'True' else 'fix'}"
        if mk_eps:
            mk += f" |g| {mk_eps[-1][1]:.3f}/{mk_eps[-1][2]:.3f}"

    finished = "=== Finished" in txt or md is not None
    rows.append(dict(jid=jid, arm=arm, seed=seed, log_state="done" if finished else "dead",
                     epochs=epochs, tq=tq, best_map=best_map, best_ep=best_ep,
                     swa=float(sw.group(1)) if sw else None,
                     eval_map=float(ev[-1]) if ev else None,
                     note=note, mk=mk, mk_eps=mk_eps))

# Queued / just-started jobs with no log yet: stub rows from squeue.
pat_re = re.compile("^" + re.escape(pat).replace(r"\*", ".*") + "$")
seen = {r["jid"] for r in rows}
for jid, nm in qname.items():
    if jid in seen or not pat_re.match(nm):
        continue
    arm, _, seed = parse_name(nm)
    rows.append(dict(jid=jid, arm=arm, seed=seed, log_state="PD", epochs=[], tq=None,
                     best_map=None, best_ep=None, swa=None, eval_map=None, note=[],
                     mk="v4" if "v4markov" in nm else None, mk_eps=[]))

# --- sacct: final state + elapsed for everything not live in squeue ------------
acct = {}
ids = [r["jid"] for r in rows]
for i in range(0, len(ids), 200):
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids[i:i + 200]),
                              "-o", "JobID,State,Elapsed"],
                             capture_output=True, text=True, timeout=30).stdout
        for ln in out.splitlines():
            p = ln.split("|")
            if len(p) >= 3:
                acct[p[0]] = (p[1].split()[0], p[2])
    except Exception as e:  # noqa: BLE001
        print(f"(sacct unavailable: {e})", file=sys.stderr)
        break

for r in rows:
    live = state.get(r["jid"])
    a_state, a_elapsed = acct.get(r["jid"], (None, None))
    if live:
        r["st"] = live
    elif a_state:
        r["st"] = SACCT_SHORT.get(a_state, a_state[:8])
    else:
        r["st"] = r["log_state"]
    r["elapsed"] = fmt_elapsed(a_elapsed) if a_elapsed and r["st"] != "PD" else "-"
    r["node"] = where.get(r["jid"], "-") if live else "-"
    if r["st"] == "PD":
        reason = r["node"].strip("()").split(",")[0]
        r["node"] = PD_REASON.get(reason.split(" ")[0], reason)
    if r["st"] in FAILED_STATES and "ERR" in r["note"]:
        r["note"].remove("ERR")          # the state already says why

    running = r["st"] in ("R", "CG")
    last = r["epochs"][-1] if r["epochs"] else None
    r["last"] = last
    r["ep"] = f"{last[0]}/{last[1]}" if last else "-"
    r["prog"] = ""
    if running and r["tq"]:
        phase, a_, b_, it, el, _rate, _unit = r["tq"].groups()
        r["ep"] = "SWA" if a_ == "0" and last else f"{a_}/{b_}"   # epoch 0 = post-fit SWA eval
        parts = [int(x) for x in el.split(":")]
        secs = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
        r["prog"] = f"{phase[:2]} {int(it):>5}it @ {(int(it) / secs) if secs else 0.0:4.1f}it/s"

# Default: one row per (arm, seed), newest job id. --all-runs keeps every log.
n_total = len(rows)
if not all_runs:
    newest = {}
    for r in rows:
        key = (r["arm"], r["seed"])
        if key not in newest or int(r["jid"]) > int(newest[key]["jid"]):
            newest[key] = r
    rows = list(newest.values())
n_hidden = n_total - len(rows)
# Arms run with several seeds: show the seed so the rows are distinguishable.
seeds_per_arm = {}
for r in rows:
    seeds_per_arm.setdefault(r["arm"], set()).add(r["seed"])
for r in rows:
    if len(seeds_per_arm[r["arm"]]) > 1 and r["seed"]:
        r["arm"] = f"{r['arm']}_s{r['seed']}"

counts = {}
for r in rows:
    k = "running" if r["st"] in ("R", "CG") else "pending" if r["st"] == "PD" else \
        "failed" if r["st"] in FAILED_STATES else "done"
    counts[k] = counts.get(k, 0) + 1
if only_active:
    rows = [r for r in rows if r["st"] in ("R", "CG", "PD")]
if only_failed:
    rows = [r for r in rows if r["st"] in FAILED_STATES]
rows.sort(key=lambda r: (STATE_RANK.get(r["st"], 2), r["arm"]))

# --- render --------------------------------------------------------------------
if not rows:
    sys.exit(f"no jobs match '{pat}'" + (" with that filter" if only_active or only_failed else ""))

# ep·* = LATEST completed epoch; BEST = early-stopping metric (max over epochs @ epoch),
# or the scored mAP of an eval-only job; SWA = mAP of the SWA-averaged weights.
# MARKOV (only when a v4 Markov job is listed): K steps (+"fix" = frozen transition)
# and the latest per-class gate |g| mean/max - ~0 means the layer is not being used.
show_mk = any(r.get("mk") for r in rows)
hdr = ["JOB", "ARM", "ST", "NODE", "TIME", "EPOCH", "PROGRESS",
       "ep·mAP", "ep·mF1", "ep·mECE", "ep·vl", "BEST mAP", "SWA"] + \
      (["MARKOV"] if show_mk else []) + ["NOTE"]


def cells(r):
    l = r["last"]
    if r["best_map"] is not None:
        best = f"{r['best_map']:.4f}@{r['best_ep']}"
    elif r["eval_map"] is not None:
        best = f"{r['eval_map']:.4f} ev"
    else:
        best = "-"
    return [
        r["jid"], r["arm"], r["st"], (r["node"] or "-")[:12], r["elapsed"], r["ep"], r["prog"],
        f"{l[4]:.4f}" if l else "-",
        f"{l[6]:.4f}" if l else "-",
        f"{l[7]:.4f}" if l else "-",
        f"{l[3]:.4f}" if l else "-",
        best,
        f"{r['swa']:.4f}" if r["swa"] is not None else "-",
    ] + ([r.get("mk") or "-"] if show_mk else []) + [" ".join(dict.fromkeys(r["note"]))]


table = [hdr] + [cells(r) for r in rows]
w = [max(len(str(x)) for x in col) for col in zip(*table)]
summary = " | ".join(f"{counts.get(k, 0)} {k}" for k in ("running", "pending", "done", "failed"))
print(f"ISBI2026 monitor  —  {datetime.now():%F %T}  —  {summary}  ({pat})"
      + (f"  —  {n_hidden} older reruns hidden (--all-runs)" if n_hidden else "") + "\n")
for i, row in enumerate(table):
    print("  ".join(str(x).ljust(w[j]) for j, x in enumerate(row)).rstrip())
    if i == 0:
        print("  ".join("-" * w[j] for j in range(len(w))))

if show_curve:
    for r in rows:
        if r["epochs"]:
            print(f"\n{r['arm']} (job {r['jid']}):")
            gates = r.get("mk_eps") or []
            for i, e in enumerate(r["epochs"]):
                star = "  <- best" if e[0] == r["best_ep"] else ""
                g = f"  |g|={gates[i][1]:.3f}/{gates[i][2]:.3f}" if i < len(gates) else ""
                print(f"  ep {e[0]:>2}/{e[1]}  mAP={e[4]:.4f}  mAUC={e[5]:.4f}  "
                      f"mF1={e[6]:.4f}  mECE={e[7]:.4f}  "
                      f"train={e[2]:.4f} val={e[3]:.4f}{g}{star}")
            if r["swa"] is not None:
                print(f"  SWA   mAP={r['swa']:.4f}")

# Best training run (excluding smoke runs) and best eval-only job (TTA / ensembles).
trained = [r for r in rows if r["best_map"] is not None and "smoke" not in r["arm"]
           and not only_failed]
if trained:
    top = max(trained, key=lambda r: max(r["best_map"], r["swa"] or 0))
    swa_txt = f", SWA {top['swa']:.4f}" if top["swa"] is not None else ""
    print(f"\nbest run    : {top['arm']}  mAP {top['best_map']:.4f} @ep{top['best_ep']}{swa_txt}  (job {top['jid']})")
evals = [r for r in rows if r["eval_map"] is not None]
if evals:
    top = max(evals, key=lambda r: r["eval_map"])
    print(f"best eval   : {top['arm']}  mAP {top['eval_map']:.4f}  (job {top['jid']}; TTA/ensemble eval job)")
repro = next((r for r in rows if r["arm"] == "bug_repro" and r["best_map"] is not None), None)
if repro:
    print(f"gate (repro): bug_repro mAP {repro['best_map']:.4f}  "
          f"(Δ vs 0.435 baseline = {repro['best_map'] - 0.435:+.4f})")

# Markov jobs: delta vs the matched no-layer baseline (same recipe + seed).
mk_rows = [r for r in rows if r.get("mk") and r["best_map"] is not None and "smoke" not in r["arm"]]
bl_path = os.path.join(LOGDIR, MK_BASELINE[1])
if mk_rows and os.path.exists(bl_path):
    bt = open(bl_path, errors="replace").read()
    bd, bs = BEST_DONE.search(bt), SWA.search(bt)
    if bd:
        b_best, b_swa = float(bd.group(2)), float(bs.group(1)) if bs else None
        print(f"\nmarkov vs baseline {MK_BASELINE[0]} (best {b_best:.4f}"
              + (f", SWA {b_swa:.4f}" if b_swa is not None else "") + "):")
        for r in sorted(mk_rows, key=lambda r: r["arm"]):
            line = f"  {r['arm']:<22} best {r['best_map']:.4f} ({r['best_map'] - b_best:+.4f})"
            if r.get("swa") is not None and b_swa is not None:
                line += f"   SWA {r['swa']:.4f} ({r['swa'] - b_swa:+.4f})"
            print(line)
