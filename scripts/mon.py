#!/home/psytp7/ISBI_2026/.venv/bin/python3
"""CLI monitor for ISBI 2026 Stage-2 training jobs.

Parses the SLURM .out logs under ~/logs for the per-epoch
    Epoch NN/MM | train_loss=.. val_loss=.. | mAP=.. mAUC=.. mF1=.. mECE=..
lines plus live tqdm step progress, merges in `squeue` state, and prints one
table with best-mAP-so-far per job.

    scripts/mon.py                      # current v3s2 seed-86 sweep
    scripts/mon.py 'isbi2026_v3s2_*'    # custom glob (matched under $LOGDIR)
    scripts/mon.py --all                # every isbi2026_* log
    scripts/mon.py --epochs             # also print each job's full epoch curve
    watch -n 20 scripts/mon.py          # live refresh
"""
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

LOGDIR = os.environ.get("LOGDIR", "/home/psytp7/logs")

args = sys.argv[1:]
show_curve = "--epochs" in args
all_runs = "--all-runs" in args  # keep every log, not just newest per arm
args = [a for a in args if a not in ("--epochs", "--all-runs")]
pat = "isbi2026_v3*" if "--all" in args else "isbi2026_*"  # v3s2_* + v3seed_*
args = [a for a in args if a != "--all"]
if args:
    pat = args[0]

EP = re.compile(
    r"Epoch (\d+)/(\d+) \| train_loss=([\d.]+) val_loss=([\d.]+) \| "
    r"mAP=([\d.]+) mAUC=([\d.]+) mF1=([\d.]+) mECE=([\d.]+)"
)
TQDM = re.compile(
    r"(Train|Eval)\s+(\d+)/(\d+):\s+(\d+)it \[([\d:]+),\s+([\d.]+)(it/s|s/it)\]"
)
BEST_ES = re.compile(r"Best (?:mAP|mAUC|mF1|mECE|val_loss)=([\d.]+) at epoch (\d+)")
BEST_DONE = re.compile(r"Done\. Best epoch: (\d+) \| Best \w+: ([\d.]+)")

# --- squeue state ---------------------------------------------------------
state, where, qname = {}, {}, {}
try:
    out = subprocess.run(
        ["squeue", "--me", "-h", "-o", "%i|%t|%R|%j"],
        capture_output=True, text=True, timeout=15,
    ).stdout
    for ln in out.splitlines():
        p = ln.split("|")
        if len(p) >= 2:
            state[p[0]] = p[1]
            where[p[0]] = p[2] if len(p) > 2 else "-"
            qname[p[0]] = p[3] if len(p) > 3 else ""
except Exception as e:  # noqa: BLE001
    print(f"(squeue unavailable: {e})", file=sys.stderr)

# --- parse logs ----------------------------------------------------------
files = sorted(glob.glob(f"{LOGDIR}/{pat}_*.out")) or sorted(glob.glob(f"{LOGDIR}/{pat}.out"))

rows = []
def parse_name(b):
    """log basename (no .out) -> (arm_label, jobid)."""
    ms = re.match(r"isbi2026_v3seed_s(\d+)_(\d+)$", b)
    if ms:
        return f"seed{ms.group(1)}", ms.group(2)
    m = re.match(r"isbi2026_(?:v3s2_)?(.+?)_s\d+_(\d+)$", b) or re.match(r"(.+)_(\d+)$", b)
    return (m.group(1), m.group(2)) if m else (b, "?")


for f in files:
    b = os.path.basename(f)[:-4]
    arm, jid = parse_name(b)
    txt = open(f, errors="replace").read().replace("\r", "\n")

    epochs = [
        (int(e[0]), int(e[1]), float(e[2]), float(e[3]),
         float(e[4]), float(e[5]), float(e[6]), float(e[7]))
        for e in EP.findall(txt)
    ]
    last = epochs[-1] if epochs else None
    best_map = best_ep = None
    for ep in epochs:
        if best_map is None or ep[4] > best_map:
            best_map, best_ep = ep[4], ep[0]
    md, me = BEST_DONE.search(txt), BEST_ES.search(txt)
    if md:
        best_ep, best_map = int(md.group(1)), float(md.group(2))
    elif me:
        best_map, best_ep = float(me.group(1)), int(me.group(2))

    tq = None
    for tq in TQDM.finditer(txt):
        pass
    cur_ep = f"{last[0]}/{last[1]}" if last else "-"
    prog = ""
    if tq:
        phase, a_, b_, it, el, _rate, _unit = tq.groups()
        cur_ep = f"{a_}/{b_}"
        parts = [int(x) for x in el.split(":")]
        secs = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
        avg = (int(it) / secs) if secs else 0.0
        prog = f"{phase[:2]} {int(it):>5}it @ {avg:4.1f}it/s"

    note = []
    if "Device: cpu" in txt:
        note.append("!!CPU-FALLBACK")
    if "CUDA unknown error" in txt:
        note.append("cuda-init-fail")
    for k in ("Traceback (most recent call last)", "slurmstepd: error",
              "srun: error", "CANCELLED", "OUT_OF_MEMORY", "CUDA out of memory"):
        if k in txt:
            note.append("ERR")
            break
    if "Early stopping triggered" in txt:
        note.append("early-stop")
    if "=== Finished" in txt or md:
        note.append("done")

    st = state.get(jid) or ("done" if ("=== Finished" in txt or md) else "dead")
    rows.append(dict(jid=jid, arm=arm, st=st, node=where.get(jid, "-"),
                     ep=cur_ep, prog=prog, last=last,
                     best_map=best_map, best_ep=best_ep,
                     epochs=epochs, note=" ".join(dict.fromkeys(note))))

# Queued jobs with no log yet: add a stub row from squeue (only ones whose job
# name is consistent with the requested glob).
pat_re = re.compile("^" + re.escape(pat).replace(r"\*", ".*"))
seen_jids = {r["jid"] for r in rows}
for jid, nm in qname.items():
    if jid in seen_jids or not pat_re.match(nm):
        continue
    ms = re.match(r"isbi2026_v3seed_s(\d+)$", nm)
    qm = re.match(r"isbi2026_(?:v3s2_)?(.+?)_s\d+$", nm)
    arm = f"seed{ms.group(1)}" if ms else (qm.group(1) if qm else None)
    if arm is None:
        continue
    rows.append(dict(jid=jid, arm=arm, st=state.get(jid, "PD"),
                     node=where.get(jid, "-"), ep="-", prog="", last=None,
                     best_map=None, best_ep=None, epochs=[], note="queued"))

# Default: one row per arm (newest job id). --all-runs keeps every log.
if not all_runs:
    newest = {}
    for r in rows:
        cur = newest.get(r["arm"])
        if cur is None or int(r["jid"]) > int(cur["jid"]):
            newest[r["arm"]] = r
    rows = list(newest.values())
rows.sort(key=lambda r: (r["st"] in ("done", "dead"), r["arm"]))

# --- render ------------------------------------------------------------
if not rows:
    sys.exit(f"no logs or queued jobs match '{pat}'")

# mAP/mF1/mECE/vloss below are the LATEST completed epoch; "BEST mAP" is the
# early-stopping metric (max over epochs, @ its epoch).
hdr = ["JOB", "ARM", "ST", "NODE", "EPOCH", "PROGRESS",
       "ep·mAP", "ep·mF1", "ep·mECE", "ep·vl", "BEST mAP", "NOTE"]


def cells(r):
    l = r["last"]
    return [
        r["jid"], r["arm"], r["st"], (r["node"] or "-")[:12], r["ep"], r["prog"],
        f"{l[4]:.4f}" if l else "-",
        f"{l[6]:.4f}" if l else "-",
        f"{l[7]:.4f}" if l else "-",
        f"{l[3]:.4f}" if l else "-",
        f"{r['best_map']:.4f}@{r['best_ep']}" if r["best_map"] is not None else "-",
        r["note"],
    ]


table = [hdr] + [cells(r) for r in rows]
w = [max(len(str(x)) for x in col) for col in zip(*table)]
print(f"ISBI2026 stage-2 monitor  —  {datetime.now():%F %T}  —  "
      f"{len(rows)} jobs  ({pat})\n")
for i, row in enumerate(table):
    print("  ".join(str(x).ljust(w[j]) for j, x in enumerate(row)).rstrip())
    if i == 0:
        print("  ".join("-" * w[j] for j in range(len(w))))

if show_curve:
    for r in rows:
        if r["epochs"]:
            print(f"\n{r['arm']} (job {r['jid']}):")
            for e in r["epochs"]:
                star = "  <- best" if e[0] == r["best_ep"] else ""
                print(f"  ep {e[0]:>2}/{e[1]}  mAP={e[4]:.4f}  mAUC={e[5]:.4f}  "
                      f"mF1={e[6]:.4f}  mECE={e[7]:.4f}  "
                      f"train={e[2]:.4f} val={e[3]:.4f}{star}")

scored = [r for r in rows if r["best_map"] is not None]
if scored:
    top = max(scored, key=lambda r: r["best_map"])
    print(f"\nbest so far : {top['arm']}  mAP {top['best_map']:.4f} @ep{top['best_ep']}  (job {top['jid']})")
    repro = next((r for r in rows if r["arm"] == "bug_repro" and r["best_map"] is not None), None)
    if repro:
        print(f"gate (repro): bug_repro mAP {repro['best_map']:.4f}  "
              f"(Δ vs 0.435 baseline = {repro['best_map'] - 0.435:+.4f})")
