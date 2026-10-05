#!/usr/bin/env python3
"""speed_summary.py — the table behind PORTING-SPEC's speed section, from a `bench/speed.sh` output root.

Per arm: gate state, whole-run s/frame (pipeline only — no PNG writing, no LQ preparation, no weight load, the same
region the upstream torch runner times), steady-state s/frame (chunks after the first: 8 frames each, past the
first chunk's 25-frame start and kernel compilation), peak phys. Medians per configuration (the label minus its
trailing a/b). Engine arms are end to end (decode + upscale + HEVC encode).

usage: python3 bench/speed_summary.py <outRoot> [--csv out.csv]
"""
import csv, json, os, re, statistics, sys

root = sys.argv[1]
rows = []
gates = {}
for line in open(os.path.join(root, "speed.log")):
    m = re.search(r"start (\S+)", line)
    if m and gates.get("_last"):
        gates[m.group(1)] = gates.pop("_last")
    m = re.search(r"gate passed: gpu (\d+)% thermal (\w+)", line)
    if m:
        gates["_last"] = (int(m.group(1)), m.group(2))

for label in sorted(os.listdir(root)):
    p = os.path.join(root, label)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "run.json")):
        r = json.load(open(os.path.join(p, "run.json")))
        steady = None
        cs = r.get("chunk_s")
        if cs and len(cs) > 1:
            steady = sum(cs[1:]) / (8 * (len(cs) - 1))
        rows.append(dict(arm=label, kind="torch" if "device" in r else "mlx", target=r["target"],
                         gpu=gates.get(label, ("?", "?"))[0], thermal=gates.get(label, ("?", "?"))[1],
                         frames=r["frames_out"], run_s=round(r["run_s"], 2), s_per_frame=round(r["s_per_frame"], 3),
                         steady_s_per_frame=round(steady, 3) if steady else "",
                         peak_phys_gb=round(r["peak_phys_gb"], 1), mlx_peak_gb=round(r.get("mlx_peak_gb", 0), 1) or ""))
    elif label.endswith(".log") and os.path.splitext(label)[0].startswith("e"):
        t = open(p).read()
        m = re.search(r"engine: ×\d+ \w+ — ([\d.]+) s for ~(\d+) frames \((\d+) ms/frame.*lifetime peak ([\d.]+) GB.*MLX peak ([\d.]+) GB", t)
        if m:
            a = os.path.splitext(label)[0]
            rows.append(dict(arm=a, kind="engine e2e", target="", gpu=gates.get(a, ("?", "?"))[0],
                             thermal=gates.get(a, ("?", "?"))[1], frames=int(m.group(2)), run_s=float(m.group(1)),
                             s_per_frame=round(int(m.group(3)) / 1000, 3), steady_s_per_frame="",
                             peak_phys_gb=float(m.group(4)), mlx_peak_gb=float(m.group(5))))

keys = ["arm", "kind", "target", "gpu", "thermal", "frames", "run_s", "s_per_frame", "steady_s_per_frame",
        "peak_phys_gb", "mlx_peak_gb"]
print(" | ".join(keys))
for r in rows:
    print(" | ".join(str(r[k]) for k in keys))

print("\nmedians per configuration (whole-run s/frame · steady s/frame):")
groups = {}
for r in rows:
    groups.setdefault(re.sub(r"[ab]$", "", r["arm"]), []).append(r)
for g, rs in sorted(groups.items()):
    w = statistics.median(r["s_per_frame"] for r in rs)
    st = [r["steady_s_per_frame"] for r in rs if r["steady_s_per_frame"] != ""]
    print(f"  {g:8s} n={len(rs)}  {w:.3f}" + (f" · {statistics.median(st):.3f}" if st else "")
          + f"   (spread {min(r['s_per_frame'] for r in rs):.3f}–{max(r['s_per_frame'] for r in rs):.3f})")

if "--csv" in sys.argv:
    with open(sys.argv[sys.argv.index("--csv") + 1], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
