#!/usr/bin/env python
"""x4_quality.py — score port outputs on the FlashVSR spike's ×4 cells with the spike's OWN metric code, so MLX rows
land in the same table as the torch-MPS rows (`mlxengine-forge/Tools/flashvsr-spike/x4_results.csv`).

  python bench/x4_quality.py <cell> <armName>=<outDir> [...] [--csv out.csv]
     cell: e.g. X4-Wp/lr_clean (frames under $CELLS_DIR/<cell>, reference under $CELLS_DIR/<X4-..>/hr)

Needs FORGE_TOOLS=<mlxengine-forge>/Tools (the anime-sr-eval harness: frame loaders, SSIMULACRA2 gate, gradient).
That harness and the clips are not part of this repository (no pixels are committed); `x4_quality.csv` holds the
scores it produced.
"""
import csv, os, subprocess, sys

sys.path.insert(0, os.path.join(os.environ["FORGE_TOOLS"], "anime-sr-eval"))
import numpy as np  # noqa: E402
import run as R  # noqa: E402


def score(cell, arm, d):
    c = cell.split("/")[0]
    ref = R._series(f"{R.CELLS_DIR}/{c}/hr")
    dref = np.diff(ref, axis=0)
    S = R._series(d)
    n = min(len(S), len(ref))
    al = [float(np.mean(np.abs(S[n // 2] - ref[n // 2 + k]))) for k in (-1, 0, 1)]
    terr = np.abs(np.diff(S[:n], axis=0) - dref[: n - 1])
    csvp = f"{d}/fr.csv"
    if not os.path.exists(csvp):
        subprocess.run([R.VOSRGATE, "pair", f"{R.CELLS_DIR}/{c}/hr", d, csvp], check=True, stdout=subprocess.DEVNULL)
    fr = list(csv.DictReader(open(csvp)))
    s2 = np.array([float(x["fr"]) for x in fr])
    return dict(cell=cell, arm=arm, frames=len(S), s2_mean=float(s2.mean()), s2_p10=float(np.percentile(s2, 10)),
                psnr=float(np.mean([float(x["psnr"]) for x in fr])), terr=float(terr.mean()),
                grad=float(np.mean([R._grad(S[i]) for i in range(0, n, 4)])),
                grad_ref=float(np.mean([R._grad(ref[i]) for i in range(0, n, 4)])),
                align_minus1=al[0], align_0=al[1], align_plus1=al[2])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    cell, arms = args[0], [a.split("=", 1) for a in args[1:] if "=" in a]
    rows = [score(cell, a, d) for a, d in arms]
    for r in rows:
        print(f"{r['cell']:16s} {r['arm']:18s} S2 {r['s2_mean']:7.2f} (p10 {r['s2_p10']:7.2f}) PSNR {r['psnr']:5.2f} "
              f"terr {r['terr']:.3f} grad {r['grad']:.2f}/ref {r['grad_ref']:.2f} "
              f"align {r['align_minus1']:.2f}/{r['align_0']:.2f}/{r['align_plus1']:.2f}")
    if out:
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
