#!/usr/bin/env python3
"""Ukur strong scaling: ukuran masalah tetap, jumlah proses bertambah.

Contoh (klaster/HPC dengan MPI):
    python scripts/benchmark_scaling.py --config configs/demo.toml --procs 1 2 4 8 \
        --launcher mpirun --set terrain.ny=480 --set terrain.nx=720 --set run.duration_h=1.0

Tanpa MPI (hanya untuk mencoba skrip; thread Python tidak mewakili skala HPC):
    python scripts/benchmark_scaling.py --config configs/smoke.toml --procs 1 2 --launcher threads

Keluaran: tabel di layar, CSV, dan grafik speedup/efisiensi di folder --out.
Metrik: waktu dinding, speedup S(P) = T(1)/T(P), efisiensi E(P) = S(P)/P,
dan throughput (juta pembaruan-sel per detik) sebagai analogi "FLOPS" model ini.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run_once(config, procs, launcher, overrides, workdir):
    cmd_core = [sys.executable, "-m", "floodsim", "run", "--config", config, "--outdir", workdir]
    for o in overrides:
        cmd_core += ["--set", o]
    if launcher == "mpirun":
        cmd = ["mpirun", "-n", str(procs)] + cmd_core + ["--backend", "mpi"]
    elif launcher == "threads":
        cmd = cmd_core + ["--backend", "threads", "--parts", str(procs)]
    elif launcher == "slabs":
        cmd = cmd_core + ["--backend", "slabs", "--parts", str(procs)]
    else:
        raise SystemExit(f"launcher tidak dikenal: {launcher}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads((Path(workdir) / "summary.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--launcher", choices=["mpirun", "threads", "slabs"], default="mpirun")
    ap.add_argument("--set", action="append", default=[], metavar="KUNCI=NILAI")
    ap.add_argument("--repeat", type=int, default=1, help="ulangi tiap konfigurasi; dipakai waktu terbaik")
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()

    overrides = ["run.snapshot_interval_min=0"] + args.set   # tanpa snapshot agar murni komputasi
    rows = []
    for p in args.procs:
        best = None
        for _ in range(args.repeat):
            with tempfile.TemporaryDirectory() as tmp:
                s = run_once(args.config, p, args.launcher, overrides, tmp)
            if best is None or s["wall_time_s"] < best["wall_time_s"]:
                best = s
        rows.append(dict(procs=p, wall_s=best["wall_time_s"], steps=best["steps"],
                         mcups=best["cell_updates_per_s"] / 1e6, grid=f"{best['grid'][0]}x{best['grid'][1]}"))
        print(f"P={p:<3d} waktu={best['wall_time_s']:8.2f} s  {rows[-1]['mcups']:7.2f} juta sel-update/s")

    t1 = next((r["wall_s"] for r in rows if r["procs"] == 1), rows[0]["wall_s"] * rows[0]["procs"])
    for r in rows:
        r["speedup"] = t1 / r["wall_s"]
        r["efficiency"] = r["speedup"] / r["procs"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scaling.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n P | waktu (s) | speedup | efisiensi | juta sel-update/s")
    for r in rows:
        print(f"{r['procs']:>2d} | {r['wall_s']:9.2f} | {r['speedup']:7.2f} | {r['efficiency']:9.2%} | {r['mcups']:.2f}")

    ps = [r["procs"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=130)
    a1.plot(ps, [r["speedup"] for r in rows], "o-", label="terukur")
    a1.plot(ps, ps, "k--", lw=1, label="ideal")
    a1.set_xlabel("Jumlah proses (P)"); a1.set_ylabel("Speedup"); a1.set_title("Strong scaling")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ps, [r["efficiency"] * 100 for r in rows], "s-", color="#c44e52")
    a2.axhline(100, color="k", ls="--", lw=1)
    a2.set_xlabel("Jumlah proses (P)"); a2.set_ylabel("Efisiensi paralel (%)"); a2.set_title("Efisiensi")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scaling.png")
    print(f"\nTersimpan: {out / 'scaling.csv'}, {out / 'scaling.png'}")


if __name__ == "__main__":
    main()
