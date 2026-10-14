#!/usr/bin/env python3
"""Bandingkan dua folder keluaran (mis. serial vs MPI). Hasil seharusnya identik.

Pemakaian:
    python scripts/compare_runs.py outputs/serial outputs/mpi4 [--tol 1e-9]
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--tol", type=float, default=1e-9, help="toleransi selisih maksimum (m)")
    args = ap.parse_args()

    a = np.load(f"{args.run_a}/arrays.npz")
    b = np.load(f"{args.run_b}/arrays.npz")
    ok = True
    for key in ("max_depth", "final_depth", "arrival_s"):
        diff = float(np.abs(a[key] - b[key]).max())
        flag = "OK " if diff <= args.tol else "BEDA"
        ok &= diff <= args.tol
        print(f"[{flag}] {key:12s} selisih maksimum = {diff:.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
