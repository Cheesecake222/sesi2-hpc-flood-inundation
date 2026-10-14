"""Simpan dan muat keluaran simulasi."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .terrain import write_asc

HISTORY_FIELDS = ["t_min", "rain_mm_h", "inflow_m3s", "flooded_area_ha", "max_depth_m",
                  "stored_m3", "rain_m3", "inflow_m3", "infil_m3", "outflow_m3"]

# Kelas bahaya indikatif berdasarkan kedalaman maksimum saja (m).
HAZARD_EDGES = [0.1, 0.5, 1.0, 2.0]
HAZARD_LABELS = ["Rendah (0,1-0,5 m)", "Sedang (0,5-1,0 m)", "Tinggi (1,0-2,0 m)",
                 "Sangat tinggi (>2,0 m)"]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def area_above(depth: np.ndarray, threshold: float, dx: float) -> float:
    """Luas (hektar) sel dengan kedalaman >= threshold."""
    return float((depth >= threshold).sum()) * dx * dx / 1e4


def save_outputs(result, case, outdir) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dx, z = case.params.dx, case.terrain.z
    md = np.where(case.terrain.active, result.max_depth, np.nan)

    summary = dict(result.summary)
    summary["name"] = case.name
    summary["rain_design_total_mm"] = case.rain.total_mm()
    summary["flooded_area_ha"] = {f">={t} m": area_above(result.max_depth, t, dx)
                                  for t in (0.1, 0.5, 1.0, 2.0)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default),
                                         encoding="utf-8")

    with open(outdir / "history.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        for row in result.history:
            w.writerow({k: row[k] for k in HISTORY_FIELDS})

    arrival_min = np.where(result.arrival_s >= 0, result.arrival_s / 60.0, np.nan)
    write_asc(outdir / "max_depth.asc", md, dx, case.terrain.xll, case.terrain.yll)
    write_asc(outdir / "arrival_time_min.asc", arrival_min, dx, case.terrain.xll, case.terrain.yll,
              fmt="%.2f")
    np.savez_compressed(outdir / "arrays.npz", z=z, dx=dx, max_depth=result.max_depth,
                        arrival_s=result.arrival_s, final_depth=result.final_depth,
                        channel=case.terrain.channel if case.terrain.channel is not None
                        else np.zeros_like(z, dtype=bool))
    if result.snapshots is not None:
        np.savez_compressed(outdir / "snapshots.npz", times_s=result.snapshot_times_s,
                            depth=result.snapshots.astype(np.float16))
    return outdir


def load_outputs(outdir) -> dict:
    outdir = Path(outdir)
    data = dict(np.load(outdir / "arrays.npz"))
    data["dx"] = float(data["dx"])
    data["summary"] = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    with open(outdir / "history.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    data["history"] = {k: np.array([float(r[k]) for r in rows]) for k in HISTORY_FIELDS}
    snap = outdir / "snapshots.npz"
    if snap.exists():
        s = np.load(snap)
        data["snap_times_s"], data["snap_depth"] = s["times_s"], s["depth"].astype(np.float32)
    return data
