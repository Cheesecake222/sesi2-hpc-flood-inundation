"""Muat konfigurasi TOML dan bangun objek simulasi (Case)."""
from __future__ import annotations

import ast
import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rainfall import Hyetograph, design_storm, hyetograph_from_csv
from .runner import RunSettings
from .solver import Inflow, SolverParams
from .terrain import Terrain, make_synthetic_terrain, terrain_from_asc

VALID_SIDES = {"north", "south", "east", "west"}

DEFAULTS = {
    "run": {"name": "run", "duration_h": 6.0, "history_interval_min": 10.0,
            "snapshot_interval_min": 0.0},
    "terrain": {"source": "synthetic", "ny": 160, "nx": 240, "dx": 10.0, "seed": 7,
                "channel_width": 24.0, "channel_depth": 1.5,
                "manning_default": 0.05, "infiltration_mm_h": 10.0},
    "model": {"open_boundaries": ["south"], "cfl": 0.7, "dt_max": 10.0, "dt_min": 0.001,
              "h_tol": 0.001, "fr_max": 1.0, "boundary_slope": 0.004, "wet_threshold_m": 0.1},
    "rainfall": {"type": "design", "total_mm": 100.0, "duration_min": 180.0,
                 "peak_position": 0.4, "interval_min": 10.0},
}


@dataclass
class Case:
    name: str
    terrain: Terrain
    params: SolverParams
    rain: Hyetograph
    inflows: list
    settings: RunSettings
    config: dict


def load_config(path) -> dict:
    path = Path(path)
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_base_dir"] = str(path.parent.resolve())
    return cfg


def apply_overrides(cfg: dict, overrides) -> dict:
    """Terapkan override gaya ``terrain.ny=300`` (nilai di-parse sebagai literal Python)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override harus berbentuk kunci=nilai, bukan '{item}'")
        key, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def _section(cfg, name):
    merged = dict(DEFAULTS.get(name, {}))
    merged.update(cfg.get(name, {}))
    return merged


def build_case(cfg: dict) -> Case:
    base = Path(cfg.get("_base_dir", "."))
    run, tcfg, mcfg, rcfg = (_section(cfg, k) for k in ("run", "terrain", "model", "rainfall"))

    # ---- terrain
    if tcfg["source"] == "synthetic":
        terrain = make_synthetic_terrain(int(tcfg["ny"]), int(tcfg["nx"]), float(tcfg["dx"]),
                                         int(tcfg["seed"]), float(tcfg["channel_width"]),
                                         float(tcfg["channel_depth"]))
    elif tcfg["source"] == "asc":
        def rel(key):
            return None if tcfg.get(key) in (None, "") else str((base / tcfg[key]).resolve())
        if rel("dem") is None:
            raise ValueError("terrain.source='asc' membutuhkan terrain.dem")
        terrain = terrain_from_asc(rel("dem"), tcfg["manning_default"], tcfg["infiltration_mm_h"],
                                   rel("manning_raster"), rel("infiltration_raster"))
    else:
        raise ValueError("terrain.source harus 'synthetic' atau 'asc'")

    # ---- parameter model
    sides = frozenset(mcfg["open_boundaries"])
    if not sides <= VALID_SIDES:
        raise ValueError(f"open_boundaries harus subset dari {sorted(VALID_SIDES)}")
    params = SolverParams(dx=terrain.dx, h_tol=mcfg["h_tol"], cfl=mcfg["cfl"], dt_max=mcfg["dt_max"],
                          dt_min=mcfg["dt_min"], fr_max=mcfg["fr_max"],
                          boundary_slope=mcfg["boundary_slope"], open_sides=sides,
                          wet_threshold=mcfg["wet_threshold_m"])

    # ---- hujan
    if rcfg["type"] == "design":
        rain = design_storm(rcfg["total_mm"], rcfg["duration_min"], rcfg["peak_position"],
                            rcfg["interval_min"])
    elif rcfg["type"] == "csv":
        rain = hyetograph_from_csv((base / rcfg["file"]).resolve())
    elif rcfg["type"] == "none":
        rain = Hyetograph(np.array([0.0, 60.0]), np.array([0.0, 0.0]))
    else:
        raise ValueError("rainfall.type harus 'design', 'csv', atau 'none'")

    # ---- inflow sungai
    inflows = []
    for item in cfg.get("inflow", []):
        if "cells" in item:
            cells = [tuple(map(int, c)) for c in item["cells"]]
        elif item.get("location", "river_inlet") == "river_inlet":
            cells = list(terrain.inlet_cells)
            if not cells:
                raise ValueError("location='river_inlet' hanya tersedia pada terrain sintetis; "
                                 "gunakan cells = [[baris, kolom], ...]")
        else:
            raise ValueError("blok [[inflow]] butuh 'cells' atau location = 'river_inlet'")
        ny, nx = terrain.z.shape
        for r, c in cells:
            if not (0 <= r < ny and 0 <= c < nx):
                raise ValueError(f"sel inflow ({r}, {c}) di luar grid {ny}x{nx}")
        inflows.append(Inflow(cells, np.array(item["times_min"], float) * 60.0,
                              np.array(item["q_m3s"], float)))

    snap = float(run["snapshot_interval_min"]) * 60.0
    settings = RunSettings(duration_s=float(run["duration_h"]) * 3600.0,
                           history_dt_s=float(run["history_interval_min"]) * 60.0,
                           snapshot_dt_s=snap if snap > 0 else None)
    return Case(run["name"], terrain, params, rain, inflows, settings, cfg)
