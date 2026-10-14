"""Uji otomatis floodsim. Jalankan dengan:  pytest -q"""
import json
from pathlib import Path

import numpy as np
import pytest

from floodsim.cli import main as cli_main
from floodsim.comm import run_in_threads
from floodsim.config import apply_overrides, build_case, load_config
from floodsim.rainfall import Hyetograph, design_storm, hyetograph_from_csv
from floodsim.runner import DistributedRunner, LocalRunner, RunSettings
from floodsim.solver import Inflow, SolverParams, partition_rows
from floodsim.terrain import Terrain, make_synthetic_terrain, read_asc, write_asc

ROOT = Path(__file__).resolve().parents[1]
NO_RAIN = Hyetograph(np.array([0.0, 60.0]), np.array([0.0, 0.0]))


def small_case(ny=30, nx=40, infil=True):
    t = make_synthetic_terrain(ny, nx, 10.0, seed=3)
    if not infil:
        t.infil_mm_h[:] = 0.0
    rain = design_storm(80.0, 40.0, 0.4, 10.0)
    inflow = Inflow(t.inlet_cells, np.array([0.0, 900.0, 1800.0]), np.array([2.0, 15.0, 4.0]))
    return t, rain, inflow


def settings(duration=1800.0, snap=None):
    return RunSettings(duration_s=duration, history_dt_s=300.0, snapshot_dt_s=snap)


# --------------------------------------------------------------------- hujan
def test_design_storm_total_depth_is_exact():
    h = design_storm(120.0, 180.0, 0.4, 10.0)
    assert h.total_mm() == pytest.approx(120.0, rel=1e-12)
    assert h.intensity_mm_h_at(h.end_time_s + 1.0) == 0.0
    peak_t = h.times_s[np.argmax(h.intensity_mm_h)]
    assert 0.25 * 180 * 60 < peak_t < 0.55 * 180 * 60


def test_hyetograph_csv(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("time_min,intensity_mm_h\n0,0\n30,60\n60,0\n", encoding="utf-8")
    h = hyetograph_from_csv(f)
    assert h.total_mm() == pytest.approx(30.0)
    assert h.intensity_mm_h_at(45 * 60) == 60.0


# ------------------------------------------------------------ kekekalan massa
def test_mass_conservation_closed_domain_rain_only():
    t, rain, _ = small_case(infil=False)
    res = LocalRunner(t, SolverParams(dx=10.0), rain, [], settings(2400.0)).run()
    s = res.summary
    expected = 0.080 * t.z.size * 100.0          # 80 mm di atas seluruh sel
    assert s["rain_total_m3"] == pytest.approx(expected, rel=1e-9)
    assert s["stored_final_m3"] == pytest.approx(expected, rel=1e-9)
    assert abs(s["mass_balance_error_rel"]) < 1e-10


def test_mass_balance_with_all_terms():
    t, rain, inflow = small_case()
    p = SolverParams(dx=10.0, open_sides=frozenset({"south", "west"}))
    s = LocalRunner(t, p, rain, [inflow], settings(2400.0)).run().summary
    assert s["infiltrated_m3"] > 0 and s["outflow_m3"] > 0 and s["inflow_total_m3"] > 0
    assert abs(s["mass_balance_error_rel"]) < 1e-10


def test_depth_never_negative():
    t, rain, inflow = small_case()
    p = SolverParams(dx=10.0, open_sides=frozenset({"south"}))
    res = LocalRunner(t, p, rain, [inflow], settings(1800.0)).run()
    assert res.final_depth.min() >= 0.0 and res.max_depth.min() >= 0.0
    assert np.isfinite(res.max_depth).all()


# ------------------------------------------------------------- kondisi diam
def test_lake_at_rest_stays_at_rest():
    ny, nx, dx = 24, 30, 10.0
    y, x = np.mgrid[0:ny, 0:nx] * dx
    z = 5.0 + 0.002 * ((x - 150.0) ** 2 + (y - 120.0) ** 2) / 10.0
    level = 5.6
    t = Terrain(z=z, dx=dx, manning=np.full(z.shape, 0.03), infil_mm_h=np.zeros(z.shape),
                active=np.ones(z.shape, bool))
    runner = LocalRunner(t, SolverParams(dx=dx), NO_RAIN, [], settings(600.0))
    h0 = np.maximum(level - z, 0.0)
    assert h0.max() > 0.3 and (h0 == 0).any()     # ada danau sekaligus tepi kering
    runner.subs()[0].h[1:-1] = h0
    res = runner.run()
    assert np.abs(res.final_depth - h0).max() < 1e-9
    assert np.abs(runner.subs()[0].qx).max() < 1e-9 and np.abs(runner.subs()[0].qy).max() < 1e-9


def test_water_drains_through_open_boundary():
    ny, nx, dx = 20, 12, 10.0
    z = np.tile((ny - np.arange(ny))[:, None] * 0.5, (1, nx)).astype(float)   # miring ke selatan
    t = Terrain(z=z, dx=dx, manning=np.full(z.shape, 0.03), infil_mm_h=np.zeros(z.shape),
                active=np.ones(z.shape, bool))
    rain = Hyetograph(np.array([0.0, 600.0, 601.0]), np.array([40.0, 0.0, 0.0]))
    p = SolverParams(dx=dx, open_sides=frozenset({"south"}))
    s = LocalRunner(t, p, rain, [], RunSettings(2400.0, 600.0)).run().summary
    assert s["outflow_m3"] > 0.8 * s["rain_total_m3"]
    assert abs(s["mass_balance_error_rel"]) < 1e-10


# --------------------------------------------------------- dekomposisi domain
@pytest.mark.parametrize("nparts", [2, 3, 5])
def test_slab_decomposition_matches_serial(nparts):
    t, rain, inflow = small_case()
    p = SolverParams(dx=10.0, open_sides=frozenset({"south"}))
    ref = LocalRunner(t, p, rain, [inflow], settings(1800.0, 600.0), 1).run()
    got = LocalRunner(t, p, rain, [inflow], settings(1800.0, 600.0), nparts).run()
    assert np.abs(ref.max_depth - got.max_depth).max() < 1e-10
    assert np.abs(ref.final_depth - got.final_depth).max() < 1e-10
    assert np.abs(ref.snapshots - got.snapshots).max() < 1e-6
    assert got.summary["mass_balance_error_rel"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("nranks", [1, 3, 4])
def test_distributed_runner_matches_serial(nranks):
    """Jalur terdistribusi (dipakai MPI) diuji lewat ThreadComm."""
    t, rain, inflow = small_case()
    p = SolverParams(dx=10.0, open_sides=frozenset({"south", "east"}))
    ref = LocalRunner(t, p, rain, [inflow], settings(1500.0), 1).run()

    def work(comm):
        return DistributedRunner(t, p, rain, [inflow], settings(1500.0), comm).run()

    out = run_in_threads(work, nranks)
    assert all(r is None for r in out[1:])
    assert np.abs(ref.max_depth - out[0].max_depth).max() < 1e-10
    assert np.abs(ref.arrival_s - out[0].arrival_s).max() < 1e-6
    assert out[0].summary["nparts"] == nranks


def test_partition_rows_covers_all_rows():
    for ny, n in [(10, 3), (160, 7), (5, 5)]:
        b = partition_rows(ny, n)
        assert b[0][0] == 0 and b[-1][1] == ny
        assert all(b[k][1] == b[k + 1][0] for k in range(n - 1))
    with pytest.raises(ValueError):
        partition_rows(3, 4)


# ----------------------------------------------------------------------- I/O
def test_asc_roundtrip(tmp_path):
    a = np.random.default_rng(0).normal(size=(7, 9))
    a[2, 3] = np.nan
    write_asc(tmp_path / "x.asc", a, cellsize=12.5, xll=100.0, yll=200.0)
    b, meta = read_asc(tmp_path / "x.asc")
    assert meta["cellsize"] == 12.5 and b.shape == a.shape
    assert np.isnan(b[2, 3])
    assert np.nanmax(np.abs(a - b)) < 1e-4


def test_config_overrides_and_build(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    cfg = apply_overrides(cfg, ["terrain.ny=20", "terrain.nx=30", "rainfall.total_mm=50.5",
                                "model.open_boundaries=['south','east']"])
    case = build_case(cfg)
    assert case.terrain.z.shape == (20, 30)
    assert case.rain.total_mm() == pytest.approx(50.5)
    assert case.params.open_sides == frozenset({"south", "east"})


def test_cli_run_and_plot_end_to_end(tmp_path):
    out = tmp_path / "run"
    rc = cli_main(["run", "--config", str(ROOT / "configs" / "smoke.toml"), "--outdir", str(out),
                   "--backend", "slabs", "--parts", "3"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert abs(summary["mass_balance_error_rel"]) < 1e-10 and summary["nparts"] == 3
    for name in ("max_depth.asc", "history.csv", "arrays.npz", "snapshots.npz"):
        assert (out / name).exists()
    assert cli_main(["plot", "--run", str(out), "--no-gif"]) == 0
    assert (out / "figures" / "max_depth.png").exists()


def test_cli_run_from_asc_dem(tmp_path):
    """Jalur data nyata: DEM .asc + hujan CSV."""
    rc = cli_main(["run", "--config", str(ROOT / "configs" / "contoh_asc.toml"), "--outdir", str(tmp_path / "o")])
    assert rc == 0
    s = json.loads((tmp_path / "o" / "summary.json").read_text(encoding="utf-8"))
    assert s["rain_total_m3"] > 0 and abs(s["mass_balance_error_rel"]) < 1e-10
