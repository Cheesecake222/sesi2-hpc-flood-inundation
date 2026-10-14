"""Antarmuka baris perintah.

Contoh:
    python -m floodsim run --config configs/demo.toml
    mpirun -n 4 python -m floodsim run --config configs/demo.toml --backend mpi
    python -m floodsim plot --run outputs/demo
    python -m floodsim make-dem --out data/dem_sintetis.asc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import apply_overrides, build_case, load_config
from .outputs import save_outputs
from .runner import DistributedRunner, LocalRunner


def _print_summary(summary: dict):
    s = summary
    print("\n=== Ringkasan simulasi ===")
    print(f"grid               : {s['grid'][0]} x {s['grid'][1]} sel, dx = {s['dx_m']} m "
          f"({s['grid'][0] * s['grid'][1]:,} sel)")
    print(f"bagian domain      : {s['nparts']}")
    print(f"langkah waktu      : {s['steps']:,}")
    print(f"waktu komputasi    : {s['wall_time_s']:.2f} s "
          f"({s['cell_updates_per_s'] / 1e6:.2f} juta pembaruan-sel/detik)")
    print(f"hujan total        : {s['rain_total_m3']:,.0f} m3 (rancangan {s['rain_design_total_mm']:.1f} mm)")
    print(f"inflow sungai      : {s['inflow_total_m3']:,.0f} m3")
    print(f"infiltrasi         : {s['infiltrated_m3']:,.0f} m3")
    print(f"keluar via batas   : {s['outflow_m3']:,.0f} m3")
    print(f"tersimpan akhir    : {s['stored_final_m3']:,.0f} m3")
    print(f"galat neraca massa : {s['mass_balance_error_rel']:.2e} (relatif)")
    print(f"kedalaman maks     : {s['max_depth_m']:.2f} m")
    for k, v in s["flooded_area_ha"].items():
        print(f"luas genangan {k:>8}: {v:,.1f} ha")


def cmd_run(args) -> int:
    cfg = apply_overrides(load_config(args.config), args.set)
    case = build_case(cfg)
    outdir = Path(args.outdir) if args.outdir else Path("outputs") / case.name

    if args.backend == "serial":
        runner, root = LocalRunner(case.terrain, case.params, case.rain, case.inflows,
                                   case.settings, nparts=1), True
        result = runner.run()
    elif args.backend == "slabs":
        runner = LocalRunner(case.terrain, case.params, case.rain, case.inflows, case.settings,
                             nparts=args.parts)
        result, root = runner.run(), True
    elif args.backend == "threads":
        from .comm import run_in_threads

        def work(comm):
            return DistributedRunner(case.terrain, case.params, case.rain, case.inflows,
                                     case.settings, comm).run()
        result, root = run_in_threads(work, args.parts)[0], True
    elif args.backend == "mpi":
        from .comm import MPIComm
        comm = MPIComm()
        result = DistributedRunner(case.terrain, case.params, case.rain, case.inflows,
                                   case.settings, comm).run()
        root = comm.rank == 0
    else:  # pragma: no cover - dijaga argparse
        raise SystemExit(f"backend tidak dikenal: {args.backend}")

    if root:
        save_outputs(result, case, outdir)
        with open(outdir / "summary.json", encoding="utf-8") as fh:
            _print_summary(json.load(fh))
        print(f"\nKeluaran disimpan di: {outdir}")
        if args.plot:
            from .viz import make_all_figures
            for f in make_all_figures(outdir):
                print("  gambar:", f)
    return 0


def cmd_plot(args) -> int:
    from .viz import make_all_figures
    for f in make_all_figures(args.run, gif=not args.no_gif):
        print("dibuat:", f)
    return 0


def cmd_make_dem(args) -> int:
    from .terrain import make_synthetic_terrain, write_asc
    t = make_synthetic_terrain(args.ny, args.nx, args.dx, args.seed)
    write_asc(args.out, t.z, t.dx)
    print(f"DEM sintetis ditulis ke {args.out} ({args.ny} x {args.nx}, dx = {args.dx} m)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="floodsim", description="Simulasi genangan banjir 2D skala kecil.")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="jalankan simulasi")
    r.add_argument("--config", required=True, help="berkas konfigurasi TOML")
    r.add_argument("--outdir", help="folder keluaran (default: outputs/<nama>)")
    r.add_argument("--backend", choices=["serial", "slabs", "threads", "mpi"], default="serial",
                   help="serial | slabs (N slab, satu proses) | threads (N thread) | mpi (mpirun)")
    r.add_argument("--parts", type=int, default=2, help="jumlah slab/thread untuk backend slabs/threads")
    r.add_argument("--set", action="append", default=[], metavar="KUNCI=NILAI",
                   help="timpa konfigurasi, mis. --set terrain.ny=300 (boleh diulang)")
    r.add_argument("--plot", action="store_true", help="buat gambar setelah simulasi selesai")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("plot", help="buat gambar dari folder keluaran")
    p.add_argument("--run", required=True, help="folder keluaran hasil 'run'")
    p.add_argument("--no-gif", action="store_true", help="lewati pembuatan animasi GIF")
    p.set_defaults(func=cmd_plot)

    d = sub.add_parser("make-dem", help="tulis DEM sintetis ke berkas .asc")
    d.add_argument("--out", required=True)
    d.add_argument("--ny", type=int, default=160)
    d.add_argument("--nx", type=int, default=240)
    d.add_argument("--dx", type=float, default=10.0)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_make_dem)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
