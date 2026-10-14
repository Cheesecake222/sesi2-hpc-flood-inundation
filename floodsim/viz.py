"""Visualisasi hasil simulasi (Matplotlib, backend tanpa layar)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from .outputs import HAZARD_EDGES, HAZARD_LABELS, area_above, load_outputs  # noqa: E402
from .terrain import hillshade  # noqa: E402

DEPTH_CMAP = "YlGnBu"


def fmt_id(x, nd=1):
    """Format angka gaya Indonesia: 1.234,5."""
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _extent(z, dx):
    ny, nx = z.shape
    return [0, nx * dx, ny * dx, 0]


def _base_axes(z, dx, title, figsize=(8.5, 6.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.imshow(hillshade(z, dx), cmap="gray", extent=_extent(z, dx), vmin=0, vmax=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)   [utara di atas]")
    return fig, ax


def plot_terrain(d, path):
    z, dx = d["z"], d["dx"]
    fig, ax = _base_axes(z, dx, "Topografi (DEM) dan alur sungai")
    im = ax.imshow(z, cmap="terrain", extent=_extent(z, dx), alpha=0.55)
    if d["channel"].any():
        ax.contour(d["channel"].astype(float), levels=[0.5], colors="navy", linewidths=0.8,
                   extent=_extent(z, dx), origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Elevasi (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_max_depth(d, path):
    z, dx, md = d["z"], d["dx"], d["max_depth"]
    area = area_above(md, 0.1, dx)
    fig, ax = _base_axes(z, dx, f"Kedalaman genangan maksimum  |  area >= 0,1 m: {fmt_id(area)} ha")
    show = np.ma.masked_less(md, 0.05)
    vmax = max(2.0, float(np.nanpercentile(md[md > 0.05], 99))) if (md > 0.05).any() else 2.0
    im = ax.imshow(show, cmap=DEPTH_CMAP, extent=_extent(z, dx), vmin=0, vmax=vmax, alpha=0.92)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Kedalaman maksimum (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hazard(d, path):
    z, dx, md = d["z"], d["dx"], d["max_depth"]
    colors = ["#a6d96a", "#fee08b", "#f46d43", "#a50026"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(HAZARD_EDGES + [1e9], cmap.N)
    fig, ax = _base_axes(z, dx, "Kelas bahaya genangan (indikatif, hanya dari kedalaman maksimum)")
    ax.imshow(np.ma.masked_less(md, HAZARD_EDGES[0]), cmap=cmap, norm=norm,
              extent=_extent(z, dx), alpha=0.9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = []
    edges = HAZARD_EDGES + [np.inf]
    for k, lab in enumerate(HAZARD_LABELS):
        a = area_above(md, edges[k], dx) - (area_above(md, edges[k + 1], dx) if np.isfinite(edges[k + 1]) else 0.0)
        labels.append(f"{lab}: {fmt_id(a)} ha")
    ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeseries(d, path):
    h = d["history"]
    t_h = h["t_min"] / 60.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.2), dpi=130, sharex=True)
    width = (t_h[1] - t_h[0]) if len(t_h) > 1 else 0.1
    a1.bar(t_h, h["rain_mm_h"], width=width, color="#4c72b0", alpha=0.8, label="Hujan (mm/jam)", align="edge")
    a1.set_ylabel("Hujan (mm/jam)")
    a1.invert_yaxis()
    a1b = a1.twinx()
    a1b.plot(t_h, h["inflow_m3s"], color="#c44e52", lw=2, label="Debit sungai masuk (m³/s)")
    a1b.set_ylabel("Debit masuk (m³/s)")
    a1.set_title("Masukan: hujan dan debit sungai")
    a2.plot(t_h, h["flooded_area_ha"], color="#2a6f97", lw=2, label="Luas genangan >= 0,1 m (ha)")
    a2.set_ylabel("Luas genangan (ha)")
    a2b = a2.twinx()
    a2b.plot(t_h, h["max_depth_m"], color="#e07a1f", lw=1.6, ls="--", label="Kedalaman maks (m)")
    a2b.set_ylabel("Kedalaman maksimum (m)")
    a2.set_xlabel("Waktu (jam)")
    a2.set_title("Respons genangan")
    l1, n1 = a2.get_legend_handles_labels()
    l2, n2 = a2b.get_legend_handles_labels()
    a2.legend(l1 + l2, n1 + n2, loc="center right", fontsize=8, framealpha=0.9)
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_gif(d, path, max_frames=40, fps=5):
    if "snap_depth" not in d:
        return False
    from matplotlib.animation import FuncAnimation, PillowWriter
    z, dx = d["z"], d["dx"]
    times, depth = d["snap_times_s"], d["snap_depth"]
    idx = np.unique(np.linspace(0, len(times) - 1, min(max_frames, len(times))).astype(int))
    vmax = max(1.5, float(np.percentile(d["max_depth"][d["max_depth"] > 0.05], 98))) \
        if (d["max_depth"] > 0.05).any() else 1.5
    fig, ax = _base_axes(z, dx, "", figsize=(7.2, 5.4))
    im = ax.imshow(np.ma.masked_less(depth[idx[0]], 0.05), cmap=DEPTH_CMAP, extent=_extent(z, dx),
                   vmin=0, vmax=vmax, alpha=0.92)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Kedalaman (m)")
    fig.tight_layout()

    def update(k):
        im.set_data(np.ma.masked_less(depth[idx[k]], 0.05))
        ax.set_title(f"Kedalaman air  |  t = {fmt_id(times[idx[k]] / 3600.0, 2)} jam", fontsize=11)
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=75)
    plt.close(fig)
    return True


def make_all_figures(outdir, gif=True) -> list[Path]:
    outdir = Path(outdir)
    d = load_outputs(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    made = []
    for name, fn in (("terrain.png", plot_terrain), ("max_depth.png", plot_max_depth),
                     ("hazard.png", plot_hazard), ("timeseries.png", plot_timeseries)):
        fn(d, figdir / name)
        made.append(figdir / name)
    if gif and make_gif(d, figdir / "flood.gif"):
        made.append(figdir / "flood.gif")
    return made
