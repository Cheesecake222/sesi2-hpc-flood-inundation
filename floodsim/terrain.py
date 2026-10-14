"""Data permukaan: DEM sintetis (lembah sungai) dan I/O ESRI ASCII Grid (.asc).

Format ASCII Grid dipilih karena bisa dibaca/ditulis tanpa library GIS dan
didukung QGIS, ArcGIS, dan GDAL. Baris pertama data = baris paling utara,
sama dengan konvensi baris 0 di model ini.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Terrain:
    z: np.ndarray                 # elevasi (m), shape (ny, nx)
    dx: float                     # ukuran sel (m)
    manning: np.ndarray           # koefisien Manning n (s/m^(1/3))
    infil_mm_h: np.ndarray        # kapasitas infiltrasi (mm/jam)
    active: np.ndarray            # True = sel dihitung (False = di luar area studi)
    channel: np.ndarray | None = None          # topeng alur sungai (opsional)
    inlet_cells: list = field(default_factory=list)  # sel masuknya sungai (opsional)
    xll: float = 0.0
    yll: float = 0.0

    @property
    def shape(self):
        return self.z.shape


# --------------------------------------------------------------------- ASCII grid
def read_asc(path):
    """Baca ESRI ASCII Grid. Mengembalikan (array float, meta dict). NODATA -> NaN."""
    meta, header_lines = {}, 0
    with open(Path(path), "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and parts[0].lower() in (
                    "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter",
                    "cellsize", "nodata_value"):
                meta[parts[0].lower()] = float(parts[1])
                header_lines += 1
            else:
                break
    for key in ("ncols", "nrows", "cellsize"):
        if key not in meta:
            raise ValueError(f"{path}: header '{key}' tidak ditemukan")
    data = np.loadtxt(path, skiprows=header_lines, dtype=float, ndmin=2)
    if data.shape != (int(meta["nrows"]), int(meta["ncols"])):
        raise ValueError(f"{path}: ukuran data {data.shape} tidak cocok dengan header")
    nodata = meta.get("nodata_value", -9999.0)
    data = np.where(data == nodata, np.nan, data)
    return data, meta


def write_asc(path, arr, cellsize, xll=0.0, yll=0.0, nodata=-9999.0, fmt="%.4f"):
    """Tulis ESRI ASCII Grid. NaN ditulis sebagai NODATA."""
    arr = np.where(np.isfinite(arr), arr, nodata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"ncols {arr.shape[1]}\nnrows {arr.shape[0]}\n"
              f"xllcorner {xll}\nyllcorner {yll}\ncellsize {cellsize}\nNODATA_value {nodata}")
    np.savetxt(path, arr, fmt=fmt, header=header, comments="")


def terrain_from_asc(dem_path, manning=0.05, infil_mm_h=10.0, manning_raster=None,
                     infiltration_raster=None) -> Terrain:
    """Bangun Terrain dari DEM .asc nyata (opsional: raster Manning dan infiltrasi)."""
    z, meta = read_asc(dem_path)
    active = np.isfinite(z)
    if not active.any():
        raise ValueError("DEM tidak punya sel valid")
    z = np.where(active, z, np.nanmin(z))

    def layer(raster, default, name):
        if raster is None:
            return np.full(z.shape, float(default))
        arr, m2 = read_asc(raster)
        if arr.shape != z.shape:
            raise ValueError(f"raster {name} ({arr.shape}) harus berukuran sama dengan DEM {z.shape}")
        return np.where(np.isfinite(arr), arr, float(default))

    return Terrain(
        z=z, dx=float(meta["cellsize"]),
        manning=layer(manning_raster, manning, "Manning"),
        infil_mm_h=layer(infiltration_raster, infil_mm_h, "infiltrasi"),
        active=active,
        xll=meta.get("xllcorner", 0.0), yll=meta.get("yllcorner", 0.0),
    )


# ------------------------------------------------------------------ DEM sintetis
def _smooth(a: np.ndarray, iters: int) -> np.ndarray:
    """Haluskan dengan rata-rata 3x3 berulang (tanpa scipy)."""
    for _ in range(iters):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
    return a


def make_synthetic_terrain(ny=160, nx=240, dx=10.0, seed=7, channel_width=24.0,
                           channel_depth=1.5) -> Terrain:
    """Lembah sungai sintetis untuk demo dan pengujian.

    Isi: sungai berkelok mengalir ke selatan, dataran banjir landai, tanggul di
    sebagian tepi timur, cekungan (kolam alami) di dataran banjir, kawasan
    perkotaan (kedap air, kekasaran tinggi), dan derau mikro-topografi.
    Baris 0 = utara (hulu), baris terakhir = selatan (hilir/outlet).
    """
    rng = np.random.default_rng(seed)
    y = (np.arange(ny)[:, None] + 0.5) * dx
    x = (np.arange(nx)[None, :] + 0.5) * dx
    length, width = ny * dx, nx * dx

    xc = (width * 0.42
          + 0.10 * width * np.sin(2 * np.pi * y / (0.9 * length))
          + 0.03 * width * np.sin(2 * np.pi * y / (0.31 * length) + 1.0))     # (ny, 1)
    dist = np.abs(x - xc)

    # kemiringan memanjang + penampang lembah (dataran banjir landai, lereng di tepi)
    z = 8.0 + 0.0035 * (length - y)
    z = z + 0.0012 * np.minimum(dist, 500.0) + 0.012 * np.maximum(dist - 500.0, 0.0)

    # alur sungai (penampang parabola)
    half = channel_width / 2.0
    carve = channel_depth * np.clip(1.0 - (dist / half) ** 2, 0.0, None)
    z = z - carve
    channel = dist < half

    # tanggul di tepi timur pada segmen tengah (ada celah di hulu dan hilir)
    window = ((y > 0.30 * length) & (y < 0.65 * length)).astype(float)
    levee = 0.7 * np.exp(-(((dist - (half + 10.0)) / 12.0) ** 2)) * (x > xc) * window
    z = z + levee

    # cekungan alami di dataran banjir sisi timur
    cy = 0.70 * length
    cx = float(xc[min(ny - 1, int(0.70 * ny)), 0]) + 350.0
    z = z - 1.3 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / 140.0 ** 2))

    # mikro-topografi
    noise = _smooth(rng.normal(size=(ny, nx)), 4)
    z = z + 0.12 * noise / noise.std()

    # tutupan lahan: sungai, perkotaan, lahan terbuka
    manning = np.full((ny, nx), 0.06)
    infil = np.full((ny, nx), 12.0)
    urban = (x > 0.12 * width) & (x < 0.30 * width) & (y > 0.45 * length) & (y < 0.70 * length)
    manning[urban], infil[urban] = 0.12, 2.0
    manning[channel], infil[channel] = 0.035, 0.0

    inlet = [(1, int(c)) for c in np.where(channel[1])[0]]
    return Terrain(z=z, dx=float(dx), manning=manning, infil_mm_h=infil,
                   active=np.ones((ny, nx), dtype=bool), channel=channel, inlet_cells=inlet)


def hillshade(z: np.ndarray, dx: float, azimuth=315.0, altitude=45.0, exaggeration=3.0) -> np.ndarray:
    """Bayangan relief sederhana (0-1) untuk latar peta."""
    gy, gx = np.gradient(z * exaggeration, dx)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)
