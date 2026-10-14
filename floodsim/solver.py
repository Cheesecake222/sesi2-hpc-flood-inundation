"""Inti numerik: model genangan 2D dengan formulasi inersia lokal.

Skema mengikuti pendekatan inersia lokal (Bates et al., 2010; de Almeida et al.,
2012) yang dipakai pada model genangan seperti LISFLOOD-FP:

    q^{t+dt} = (q^t - g * h_f * dt * d(eta)/dx) / (1 + g * dt * n^2 * |q^t| / h_f^(7/3))

dengan ``eta = z + h`` (muka air), ``h_f`` kedalaman aliran pada sisi antar-sel,
``n`` koefisien Manning, dan ``q`` debit per satuan lebar (m^2/s).

Domain dipecah per baris (slab). Setiap ``Subdomain`` menyimpan satu baris
"ghost" di atas dan di bawah. Dengan begitu kode yang sama dipakai untuk:

* simulasi serial (satu subdomain, ghost = dinding),
* dekomposisi domain dalam satu proses (untuk pengujian),
* dekomposisi domain terdistribusi (MPI atau thread).

Satu langkah waktu dibagi dua fase supaya pertukaran halo bisa disisipkan:

1. ``phase1``  : hitung fluks kandidat, sumber (hujan, inflow, infiltrasi), dan
                 faktor pembatas massa per sel.
2. (tukar faktor pembatas pada baris tepi)
3. ``phase2``  : terapkan fluks yang sudah dibatasi dan perbarui kedalaman.

Faktor pembatas massa menjamin kedalaman tidak negatif dan massa kekal, bahkan
pada batas antar-subdomain, karena kedua sisi memakai fluks yang identik.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81  # percepatan gravitasi (m/s^2)
WALL_Z = 1.0e6  # elevasi "dinding" untuk baris ghost di tepi domain global


@dataclass
class SolverParams:
    """Parameter numerik dan fisik yang berlaku untuk seluruh domain."""

    dx: float                      # ukuran sel (m)
    h_tol: float = 1e-3            # kedalaman minimum agar sisi dianggap basah (m)
    cfl: float = 0.7               # bilangan CFL
    dt_max: float = 10.0           # langkah waktu maksimum (s)
    dt_min: float = 1e-3           # langkah waktu minimum (s)
    fr_max: float = 1.0            # batas bilangan Froude pada fluks
    boundary_slope: float = 0.004  # kemiringan untuk aliran keluar batas terbuka
    open_sides: frozenset = field(default_factory=frozenset)  # {"north","south","east","west"}
    wet_threshold: float = 0.1     # ambang "tergenang" untuk statistik (m)

    def dt_for(self, hmax: float) -> float:
        """Langkah waktu CFL dari kedalaman maksimum global."""
        dt = self.cfl * self.dx / np.sqrt(G * max(hmax, 1e-6))
        return float(min(max(dt, self.dt_min), self.dt_max))


@dataclass
class Inflow:
    """Sumber debit titik (mis. sungai masuk ke domain), dibagi rata ke beberapa sel."""

    cells: list                    # daftar (baris, kolom) global
    times_s: np.ndarray
    q_m3s: np.ndarray

    def q_at(self, t: float) -> float:
        """Debit total (m^3/s) pada waktu t, interpolasi linear, nilai tepi ditahan."""
        return float(np.interp(t, self.times_s, self.q_m3s))


def partition_rows(ny: int, nparts: int) -> list[tuple[int, int]]:
    """Bagi ``ny`` baris menjadi ``nparts`` slab yang hampir sama besar."""
    if nparts < 1:
        raise ValueError("nparts harus >= 1")
    if nparts > ny:
        raise ValueError(f"jumlah slab ({nparts}) tidak boleh melebihi jumlah baris ({ny})")
    base, extra = divmod(ny, nparts)
    bounds, start = [], 0
    for k in range(nparts):
        size = base + (1 if k < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


class Subdomain:
    """Satu slab baris dari domain global beserta baris ghost.

    Indeks array berukuran ``(m + 2, nx)`` memakai baris 0 dan ``m + 1`` sebagai
    ghost; sel milik subdomain berada pada ``[1:-1]``.
    """

    def __init__(self, row0, row1, z, manning, infil_mm_h, active, params, inflows=()):
        ny_g, nx = z.shape
        self.row0, self.row1 = row0, row1
        self.m = row1 - row0
        self.nx = nx
        self.p = params
        self.dx = params.dx
        self.is_top = row0 == 0
        self.is_bottom = row1 == ny_g

        def pad_const(a, value):
            """Tambah satu baris ghost berisi nilai konstan di atas dan bawah."""
            wall = np.full((1, nx), value)
            return np.vstack([wall, a, wall])

        def pad_edge(a):
            """Tambah satu baris ghost berupa salinan baris tepi."""
            return np.vstack([a[:1], a, a[-1:]])

        self.z = pad_const(z, WALL_Z)[row0:row1 + 2].copy()
        self.n = pad_edge(manning)[row0:row1 + 2].copy()
        self.f = (infil_mm_h[row0:row1] / 1000.0 / 3600.0).copy()   # m/s
        self.active = active[row0:row1].astype(float)

        self.h = np.zeros((self.m + 2, nx))
        self.qx = np.zeros((self.m, nx - 1))
        self.qy = np.zeros((self.m + 1, nx))
        self.max_h = np.zeros((self.m, nx))
        self.arrival = np.full((self.m, nx), -1.0)

        # sumber debit yang jatuh di subdomain ini (dibagi rata ke semua sel inflow)
        self._inflows = []
        for inf in inflows:
            share = 1.0 / len(inf.cells)
            local = [(r - row0, c) for (r, c) in inf.cells if row0 <= r < row1]
            if local:
                self._inflows.append((inf, local, share))

        # volume kumulatif (m^3) untuk neraca massa
        self.vol_rain = 0.0
        self.vol_inflow = 0.0
        self.vol_infil = 0.0
        self.vol_out = 0.0

        self.scale_ghost_top = None
        self.scale_ghost_bottom = None
        self._pending = None

    # ------------------------------------------------------------------ halo
    def hmax(self) -> float:
        return float(self.h[1:-1].max()) if self.m else 0.0

    def edge_h(self):
        """Baris interior paling atas dan paling bawah (untuk dikirim ke tetangga)."""
        return self.h[1].copy(), self.h[-2].copy()

    def set_ghost_h(self, top=None, bottom=None):
        if top is not None:
            self.h[0] = top
        if bottom is not None:
            self.h[-1] = bottom

    def edge_scale(self):
        sc = self._pending["scale"]
        return sc[0].copy(), sc[-1].copy()

    def set_ghost_scale(self, top=None, bottom=None):
        self.scale_ghost_top = top
        self.scale_ghost_bottom = bottom

    # ----------------------------------------------------------------- fisika
    def _momentum(self, q, hf, slope, n, dt):
        p = self.p
        wet = hf > p.h_tol
        hs = np.where(wet, hf, 1.0)
        qn = (q - G * hs * dt * slope) / (1.0 + G * dt * n * n * np.abs(q) / hs ** (7.0 / 3.0))
        qmax = p.fr_max * hs * np.sqrt(G * hs)
        qn = np.clip(qn, -qmax, qmax)
        return np.where(wet, qn, 0.0)

    def _normal_flow(self, h, n):
        """Debit keluar batas terbuka dari persamaan Manning (aliran seragam)."""
        p = self.p
        wet = h > p.h_tol
        hs = np.where(wet, h, 1.0)
        q = hs ** (5.0 / 3.0) * np.sqrt(p.boundary_slope) / n
        q = np.minimum(q, hs * np.sqrt(G * hs))
        return np.where(wet, q, 0.0)

    def phase1(self, dt, t, rain_ms):
        """Hitung fluks kandidat, sumber, dan faktor pembatas massa."""
        p, dx, m, nx = self.p, self.dx, self.m, self.nx
        z, h, n = self.z, self.h, self.n
        eta = z + h
        ez, zz = eta[1:-1], z[1:-1]

        # fluks arah-x (antar kolom), hanya untuk baris interior
        hfx = np.maximum(ez[:, :-1], ez[:, 1:]) - np.maximum(zz[:, :-1], zz[:, 1:])
        sx = (ez[:, 1:] - ez[:, :-1]) / dx
        nfx = 0.5 * (n[1:-1, :-1] + n[1:-1, 1:])
        qx = self._momentum(self.qx, hfx, sx, nfx, dt)

        # fluks arah-y (antar baris), termasuk dua sisi yang menyentuh baris ghost
        hfy = np.maximum(eta[:-1], eta[1:]) - np.maximum(z[:-1], z[1:])
        sy = (eta[1:] - eta[:-1]) / dx
        nfy = 0.5 * (n[:-1] + n[1:])
        qy = self._momentum(self.qy, hfy, sy, nfy, dt)

        # aliran keluar pada batas terbuka domain global
        sides = p.open_sides
        bw = be = np.zeros(m)
        bt = bb = np.zeros(nx)
        if "west" in sides:
            bw = self._normal_flow(h[1:-1, 0], n[1:-1, 0])
        if "east" in sides:
            be = self._normal_flow(h[1:-1, -1], n[1:-1, -1])
        if "north" in sides and self.is_top:
            bt = self._normal_flow(h[1], n[1])
        if "south" in sides and self.is_bottom:
            bb = self._normal_flow(h[-2], n[-2])

        # sumber: hujan, inflow, lalu infiltrasi
        hs = h[1:-1] + rain_ms * dt * self.active
        self.vol_rain += rain_ms * dt * self.active.sum() * dx * dx
        for inf, local, share in self._inflows:
            q = inf.q_at(t)
            add = q * share * dt / (dx * dx)
            for (r, c) in local:
                hs[r, c] += add
            self.vol_inflow += q * share * dt * len(local)
        loss = np.minimum(hs, self.f * dt)
        hs = hs - loss
        self.vol_infil += float(loss.sum()) * dx * dx

        # total aliran keluar tiap sel, lalu faktor pembatas (agar h tidak negatif)
        out = np.zeros((m, nx))
        out[:, :-1] += np.maximum(qx, 0.0)
        out[:, 1:] += np.maximum(-qx, 0.0)
        out += np.maximum(-qy[:-1], 0.0) + np.maximum(qy[1:], 0.0)
        out[:, 0] += bw
        out[:, -1] += be
        if self.is_top:
            out[0] += bt
        if self.is_bottom:
            out[-1] += bb
        depth_out = out * dt / dx
        scale = np.ones((m, nx))
        over = depth_out > hs
        scale[over] = hs[over] / depth_out[over]

        self._pending = dict(qx=qx, qy=qy, bw=bw, be=be, bt=bt, bb=bb, hs=hs, scale=scale)

    def phase2(self, dt, t_new):
        """Terapkan fluks yang sudah dibatasi dan perbarui kedalaman air."""
        pend, dx, m, nx = self._pending, self.dx, self.m, self.nx
        scale = pend["scale"]
        sp = np.ones((m + 2, nx))
        sp[1:-1] = scale
        if self.scale_ghost_top is not None:
            sp[0] = self.scale_ghost_top
        if self.scale_ghost_bottom is not None:
            sp[-1] = self.scale_ghost_bottom

        qx, qy = pend["qx"], pend["qy"]
        qx = np.where(qx > 0, qx * scale[:, :-1], qx * scale[:, 1:])
        qy = np.where(qy > 0, qy * sp[:-1], qy * sp[1:])

        net = np.zeros((m, nx))
        net[:, 1:] += qx
        net[:, :-1] -= qx
        net += qy[:-1] - qy[1:]

        bw = pend["bw"] * scale[:, 0]
        be = pend["be"] * scale[:, -1]
        net[:, 0] -= bw
        net[:, -1] -= be
        bout = bw.sum() + be.sum()
        if self.is_top:
            bt = pend["bt"] * scale[0]
            net[0] -= bt
            bout += bt.sum()
        if self.is_bottom:
            bb = pend["bb"] * scale[-1]
            net[-1] -= bb
            bout += bb.sum()
        self.vol_out += float(bout) * dt * dx

        h_new = np.maximum(pend["hs"] + net * dt / dx, 0.0)
        self.h[1:-1] = h_new
        self.qx, self.qy = qx, qy
        np.maximum(self.max_h, h_new, out=self.max_h)
        newly = (self.arrival < 0) & (h_new > self.p.wet_threshold)
        self.arrival[newly] = t_new
        self._pending = None

    # --------------------------------------------------------------- statistik
    def stored_volume(self) -> float:
        return float(self.h[1:-1].sum()) * self.dx * self.dx

    def wet_cells(self) -> int:
        return int((self.h[1:-1] > self.p.wet_threshold).sum())

    def stats_vector(self) -> np.ndarray:
        """Vektor yang dijumlahkan lintas subdomain: [sel basah, tersimpan, hujan, inflow, infiltrasi, keluar]."""
        return np.array([self.wet_cells(), self.stored_volume(), self.vol_rain,
                         self.vol_inflow, self.vol_infil, self.vol_out], dtype=float)
