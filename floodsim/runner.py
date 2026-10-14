"""Pengendali simulasi: loop waktu, pertukaran halo, dan pengumpulan hasil.

Tiga runner memakai loop waktu yang sama (``_BaseRunner.run``) dan hanya berbeda
dalam cara menukar baris halo dan mereduksi nilai global:

* ``LocalRunner``       : satu atau beberapa slab dalam satu proses.
                          Satu slab = simulasi serial biasa.
* ``DistributedRunner`` : satu slab per rank (MPI atau ``ThreadComm``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .comm import SerialComm
from .rainfall import Hyetograph
from .solver import Inflow, SolverParams, Subdomain, partition_rows
from .terrain import Terrain


@dataclass
class RunSettings:
    duration_s: float
    history_dt_s: float = 600.0
    snapshot_dt_s: float | None = None   # None = tanpa snapshot


@dataclass
class SimResult:
    max_depth: np.ndarray
    arrival_s: np.ndarray
    final_depth: np.ndarray
    history: list
    snapshot_times_s: np.ndarray
    snapshots: np.ndarray | None
    summary: dict = field(default_factory=dict)


def make_subdomains(terrain: Terrain, params: SolverParams, inflows, nparts: int):
    ny = terrain.z.shape[0]
    return [Subdomain(r0, r1, terrain.z, terrain.manning, terrain.infil_mm_h,
                      terrain.active, params, inflows)
            for (r0, r1) in partition_rows(ny, nparts)]


class _BaseRunner:
    """Loop waktu bersama. Subclass mengisi operasi pertukaran dan reduksi."""

    def __init__(self, params: SolverParams, rain: Hyetograph, inflows, settings: RunSettings,
                 grid_shape):
        self.p, self.rain, self.inflows, self.cfg = params, rain, list(inflows), settings
        self.ny, self.nx = grid_shape

    # ---- diisi oleh subclass
    def subs(self): raise NotImplementedError
    def exchange_h(self): raise NotImplementedError
    def exchange_scale(self): raise NotImplementedError
    def reduce_max(self, x): raise NotImplementedError
    def reduce_sum(self, vec): raise NotImplementedError
    def gather_rows(self, arrays): raise NotImplementedError
    def is_root(self): return True

    # ---- utilitas
    def _record(self, t):
        local = sum(s.stats_vector() for s in self.subs())
        tot = self.reduce_sum(local)
        hmax = self.reduce_max(max(s.hmax() for s in self.subs()))
        dx = self.p.dx
        return dict(
            t_min=t / 60.0,
            rain_mm_h=self.rain.intensity_mm_h_at(t),
            inflow_m3s=float(sum(i.q_at(t) for i in self.inflows)),
            flooded_area_ha=float(tot[0]) * dx * dx / 1e4,
            max_depth_m=float(hmax),
            stored_m3=float(tot[1]), rain_m3=float(tot[2]), inflow_m3=float(tot[3]),
            infil_m3=float(tot[4]), outflow_m3=float(tot[5]),
        )

    def run(self) -> SimResult | None:
        cfg, p = self.cfg, self.p
        t, step = 0.0, 0
        history = [self._record(0.0)]
        snap_t, snap_a = [], []
        next_hist = cfg.history_dt_s
        next_snap = cfg.snapshot_dt_s if cfg.snapshot_dt_s else np.inf
        tol = 1e-9
        wall0 = time.perf_counter()

        while t < cfg.duration_s - tol:
            hmax = self.reduce_max(max(s.hmax() for s in self.subs()))
            dt = p.dt_for(hmax)
            dt = min(dt, cfg.duration_s - t, next_hist - t, next_snap - t)
            rain_ms = self.rain.intensity_ms(t)

            self.exchange_h()
            for s in self.subs():
                s.phase1(dt, t, rain_ms)
            self.exchange_scale()
            for s in self.subs():
                s.phase2(dt, t + dt)
            t += dt
            step += 1

            if t >= next_hist - tol:
                history.append(self._record(t))
                next_hist += cfg.history_dt_s
            if t >= next_snap - tol:
                full = self.gather_rows([s.h[1:-1].copy() for s in self.subs()])
                if full is not None:
                    snap_t.append(t)
                    snap_a.append(full.astype(np.float32))
                next_snap += cfg.snapshot_dt_s

        wall = self.reduce_max(time.perf_counter() - wall0)
        max_depth = self.gather_rows([s.max_h for s in self.subs()])
        arrival = self.gather_rows([s.arrival for s in self.subs()])
        final = self.gather_rows([s.h[1:-1].copy() for s in self.subs()])
        if not self.is_root():
            return None

        last = history[-1]
        supplied = last["rain_m3"] + last["inflow_m3"]
        balance = last["stored_m3"] - (supplied - last["infil_m3"] - last["outflow_m3"])
        summary = dict(
            steps=step, wall_time_s=wall,
            cell_updates_per_s=step * self.ny * self.nx / wall if wall > 0 else None,
            grid=[self.ny, self.nx], dx_m=p.dx, nparts=self.nparts(),
            mass_balance_error_m3=balance,
            mass_balance_error_rel=balance / supplied if supplied > 0 else 0.0,
            rain_total_m3=last["rain_m3"], inflow_total_m3=last["inflow_m3"],
            infiltrated_m3=last["infil_m3"], outflow_m3=last["outflow_m3"],
            stored_final_m3=last["stored_m3"], max_depth_m=float(max_depth.max()),
        )
        return SimResult(
            max_depth=max_depth, arrival_s=arrival, final_depth=final, history=history,
            snapshot_times_s=np.array(snap_t),
            snapshots=np.stack(snap_a) if snap_a else None, summary=summary)

    def nparts(self): return len(self.subs())


class LocalRunner(_BaseRunner):
    """Satu proses, ``nparts`` slab (``nparts=1`` = serial)."""

    def __init__(self, terrain, params, rain, inflows, settings, nparts=1):
        super().__init__(params, rain, inflows, settings, terrain.z.shape)
        self._subs = make_subdomains(terrain, params, self.inflows, nparts)

    def subs(self): return self._subs

    def exchange_h(self):
        for a, b in zip(self._subs[:-1], self._subs[1:]):
            a.set_ghost_h(bottom=b.edge_h()[0])
            b.set_ghost_h(top=a.edge_h()[1])

    def exchange_scale(self):
        for a, b in zip(self._subs[:-1], self._subs[1:]):
            a.set_ghost_scale(top=a.scale_ghost_top, bottom=b.edge_scale()[0])
            b.set_ghost_scale(top=a.edge_scale()[1], bottom=b.scale_ghost_bottom)

    def reduce_max(self, x): return x
    def reduce_sum(self, vec): return np.asarray(vec, dtype=float)
    def gather_rows(self, arrays): return np.vstack(arrays)


class DistributedRunner(_BaseRunner):
    """Satu slab per rank; pertukaran lewat objek komunikator (MPI/ThreadComm)."""

    def __init__(self, terrain, params, rain, inflows, settings, comm=None):
        super().__init__(params, rain, inflows, settings, terrain.z.shape)
        self.comm = comm if comm is not None else SerialComm()
        r0, r1 = partition_rows(terrain.z.shape[0], self.comm.size)[self.comm.rank]
        self._sub = Subdomain(r0, r1, terrain.z, terrain.manning, terrain.infil_mm_h,
                              terrain.active, params, self.inflows)
        self._up = self.comm.rank - 1 if self.comm.rank > 0 else None
        self._down = self.comm.rank + 1 if self.comm.rank < self.comm.size - 1 else None

    def subs(self): return [self._sub]
    def is_root(self): return self.comm.rank == 0
    def nparts(self): return self.comm.size

    def exchange_h(self):
        s, c = self._sub, self.comm
        first, last = s.edge_h()
        top = c.sendrecv(last, dest=self._down, source=self._up)       # kirim ke bawah, terima dari atas
        bottom = c.sendrecv(first, dest=self._up, source=self._down)   # kirim ke atas, terima dari bawah
        s.set_ghost_h(top=top, bottom=bottom)

    def exchange_scale(self):
        s, c = self._sub, self.comm
        first, last = s.edge_scale()
        top = c.sendrecv(last, dest=self._down, source=self._up)
        bottom = c.sendrecv(first, dest=self._up, source=self._down)
        s.set_ghost_scale(top=top, bottom=bottom)

    def reduce_max(self, x): return self.comm.allreduce_max(x)
    def reduce_sum(self, vec): return self.comm.allreduce_sum(vec)

    def gather_rows(self, arrays):
        parts = self.comm.gather(arrays[0])
        return np.vstack(parts) if parts is not None else None
