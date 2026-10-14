"""Curah hujan: hyetograph konstan-per-interval, hujan rencana, dan pembaca CSV."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Hyetograph:
    """Intensitas hujan seragam di seluruh domain.

    ``intensity_mm_h[k]`` berlaku pada interval ``[times_s[k], times_s[k+1])``.
    Nilai terakhir hanya penanda akhir hujan dan diabaikan (setelah itu hujan = 0).
    """

    times_s: np.ndarray
    intensity_mm_h: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.times_s, dtype=float)
        i = np.asarray(self.intensity_mm_h, dtype=float)
        if t.ndim != 1 or t.shape != i.shape or len(t) < 2:
            raise ValueError("hyetograph butuh minimal dua baris dengan panjang waktu = panjang intensitas")
        if np.any(np.diff(t) <= 0):
            raise ValueError("waktu hyetograph harus naik monoton")
        if np.any(i < 0):
            raise ValueError("intensitas hujan tidak boleh negatif")
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "intensity_mm_h", i)

    def intensity_mm_h_at(self, t: float) -> float:
        k = int(np.searchsorted(self.times_s, t, side="right")) - 1
        if k < 0 or k >= len(self.times_s) - 1:
            return 0.0
        return float(self.intensity_mm_h[k])

    def intensity_ms(self, t: float) -> float:
        """Intensitas dalam m/s (satuan yang dipakai solver)."""
        return self.intensity_mm_h_at(t) / 1000.0 / 3600.0

    def total_mm(self) -> float:
        return float(np.sum(self.intensity_mm_h[:-1] * np.diff(self.times_s)) / 3600.0)

    @property
    def end_time_s(self) -> float:
        return float(self.times_s[-1])


def design_storm(total_mm: float, duration_min: float, peak_position: float = 0.4,
                 interval_min: float = 10.0, shape: float = 3.0) -> Hyetograph:
    """Hujan rencana berbentuk distribusi Beta (puncak di ``peak_position`` * durasi).

    Total kedalaman hujan tepat sama dengan ``total_mm``.
    """
    if not 0.0 < peak_position < 1.0:
        raise ValueError("peak_position harus di antara 0 dan 1")
    if total_mm < 0 or duration_min <= 0 or interval_min <= 0:
        raise ValueError("total_mm >= 0, duration_min > 0, interval_min > 0")
    nb = max(1, int(round(duration_min / interval_min)))
    edges = np.linspace(0.0, 1.0, nb + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    a = shape
    b = (a - 1.0) / peak_position - a + 2.0     # modus Beta(a, b) = peak_position
    w = mid ** (a - 1.0) * (1.0 - mid) ** (b - 1.0)
    depth_mm = total_mm * w / w.sum()
    step_min = duration_min / nb
    intensity = depth_mm / (step_min / 60.0)    # mm/jam
    times = np.arange(nb + 1) * step_min * 60.0
    return Hyetograph(times, np.append(intensity, 0.0))


def hyetograph_from_csv(path) -> Hyetograph:
    """Baca CSV dengan kolom ``time_min`` dan ``intensity_mm_h``.

    Setiap baris berlaku sampai baris berikutnya; baris terakhir sebaiknya
    berintensitas 0 sebagai penanda akhir hujan.
    """
    rows = []
    with open(Path(path), newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            rows.append((float(rec["time_min"]) * 60.0, float(rec["intensity_mm_h"])))
    if len(rows) < 2:
        raise ValueError(f"{path}: minimal dua baris data")
    rows.sort()
    t = np.array([r[0] for r in rows])
    i = np.array([r[1] for r in rows])
    if t[0] > 0:  # sebelum data pertama: tidak hujan
        t = np.insert(t, 0, 0.0)
        i = np.insert(i, 0, 0.0)
    return Hyetograph(t, i)
