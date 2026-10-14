"""Abstraksi komunikasi antar-proses.

Runner terdistribusi hanya butuh empat operasi: ``sendrecv``, ``allreduce_max``,
``allreduce_sum``, dan ``gather``. Tiga implementasi disediakan:

* ``MPIComm``    : pembungkus tipis ``mpi4py`` untuk klaster/HPC sungguhan.
* ``ThreadComm`` : replika semantik MPI memakai thread dalam satu proses. Dipakai
                   untuk menguji jalur terdistribusi tanpa MPI (CI, laptop).
* ``SerialComm`` : satu "proses", tanpa komunikasi.
"""
from __future__ import annotations

import queue
import threading

import numpy as np


class SerialComm:
    rank = 0
    size = 1

    def sendrecv(self, send, dest, source):
        return None

    def allreduce_max(self, x):
        return x

    def allreduce_sum(self, vec):
        return np.asarray(vec, dtype=float)

    def gather(self, obj):
        return [obj]


class MPIComm:
    """Pembungkus ``mpi4py.MPI.COMM_WORLD``."""

    def __init__(self, comm=None):
        from mpi4py import MPI  # impor lambat agar paket tetap bisa dipakai tanpa mpi4py
        self._MPI = MPI
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def sendrecv(self, send, dest, source):
        MPI = self._MPI
        send = np.ascontiguousarray(send)
        recv = np.empty_like(send)
        self.comm.Sendrecv(
            send, dest=MPI.PROC_NULL if dest is None else dest, sendtag=0,
            recvbuf=recv, source=MPI.PROC_NULL if source is None else source, recvtag=0)
        return None if source is None else recv

    def allreduce_max(self, x):
        return self.comm.allreduce(x, op=self._MPI.MAX)

    def allreduce_sum(self, vec):
        vec = np.ascontiguousarray(vec, dtype=float)
        out = np.empty_like(vec)
        self.comm.Allreduce(vec, out, op=self._MPI.SUM)
        return out

    def gather(self, obj):
        return self.comm.gather(obj, root=0)


class _Shared:
    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n)
        self.slots = [None] * n
        self.queues = {(s, d): queue.Queue() for s in range(n) for d in range(n)}


class ThreadComm:
    """Komunikator berbasis thread dengan semantik mirip MPI (untuk pengujian)."""

    TIMEOUT = 120.0

    def __init__(self, rank, shared):
        self.rank, self.size, self._sh = rank, shared.n, shared

    @classmethod
    def group(cls, n):
        sh = _Shared(n)
        return [cls(r, sh) for r in range(n)]

    def sendrecv(self, send, dest, source):
        if dest is not None:
            self._sh.queues[(self.rank, dest)].put(np.array(send, copy=True))
        if source is None:
            return None
        return self._sh.queues[(source, self.rank)].get(timeout=self.TIMEOUT)

    def _exchange(self, value):
        sh = self._sh
        sh.slots[self.rank] = value
        sh.barrier.wait(self.TIMEOUT)
        snapshot = list(sh.slots)
        sh.barrier.wait(self.TIMEOUT)
        return snapshot

    def allreduce_max(self, x):
        return max(self._exchange(x))

    def allreduce_sum(self, vec):
        return np.sum(self._exchange(np.array(vec, dtype=float)), axis=0)

    def gather(self, obj):
        everything = self._exchange(obj)
        return everything if self.rank == 0 else None


def run_in_threads(fn, nthreads):
    """Jalankan ``fn(comm)`` pada ``nthreads`` thread; kembalikan daftar hasil per rank."""
    comms = ThreadComm.group(nthreads)
    results, errors = [None] * nthreads, []

    def worker(r):
        try:
            results[r] = fn(comms[r])
        except BaseException as exc:  # noqa: BLE001 - diteruskan ke thread utama
            errors.append(exc)
            comms[r]._sh.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        # utamakan galat asli, bukan BrokenBarrierError turunan dari barrier.abort()
        real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        raise (real or errors)[0]
    return results
