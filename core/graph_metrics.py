"""
core/graph_metrics.py
---------------------
Communication-graph analysis: adjacency, Laplacian, algebraic connectivity λ₂.

Why λ₂ matters
--------------
For Laplacian L = D − A:
    0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ
λ₂ (the Fiedler value) is the algebraic connectivity:
  • λ₂ = 0   ⇔  graph disconnected — gossip cannot reach all nodes
  • larger λ₂ ⇔ faster consensus
  • per-step error contraction bounded by |1 − ε·λ₂|

We plot λ₂(t) live to expose network health and to validate that
consensus speeds up / slows down accordingly.
"""

import numpy as np

import config


class CommGraph:
    """Communication graph among alive robots within comm range."""

    def __init__(self, robots):
        self.robots = robots
        self.A: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self._alive_ids: list = []

    def update(self):
        alive = [r for r in self.robots if r.is_alive]
        self._alive_ids = [r.id for r in alive]
        n = len(alive)
        if n < 2:
            self.A = self.L = None
            return

        A = np.zeros((n, n), dtype=float)
        for i, ri in enumerate(alive):
            for j in range(i + 1, n):
                rj = alive[j]
                if np.linalg.norm(ri.position - rj.position) < ri.comm_r:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        D = np.diag(A.sum(axis=1))
        self.A = A
        self.L = D - A

    def lambda_2(self) -> float:
        if self.L is None or self.L.shape[0] < 2:
            return 0.0
        eigs = np.linalg.eigvalsh(self.L)
        return float(sorted(eigs)[1])

    def is_connected(self) -> bool:
        return self.lambda_2() > 1e-9

    def num_components(self) -> int:
        if self.L is None or self.L.shape[0] < 1:
            return 0
        eigs = np.linalg.eigvalsh(self.L)
        return int(np.sum(eigs < 1e-9))

    def edges(self):
        if self.A is None:
            return
        for i in range(self.A.shape[0]):
            for j in range(i + 1, self.A.shape[1]):
                if self.A[i, j] > 0:
                    yield self._alive_ids[i], self._alive_ids[j]
