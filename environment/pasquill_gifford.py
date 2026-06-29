"""
environment/pasquill_gifford.py
-------------------------------
Analytical Pasquill–Gifford Gaussian Plume model — direct
implementation of Eq. (2) from Facinelli, Larcher, Brunelli, Fontanelli
"Cooperative UAVs Gas Monitoring using Distributed Consensus"
(IEEE COMPSAC 2019, arXiv:1907.07279).

The local concentration C(x, y, z) at distance (x, y, z) from a stack
of height H_s is:

    C(x, y, z, H_s) = (Q / w)
                    * 1 / (2 * pi * sigma_y(x) * sigma_z(x))
                    * exp(-y^2 / (2 * sigma_y(x)^2))
                    * [ exp(-(z - H_s)^2 / (2 * sigma_z(x)^2))
                      + exp(-(z + H_s)^2 / (2 * sigma_z(x)^2)) ]   (Eq. 2)

with
    sigma_y(x) = c * x^d
    sigma_z(x) = a * x^b

where (a, b, c, d) come from the Pasquill–Gifford stability table
(Davidson 1990, also reproduced in Facinelli et al. § II-A).

This module provides a stand-alone, vectorised evaluation of the
plume on the simulator grid so it can be used as a drop-in
replacement of the grid-based diffusion model. The same `GasField`
interface is preserved: `step()`, `sample_at_px()`,
`peak_in_radius_px()`, `add_source()`, `clear()`, etc.
"""

import numpy as np
import config


# Pasquill stability table (a, b, c, d) coefficients per stability class.
# Values from Davidson, J. Air & Waste Management Assoc. 40(8), 1990.
# Class A = extremely unstable, F = stable. The default below ("D")
# corresponds to neutral conditions and matches the figures in
# Facinelli et al. 2019.
PASQUILL_TABLE = {
    "A": {"a": 213.0, "b": 0.894, "c": 440.8, "d": 1.941},
    "B": {"a": 156.0, "b": 0.894, "c": 100.0, "d": 1.149},
    "C": {"a": 104.0, "b": 0.894, "c": 61.0,  "d": 0.911},
    "D": {"a": 68.0,  "b": 0.894, "c": 33.2,  "d": 0.725},
    "E": {"a": 50.5,  "b": 0.894, "c": 22.8,  "d": 0.678},
    "F": {"a": 34.0,  "b": 0.894, "c": 14.35, "d": 0.740},
}


class PasquillGiffordPlume:
    """
    Vectorised Pasquill–Gifford plume on the simulator grid.

    Coordinates are in pixels (the simulator native unit).
    The conversion factor PX_PER_M = 10 means 1 m = 10 px.
    Stack height H_s and emission rate Q are passed via config.

    Note: when multiple sources are active they are superposed
    linearly (a property of the underlying diffusion equation).
    """

    PX_PER_M = 10.0           # 1 m = 10 px on our map

    def __init__(self, stability="D"):
        self.stability   = stability
        self.coeff       = PASQUILL_TABLE[stability]
        self.grid        = np.zeros((config.GRID_H, config.GRID_W),
                                    dtype=float)
        self.sources     = []          # list of (px_x, px_y) tuples
        self.wind        = np.array(config.WIND_VEC, dtype=float)
        self._wind_age   = 0
        # Pre-compute grid cell centres in pixel space, then in metres
        ys = (np.arange(config.GRID_H) + 0.5) / config.GRID_H * config.MAP_H
        xs = (np.arange(config.GRID_W) + 0.5) / config.GRID_W * config.MAP_W
        self._gx_px, self._gy_px = np.meshgrid(xs, ys)         # (H, W)
        # Receptor height z = stack height by default (UAV / ground
        # level co-planar to source as in eq.3-4 of the paper)
        self.H_s_m = float(getattr(config, "STACK_HEIGHT_M", 3.0))
        self.z_m   = self.H_s_m
        # Emission rate Q (mass per second). Scaled to give max ≈ 1
        # in the grid for visual compatibility with the diffusion model
        self.Q     = float(getattr(config, "STACK_EMISSION", 5.0))

    # ── Control ──────────────────────────────────────────────────────────────

    def add_source(self, source_px):
        self.sources.append(tuple(source_px))

    def stop_all(self):
        self.sources.clear()

    def clear(self):
        self.grid[:] = 0.0
        self.sources.clear()

    @property
    def emitting(self):
        return len(self.sources) > 0

    # ── Pasquill σ_y, σ_z as a function of along-wind distance x ─────────────

    def _sigmas(self, x_m):
        """σ_y(x) = c·x^d, σ_z(x) = a·x^b. x_m in metres, ≥ 0."""
        x_safe = np.maximum(x_m, 1.0)         # avoid x = 0
        c, d   = self.coeff["c"], self.coeff["d"]
        a, b   = self.coeff["a"], self.coeff["b"]
        # Divide by 100 to keep σ in a reasonable range for our 90 m map
        sigma_y = (c * np.power(x_safe, d)) / 100.0
        sigma_z = (a * np.power(x_safe, b)) / 100.0
        return sigma_y, sigma_z

    # ── Concentration at a single point — Eq. (2) and (3) of the paper ──────

    def _concentration_pt(self, source_px, x_px, y_px, z_m=None):
        """
        Compute C at receptor (x_px, y_px) for one source at source_px.
        Wind blows along +X by convention; we rotate the coordinate so
        +x is the along-wind direction from the source.
        """
        if z_m is None:
            z_m = self.z_m

        sx_px, sy_px = source_px
        # Receptor offset from source in pixels
        dx_px = x_px - sx_px
        dy_px = y_px - sy_px

        # Rotate so +x aligns with wind direction
        w = self.wind
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-6:
            # No wind → use raw axes
            xr = dx_px
            yr = dy_px
        else:
            wx, wy = w / w_norm
            xr =  wx * dx_px + wy * dy_px        # along-wind
            yr = -wy * dx_px + wx * dy_px        # cross-wind

        # Convert to metres
        x_m = xr / self.PX_PER_M
        y_m = yr / self.PX_PER_M

        # Upwind (x < 0) → no concentration in the Gaussian-plume model
        # (the plume is advected downwind from the stack)
        upwind = x_m < 0

        sigma_y, sigma_z = self._sigmas(np.abs(x_m))

        # Wind speed in m/s (norm of wind vector, scaled for realism)
        w_speed = max(0.1, w_norm * 10.0)

        # Eq. (2) of Facinelli/Fontanelli 2019
        prefactor = (self.Q / w_speed) * (1.0 /
                    (2.0 * np.pi * sigma_y * sigma_z))
        gauss_y   = np.exp(-(y_m ** 2) / (2.0 * sigma_y ** 2))
        gauss_z   = (np.exp(-((z_m - self.H_s_m) ** 2) /
                            (2.0 * sigma_z ** 2)) +
                     np.exp(-((z_m + self.H_s_m) ** 2) /
                            (2.0 * sigma_z ** 2)))

        C = prefactor * gauss_y * gauss_z
        # Mask out upwind regions (no diffusion against wind)
        C = np.where(upwind, 0.0, C)
        return C

    # ── Per-step grid update ─────────────────────────────────────────────────

    def step(self):
        """
        Recompute the analytical plume on the whole grid.

        Implementation note: Eq. (2) is an instantaneous steady-state
        formula. We model the leak as a build-up to that steady state
        by EWMA on the analytical field, so the visual transient
        (gradual leak appearance) is preserved.
        """
        if not self.sources:
            # Decay any residual gas
            self.grid *= 0.95
            self.grid = np.where(self.grid < 1e-4, 0.0, self.grid)
            return

        # Compute analytical field from all sources (linear superposition)
        steady = np.zeros_like(self.grid)
        for src in self.sources:
            steady += self._concentration_pt(
                src, self._gx_px, self._gy_px)

        # Normalise so the maximum is around 1 — keeps the colormap
        # consistent with the diffusion model
        m = steady.max()
        if m > 1e-9:
            steady = steady / m

        # EWMA build-up (so the leak appears gradually, matching
        # the dynamics of the diffusion model and giving the robots
        # time to detect it through patrolling)
        alpha = 0.10
        self.grid = (1 - alpha) * self.grid + alpha * steady
        self.grid = np.clip(self.grid, 0.0, 1.0)

        # Slow wind drift (same as in diffusion model)
        self._wind_age += 1
        if self._wind_age > 55:
            self._wind_age = 0
            self.wind = np.clip(np.array(config.WIND_VEC) +
                                np.random.uniform(-0.04, 0.04, 2),
                                -0.3, 0.3)

    # ── Queries — same interface as GasField ─────────────────────────────────

    def _px_to_cell(self, x, y):
        cx = int(np.clip(x / config.MAP_W * config.GRID_W,
                          0, config.GRID_W - 1))
        cy = int(np.clip(y / config.MAP_H * config.GRID_H,
                          0, config.GRID_H - 1))
        return cx, cy

    def _cell_to_px(self, cx, cy):
        x = (cx + 0.5) / config.GRID_W * config.MAP_W
        y = (cy + 0.5) / config.GRID_H * config.MAP_H
        return x, y

    def sample_at_px(self, x, y, radius_px):
        cx, cy = self._px_to_cell(x, y)
        r = max(1, int(radius_px / config.MAP_W * config.GRID_W))
        x0, x1 = max(0, cx - r), min(config.GRID_W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(config.GRID_H, cy + r + 1)
        return float(self.grid[y0:y1, x0:x1].mean())

    def peak_in_radius_px(self, x, y, radius_px):
        cx, cy = self._px_to_cell(x, y)
        r = max(1, int(radius_px / config.MAP_W * config.GRID_W))
        x0, x1 = max(0, cx - r), min(config.GRID_W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(config.GRID_H, cy + r + 1)
        patch = self.grid[y0:y1, x0:x1]
        if patch.max() < 0.01:
            return None
        idx_y, idx_x = np.unravel_index(patch.argmax(), patch.shape)
        return self._cell_to_px(x0 + idx_x, y0 + idx_y)

    def global_peak_px(self):
        if self.grid.max() < 0.01:
            return None
        idx_y, idx_x = np.unravel_index(self.grid.argmax(), self.grid.shape)
        return self._cell_to_px(idx_x, idx_y)

    def is_clear(self, threshold=0.02):
        return self.grid.max() < threshold
