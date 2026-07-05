"""
environment/gas_field.py
------------------------
2-D toxic gas field with controllable point sources (one or many).

Dynamics per step:
  1. inject gas at every active source cell
  2. Gaussian diffusion
  3. wind advection
  4. linear decay

Multiple-source support is required so we can match the Kapoutsis et al.
setup with NUM_GAS_SOURCES > 1.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

import config


def _px_to_cell(x, y):
    cx = int(np.clip(x / config.MAP_W * config.GRID_W, 0, config.GRID_W - 1))
    cy = int(np.clip(y / config.MAP_H * config.GRID_H, 0, config.GRID_H - 1))
    return cx, cy


def _cell_to_px(cx, cy):
    x = (cx + 0.5) / config.GRID_W * config.MAP_W
    y = (cy + 0.5) / config.GRID_H * config.MAP_H
    return x, y


class GasField:
    """Gas concentration grid + multi-source emission control."""

    def __init__(self):
        self.grid       = np.zeros((config.GRID_H, config.GRID_W), dtype=float)
        # active sources: list of (px_x, px_y) tuples
        self.sources    = []
        self._source_cells = []
        self.wind       = np.array(config.WIND_VEC, dtype=float)
        self._wind_age  = 0

    # Control 

    def add_source(self, source_px):
        """Activate a new emission point."""
        self.sources.append(tuple(source_px))
        self._source_cells.append(_px_to_cell(*source_px))

    def stop_all(self):
        self.sources.clear()
        self._source_cells.clear()

    def clear(self):
        self.grid[:] = 0.0
        self.stop_all()

    @property
    def emitting(self):
        return len(self.sources) > 0

    # Dynamics 

    def step(self):
        # inject gas at all active sources
        for cx, cy in self._source_cells:
            self.grid[cy, cx] = min(self.grid[cy, cx] + config.GAS_EMIT_RATE, 1.0)

        self.grid = gaussian_filter(self.grid, sigma=config.GAS_DIFFUSION)
        self.grid = nd_shift(self.grid,
                             shift=[self.wind[1], self.wind[0]],
                             mode="constant", cval=0.0)
        self.grid = np.clip(self.grid - config.GAS_DECAY, 0, 1)

        # slow wind drift
        self._wind_age += 1
        if self._wind_age > 55:
            self._wind_age = 0
            self.wind = np.clip(np.array(config.WIND_VEC) +
                                np.random.uniform(-0.04, 0.04, 2), -0.3, 0.3)

    # Queries 

    def sample_at_px(self, x, y, radius_px):
        """Mean concentration in a circle around (x, y)."""
        cx, cy = _px_to_cell(x, y)
        r = max(1, int(radius_px / config.MAP_W * config.GRID_W))
        x0, x1 = max(0, cx - r), min(config.GRID_W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(config.GRID_H, cy + r + 1)
        return float(self.grid[y0:y1, x0:x1].mean())

    def peak_in_radius_px(self, x, y, radius_px):
        """Pixel position of the highest cell in the sensing radius."""
        cx, cy = _px_to_cell(x, y)
        r = max(1, int(radius_px / config.MAP_W * config.GRID_W))
        x0, x1 = max(0, cx - r), min(config.GRID_W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(config.GRID_H, cy + r + 1)
        patch = self.grid[y0:y1, x0:x1]
        if patch.max() < 0.01:
            return None
        idx_y, idx_x = np.unravel_index(patch.argmax(), patch.shape)
        return _cell_to_px(x0 + idx_x, y0 + idx_y)

    def global_peak_px(self):
        """Pixel position of the strongest grid cell overall."""
        if self.grid.max() < 0.01:
            return None
        idx_y, idx_x = np.unravel_index(self.grid.argmax(), self.grid.shape)
        return _cell_to_px(idx_x, idx_y)

    def is_clear(self, threshold=0.02):
        return self.grid.max() < threshold
