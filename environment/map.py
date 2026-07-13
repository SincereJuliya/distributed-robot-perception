"""
environment/map.py
------------------
Map geometry: bounds, spawn helpers

"""

import random
import numpy as np
from shapely.geometry import box, Point
from shapely.ops import unary_union

import config


def build_free_space():
    """
    Build the navigable free space as a shapely polygon.
    """
    outer = box(0, 0, config.MAP_W, config.MAP_H)
    return outer


def random_spawn_points(n, free_space, min_sep=80, margin=80, max_tries=15_000):
    """Sample n spawn points inside the free-space polygon."""
    pts, tries = [], 0
    while len(pts) < n and tries < max_tries:
        x = random.uniform(margin, config.MAP_W - margin)
        y = random.uniform(margin, config.MAP_H - margin)
        if free_space is not None and not free_space.contains(Point(x, y)):
            tries += 1; continue
        if any(((x - px) ** 2 + (y - py) ** 2) ** 0.5 < min_sep
               for px, py in pts):
            tries += 1; continue
        pts.append((x, y))
        tries += 1
    return pts


def random_leak_point(robots, free_space, margin=80, min_dist_to_robots=60):
    """
    Pick a uniformly random leak location inside the free space, with a
    minimum-distance requirement from existing robots (so it doesn't spawn
    inside a robot's sensing circle).

    Truly random - every cycle gives a fresh location.
    """
    alive = [r for r in robots if r.is_alive]
    for _ in range(200):
        x = random.uniform(margin, config.MAP_W - margin)
        y = random.uniform(margin, config.MAP_H - margin)
        if free_space is not None and not free_space.contains(Point(x, y)):
            continue
        if alive:
            min_d = min(((x - r.position[0]) ** 2 +
                         (y - r.position[1]) ** 2) ** 0.5 for r in alive)
            if min_d < min_dist_to_robots:
                continue
        return (x, y)
    # Fallback: ignore distance constraint
    return (random.uniform(margin, config.MAP_W - margin),
            random.uniform(margin, config.MAP_H - margin))