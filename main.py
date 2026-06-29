"""
main.py
-------
Entry point. Builds the simulation and launches the visualiser.

  python main.py
"""

import config
from simulation    import Simulation
from visualisation import Visualizer


def banner():
    print("=" * 70)
    print()
    print(f"  Team       : {config.NUM_ROBOTS} ground robots (homogeneous)")
    print(f"  Map        : {config.MAP_W} × {config.MAP_H} px")
    print()
    print("  Modes:  1 = Hazard Localization")
    print("          2 = Sensor Degradation & Trust")
    print("          3 = Formation Control")
    print("=" * 70)


if __name__ == "__main__":
    banner()
    Visualizer(Simulation()).run()
