"""
config/default.py
-----------------
All tunable parameters in one place
"""

# Map 
MAP_W = 900
MAP_H = 600

# Optional obstacles
USE_OBSTACLES = False
# When enabled, OBSTACLES is a list of (x0, y0, x1, y1) rectangles
OBSTACLES = [
    (200, 120, 280, 360),
    (430, 250, 520, 480),
    (640, 100, 720, 320),
]

# Gas field 
GRID_W = 90
GRID_H = 60

# Plume model selector "diffusion" (fast, default) or "pasquill_gifford"
PLUME_MODEL = "diffusion"

# Atmospheric stability class (A=most unstable, F=stable). "D" = neutral
PASQUILL_STABILITY = "D"

# Gaussian-plume stack parameters
STACK_HEIGHT_M  = 3.0       # H_s stack height in metres
STACK_EMISSION  = 5.0       # Q emission rate (g/s)

# Diffusion-model parameters (used when PLUME_MODEL == "diffusion")
NUM_GAS_SOURCES = 1
GAS_EMIT_RATE   = 0.92
GAS_DIFFUSION   = 0.45
GAS_DECAY       = 0.003
WIND_VEC        = (0.06, 0.02)
DETECT_THRESH   = 0.005

# Sensor model MiniPID 2 by Ion Science
# Rated sensitivity ≈ 5 ppb. We follow Facinelli et al. and set the
# detection threshold to h_m = 6 × sensitivity = 30 ppb conservatively
# In our normalised gas units (max=1 ≈ 100 µg/m³) this is 0.0003;
# we use 0.005 to keep detection events happen within reasonable patrol
# time while still being below the spec-derived threshold
MINIPID2_SENSITIVITY_PPB    = 5.0    # rated sensitivity
MINIPID2_NOISE_FLOOR_PPB    = 10.0   # 1-sigma noise floor
MINIPID2_RANGE_M            = 7.0    # effective range (~70 px)

# Noise std σ(d) = σ0 · (1 + (d/ρ)²)
# σ0 calibrated from datasheet: noise_floor / max_concentration ≈ 0.01
MINIPID2_SIGMA_0            = 0.01

# Robots (homogeneous ground team)
NUM_ROBOTS      = 5
ROBOT_SPEED     = 2.0
SENSE_RADIUS    = 70
COMM_RADIUS     = 500       # px ≈ 50 m typical for WiFi/radio at this scale

SENSOR_QUALITY  = 1.0           # base gossip weight
DEGRADE_RATE    = 0.006         # exposure per step while in gas
SAFE_THRESHOLD  = 0.10
DEGRADE_STAGE1  = 0.30
DEGRADE_STAGE2  = 0.65
DEGRADE_FATAL   = 1.0

# Gossip Consensus
GOSSIP_ROUNDS   = 18

# Distributed Kalman Filter
DKF_INIT_COV      = 8000.0
DKF_OBS_NOISE     = 1500.0
DKF_PROCESS_NOISE = 25.0

# Lloyd–Voronoi
LLOYD_ALPHA     = 0.08

# Exploration strategy "lloyd" (default) or "random_walk"
EXPLORATION_STRATEGY = "lloyd"

# Consensus detection 
CONSENSUS_THRESH       = 25   # px tighter now that seeds are low-noise
CONSENSUS_STABLE_STEPS = 30

# Scenario timing 
MONITOR_STEPS = 200       # patrol time before next automatic leak
REPORT_HOLD_STEPS = 220   # hold REPORTED phase so user sees robots
                          # actively returning to their Voronoi cells

# Trust / Reputation (Mode 2)
TRUST_LEARNING_RATE = 0.05      # how fast trust adapts to disagreement
TRUST_MIN           = 0.05
TRUST_MAX           = 1.20
TRUST_DISAGREE_PX   = 60        # px distance counted as "disagreement"
TRUST_EXCLUDE_THRESHOLD = 0.40  # below this, robot is excluded from
                                # gossip and from the final network belief
                                # (purely observed criterion see kalman.py)

# Formation Control (Mode 3)
FORMATION_SHAPE   = "circle"    # "circle", "line", "v"
FORMATION_RADIUS  = 90          # for circle / V
FORMATION_SPACING = 50          # for line
FORMATION_GAIN    = 0.06        # consensus convergence rate
LEADER_PATH_TYPE  = "random_waypoints" # "circle", "lissajous", "random_waypoints"

# Animation
FPS       = 30
TRAIL_LEN = 50

# Visual theme (light / academic)
THEME_BG       = "#ffffff"
THEME_PANEL    = "#fafafa"
THEME_GRID     = "#e8e8ee"
THEME_TEXT     = "#1a1a2e"
THEME_DIM      = "#9aa0b0"
THEME_BORDER   = "#cfd0d8"
THEME_ACCENT   = "#3b7a8c"      # muted teal
THEME_WARN     = "#c89000"
THEME_DANGER   = "#b04040"
THEME_OK       = "#5a8b5a"

# Muted robot palette
PALETTE = [
    "#3b7a8c",  # teal
    "#b04040",  # muted red
    "#7a5a8c",  # purple
    "#c89000",  # ochre
    "#5a8b5a",  # green
    "#8c6a3b",  # brown
    "#4a6080",  # slate blue
    "#a86060",  # rose
]
