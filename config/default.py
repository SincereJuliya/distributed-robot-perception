"""
config/default.py
-----------------
All tunable parameters in one place.
"""

# Map
MAP_W = 900
MAP_H = 600

# Gas field grid
GRID_W = 90
GRID_H = 60

# Plume model: "diffusion" or "pasquill_gifford"
PLUME_MODEL = "diffusion"

# Atmospheric stability class (A=most unstable, F=stable). "D" = neutral
PASQUILL_STABILITY = "D"

# Gaussian-plume stack parameters
STACK_HEIGHT_M  = 3.0       # H_s stack height (m)
STACK_EMISSION  = 5.0       # Q emission rate (g/s)

# Diffusion-model parameters (PLUME_MODEL == "diffusion")
NUM_GAS_SOURCES = 1
GAS_EMIT_RATE   = 0.92
GAS_DIFFUSION   = 0.45      # Dc
GAS_DECAY       = 0.003     # kappa
WIND_VEC        = (0.06, 0.02)
DETECT_THRESH   = 0.005     # normalised units; tuned so detection happens
                            # within a reasonable patrol time

# Sensor noise std: sigma(d) = sigma_0 * (1 + (d/rho)^2)
# sigma_0 in normalised units (datasheet noise floor / max concentration)
MINIPID2_SIGMA_0 = 0.01

# Robots (homogeneous ground team)
NUM_ROBOTS      = 5
ROBOT_SPEED     = 2.0        # v_max (px/step)
SENSE_RADIUS    = 70         # rho (px)
COMM_RADIUS     = 500        # rc (px); default for Mode 1, swept in Exp. 2

SENSOR_QUALITY  = 1.0        # base gossip weight q
DEGRADE_RATE    = 0.006      # exposure per step while in gas
SAFE_THRESHOLD  = 0.10
DEGRADE_STAGE1  = 0.30       # healthy -> noisy
DEGRADE_STAGE2  = 0.65       # noisy -> failing
DEGRADE_FATAL   = 1.0        # failing -> dead

# Gossip consensus
GOSSIP_ROUNDS   = 18         # rounds per simulation step

# Distributed Kalman Filter
DKF_INIT_COV      = 8000.0
DKF_OBS_NOISE     = 1500.0   # R0 in R = max(R_min, R0 (d/rho)^2)
DKF_PROCESS_NOISE = 25.0     # q in P <- P + qI

# Lloyd-Voronoi
LLOYD_ALPHA     = 0.08       # centroid smoothing

# Exploration strategy: "lloyd" (default) or "random_walk"
EXPLORATION_STRATEGY = "lloyd"

# Consensus detection
CONSENSUS_THRESH       = 25   # Theta (px)
CONSENSUS_STABLE_STEPS = 30   # consecutive steps below Theta before REPORTED

# Scenario timing
MONITOR_STEPS = 200          # patrol time before next automatic leak
REPORT_HOLD_STEPS = 220      # hold REPORTED so robots visibly return to cells

# Trust / reputation (Mode 2)
TRUST_LEARNING_RATE = 0.05   # alpha, EMA rate
TRUST_MIN           = 0.05
TRUST_MAX           = 1.20
TRUST_DISAGREE_PX   = 60      # d_theta, disagreement scale (px)
TRUST_EXCLUDE_THRESHOLD = 0.40  # tau_exc; below this, excluded from gossip
                                # and from the final network belief

# Formation control (Mode 3)
FORMATION_SHAPE   = "circle"    # "circle", "line", "v"
FORMATION_RADIUS  = 90          # circle / V
FORMATION_SPACING = 50          # line
FORMATION_GAIN    = 0.2         # consensus convergence rate (Mode 3)
LEADER_PATH_TYPE  = "random_waypoints"  # "circle", "lissajous", "random_waypoints"

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
THEME_ACCENT   = "#3b7a8c"
THEME_WARN     = "#c89000"
THEME_DANGER   = "#b04040"
THEME_OK       = "#5a8b5a"

# Robot palette
PALETTE = [
    "#3b7a8c",  # teal
    "#b04040",  # red
    "#7a5a8c",  # purple
    "#c89000",  # ochre
    "#5a8b5a",  # green
    "#8c6a3b",  # brown
    "#4a6080",  # slate blue
    "#a86060",  # rose
]