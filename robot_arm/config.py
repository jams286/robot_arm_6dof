"""
Configuration: joint limits, DH presets, solver parameters.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Joint limits (radians) — [min, max] per joint
# ---------------------------------------------------------------------------
DEFAULT_JOINT_LIMITS = np.array([
    [-np.pi,     np.pi    ],   # Joint 1  – base rotation
    [-np.pi/2,   np.pi/2  ],   # Joint 2  – shoulder
    [-np.pi,     np.pi    ],   # Joint 3  – elbow
    [-np.pi,     np.pi    ],   # Joint 4  – wrist roll
    [-np.pi/2,   np.pi/2  ],   # Joint 5  – wrist pitch
    [-np.pi,     np.pi    ],   # Joint 6  – wrist yaw
])

# ---------------------------------------------------------------------------
# Denavit-Hartenberg presets
# Each row: [a_i, alpha_i, d_i, theta_offset_i]  (SI units: metres, radians)
# theta_offset is added to the variable joint angle (useful for zero-pose cal)
# ---------------------------------------------------------------------------

# Generic 6-DOF arm (similar to a UR5 kinematic layout)
UR5_LIKE = {
    "name": "UR5-like",
    "dh": np.array([
        # a       alpha          d       theta_offset
        [0.000,   np.pi/2,   0.089159,   0.000],   # Joint 1
        [0.425,   0.000,     0.000,      0.000],   # Joint 2
        [0.392,   0.000,     0.000,      0.000],   # Joint 3
        [0.000,   np.pi/2,   0.109150,   0.000],   # Joint 4
        [0.000,  -np.pi/2,   0.094650,   0.000],   # Joint 5
        [0.000,   0.000,     0.082300,   0.000],   # Joint 6
    ]),
    "joint_limits": DEFAULT_JOINT_LIMITS,
}

# Simplified educational arm with unit lengths
SIMPLE_ARM = {
    "name": "Simple educational",
    "dh": np.array([
        [0.000,   np.pi/2,   0.4,   0.000],
        [0.500,   0.000,     0.0,   0.000],
        [0.400,   0.000,     0.0,   0.000],
        [0.000,   np.pi/2,   0.3,   0.000],
        [0.000,  -np.pi/2,   0.0,   0.000],
        [0.000,   0.000,     0.2,   0.000],
    ]),
    "joint_limits": DEFAULT_JOINT_LIMITS,
}

# PUMA 560 — classic industrial 6-DOF (spherical wrist)
PUMA_560 = {
    "name": "PUMA 560",
    "dh": np.array([
        # a       alpha          d       theta_offset
        [0.000,   np.pi/2,   0.6604,   0.000],   # Joint 1 — waist
        [0.4318,  0.000,     0.000,    0.000],   # Joint 2 — shoulder
        [0.0203,  np.pi/2,   0.1503,   0.000],   # Joint 3 — elbow
        [0.000,  -np.pi/2,   0.4331,   0.000],   # Joint 4 — wrist roll
        [0.000,   np.pi/2,   0.000,    0.000],   # Joint 5 — wrist pitch
        [0.000,   0.000,     0.0560,   0.000],   # Joint 6 — wrist yaw (flange)
    ]),
    "joint_limits": DEFAULT_JOINT_LIMITS,
}

# Stanford arm — 5R+1P (prismatic J3), approximated as all-revolute here
STANFORD_LIKE = {
    "name": "Stanford-like",
    "dh": np.array([
        [0.000,  -np.pi/2,   0.4120,   0.000],
        [0.000,   np.pi/2,   0.1540,   0.000],
        [0.000,   0.000,     0.6630,   0.000],
        [0.000,  -np.pi/2,   0.000,    0.000],
        [0.000,   np.pi/2,   0.000,    0.000],
        [0.000,   0.000,     0.2630,   0.000],
    ]),
    "joint_limits": DEFAULT_JOINT_LIMITS,
}

PRESETS = {
    "ur5_like":  UR5_LIKE,
    "simple":    SIMPLE_ARM,
    "puma560":   PUMA_560,
    "stanford":  STANFORD_LIKE,
}

# ---------------------------------------------------------------------------
# IK solver parameters
# ---------------------------------------------------------------------------
IK_CONFIG = {
    "max_iter": 1000,
    "tol_pos": 1e-4,          # metres
    "tol_ori": 1e-3,          # radians
    "alpha_transpose": 0.5,   # step size – Jacobian transpose
    "alpha_pinv": 1.0,        # step size – pseudoinverse
    "damping": 0.05,          # λ for Damped Least Squares
    "singularity_thresh": 1e-3,
    "pos_weight": 1.0,        # weight of position error in 6-D error
    "ori_weight": 0.3,        # weight of orientation error
}

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
VIZ_CONFIG = {
    "link_color":     "#4A90D9",
    "joint_color":    "#E87040",
    "ee_color":       "#2ECC71",
    "target_color":   "#E74C3C",
    "trajectory_color": "#9B59B6",
    "workspace_alpha": 0.05,
    "figure_size":    (14, 7),
}
