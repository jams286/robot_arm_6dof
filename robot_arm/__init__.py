"""
robot_arm — 6-DOF revolute serial manipulator toolkit.

Public API
----------
Robot6DOF             Main robot class
dh_matrix             Single DH transform
forward_kinematics    FK chain (all frames)
get_end_effector_pose FK → end-effector only
geometric_jacobian    6×n geometric Jacobian
manipulability_index  Yoshikawa measure
is_singular           Singularity check
ik_jacobian_transpose IK solver (JT)
ik_pseudoinverse      IK solver (pseudoinverse)
ik_dls                IK solver (damped least squares)
IKResult / IKStatus   Result containers
multi_segment_trajectory  Joint-space trajectory
RobotVisualizer       Interactive 3D viewer
plot_ik_convergence   IK convergence plot
"""

from .forward_kinematics import (
    dh_matrix,
    forward_kinematics,
    get_end_effector_pose,
    get_joint_positions,
    rotation_matrix_to_euler_zyx,
    pose_error,
)

from .jacobian import (
    geometric_jacobian,
    manipulability_index,
    minimum_singular_value,
    is_singular,
    condition_number,
)

from .inverse_kinematics import (
    ik_jacobian_transpose,
    ik_pseudoinverse,
    ik_dls,
    IKResult,
    IKStatus,
)

from .robot import Robot6DOF

from .trajectory import (
    cubic_interpolation,
    linear_interpolation,
    multi_segment_trajectory,
    cartesian_line,
)

from .visualizer import RobotVisualizer, plot_ik_convergence

from .config import PRESETS, IK_CONFIG
