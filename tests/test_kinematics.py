"""
Unit tests for forward kinematics, Jacobian, and IK solvers.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_arm import (
    Robot6DOF,
    dh_matrix,
    forward_kinematics,
    get_end_effector_pose,
    geometric_jacobian,
    manipulability_index,
    is_singular,
    ik_jacobian_transpose,
    ik_pseudoinverse,
    ik_dls,
    IKStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def robot():
    return Robot6DOF(preset="ur5_like")


@pytest.fixture
def simple_robot():
    return Robot6DOF(preset="simple")


# ---------------------------------------------------------------------------
# DH matrix tests
# ---------------------------------------------------------------------------

class TestDHMatrix:
    def test_identity_zero_params(self):
        T = dh_matrix(0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_shape(self):
        T = dh_matrix(0.3, 0.1, 0.5, np.pi/4)
        assert T.shape == (4, 4)

    def test_rotation_z(self):
        """Pure Z rotation: a=0, d=0, alpha=0."""
        theta = np.pi / 3
        T = dh_matrix(theta, 0.0, 0.0, 0.0)
        c, s = np.cos(theta), np.sin(theta)
        expected = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        np.testing.assert_allclose(T, expected, atol=1e-12)

    def test_determinant_is_one(self):
        """DH matrix must be a proper homogeneous transform."""
        T = dh_matrix(0.7, 0.2, 0.4, 1.1)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10

    def test_bottom_row(self):
        T = dh_matrix(1.0, 0.5, 0.3, 0.8)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# Forward Kinematics tests
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    def test_home_config_not_nan(self, robot):
        T = robot.fk(np.zeros(6))
        assert not np.any(np.isnan(T))

    def test_shape_is_4x4(self, robot):
        T = robot.fk(np.zeros(6))
        assert T.shape == (4, 4)

    def test_rotation_matrix_orthonormal(self, robot):
        q = np.radians([10, -30, 60, 15, -20, 5])
        T = robot.fk(q)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_joint_positions_count(self, robot):
        q = np.zeros(6)
        pts = robot.joint_positions(q)
        assert pts.shape == (7, 3)   # base + 6 joints

    def test_reproducible(self, robot):
        q = np.radians([20, -40, 70, 10, 30, -15])
        T1 = robot.fk(q)
        T2 = robot.fk(q)
        np.testing.assert_allclose(T1, T2, atol=1e-15)

    def test_different_configs_differ(self, robot):
        T1 = robot.fk(np.zeros(6))
        T2 = robot.fk(np.radians([0, -90, 0, 0, 0, 0]))
        assert not np.allclose(T1, T2)


# ---------------------------------------------------------------------------
# Jacobian tests
# ---------------------------------------------------------------------------

class TestJacobian:
    def test_shape(self, robot):
        J = robot.jacobian(np.zeros(6))
        assert J.shape == (6, 6)

    def test_not_all_zeros(self, robot):
        J = robot.jacobian(np.zeros(6))
        assert np.linalg.norm(J) > 1e-6

    def test_numerical_jacobian_consistency(self, robot):
        """Finite-difference check of the geometric Jacobian (position part)."""
        q    = np.radians([10, -30, 60, 0, 20, -10])
        J    = robot.jacobian(q)
        eps  = 1e-6
        J_fd = np.zeros((3, 6))

        T0 = robot.fk(q)
        p0 = T0[:3, 3]

        for i in range(6):
            dq       = np.zeros(6)
            dq[i]    = eps
            T_plus   = get_end_effector_pose(q + dq, robot.dh_params)
            J_fd[:, i] = (T_plus[:3, 3] - p0) / eps

        np.testing.assert_allclose(J[:3, :], J_fd, atol=1e-4)

    def test_manipulability_positive(self, robot):
        man = robot.manipulability(np.zeros(6))
        assert man >= 0

    def test_singular_config_detection(self, robot):
        """Fully extended arm is often near singular."""
        q_sing = np.zeros(6)
        report = robot.check_singularity(q_sing)
        assert "singular" in report
        assert "manipulability" in report
        assert "sigma_min" in report


# ---------------------------------------------------------------------------
# Inverse Kinematics tests
# ---------------------------------------------------------------------------

class TestIK:
    def _make_target(self, robot):
        q_true = np.radians([20, -40, 80, 10, 30, -15])
        return robot.fk(q_true), q_true

    def _initial_q(self, robot):
        return np.radians([5, -20, 50, 5, 15, -5])

    def test_dls_converges(self, robot):
        T_target, _ = self._make_target(robot)
        result = ik_dls(
            T_target, self._initial_q(robot),
            robot.dh_params, robot.joint_limits,
            max_iter=1000, tol_pos=1e-4,
        )
        assert result.success
        assert result.pos_error < 1e-3  # 1 mm

    def test_pinv_converges(self, robot):
        T_target, _ = self._make_target(robot)
        result = ik_pseudoinverse(
            T_target, self._initial_q(robot),
            robot.dh_params, robot.joint_limits,
            max_iter=1000, tol_pos=1e-4,
        )
        assert result.success or result.pos_error < 5e-3

    def test_transpose_converges(self, robot):
        T_target, _ = self._make_target(robot)
        result = ik_jacobian_transpose(
            T_target, self._initial_q(robot),
            robot.dh_params, robot.joint_limits,
            alpha=0.5, max_iter=3000, tol_pos=1e-4,
        )
        # transpose is slower; allow more tolerance
        assert result.pos_error < 0.01

    def test_ik_fk_roundtrip(self, robot):
        """IK solution must satisfy FK(q_sol) ≈ T_target."""
        q_true   = np.radians([25, -35, 75, 10, 25, -10])
        T_target = robot.fk(q_true)
        q0       = self._initial_q(robot)

        result = ik_dls(
            T_target, q0, robot.dh_params, robot.joint_limits,
            max_iter=1000, tol_pos=1e-4,
        )
        T_sol = get_end_effector_pose(result.joint_angles, robot.dh_params)
        np.testing.assert_allclose(T_sol[:3, 3], T_target[:3, 3], atol=1e-3)

    def test_joint_limits_respected(self, robot):
        q_true   = np.radians([25, -35, 75, 10, 25, -10])
        T_target = robot.fk(q_true)
        result   = ik_dls(
            T_target, np.zeros(6), robot.dh_params, robot.joint_limits,
        )
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        assert np.all(result.joint_angles >= lo - 1e-10)
        assert np.all(result.joint_angles <= hi + 1e-10)

    def test_robot_ik_method_api(self, robot):
        robot.home()
        q_true   = np.radians([15, -30, 60, 5, 20, -10])
        T_target = robot.fk(q_true)
        robot.home()
        result = robot.ik(T_target, method="dls")
        assert result is not None
        assert hasattr(result, "success")

    def test_unreachable_target_does_not_crash(self, robot):
        T_far = np.eye(4)
        T_far[:3, 3] = [100.0, 100.0, 100.0]   # way out of reach
        result = ik_dls(
            T_far, np.zeros(6), robot.dh_params, robot.joint_limits,
            max_iter=200,
        )
        assert not result.success
        assert result.status != IKStatus.CONVERGED


# ---------------------------------------------------------------------------
# Robot class integration
# ---------------------------------------------------------------------------

class TestRobotClass:
    def test_preset_ur5(self):
        robot = Robot6DOF(preset="ur5_like")
        assert robot.n_joints == 6

    def test_preset_simple(self):
        robot = Robot6DOF(preset="simple")
        assert robot.n_joints == 6

    def test_joint_limit_clipping(self):
        robot = Robot6DOF()
        huge  = np.full(6, 999.0)
        robot.q = huge
        assert np.all(robot.q <= robot.joint_limits[:, 1])

    def test_random_config_in_limits(self):
        robot = Robot6DOF()
        for _ in range(20):
            q = robot.random_config()
            assert np.all(q >= robot.joint_limits[:, 0])
            assert np.all(q <= robot.joint_limits[:, 1])

    def test_home_resets_to_zero(self):
        robot = Robot6DOF()
        robot.q = robot.random_config()
        robot.home()
        np.testing.assert_allclose(robot.q, np.zeros(6))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
