"""Unit tests for stance tracking and drift diagnostics."""
import mujoco
import numpy as np
import pytest

from g1_ping_pong_lab.diagnostics import StanceTracker, FeedStanceSummary


@pytest.fixture(scope="module")
def scene_model():
    return mujoco.MjModel.from_xml_path("assets/scene.xml")


class TestStanceTracker:
    def test_tracker_initialization(self, scene_model):
        tracker = StanceTracker(scene_model)
        assert tracker.pelvis_id > 0
        assert tracker.torso_id > 0
        assert tracker.lf_id > 0
        assert tracker.rf_id > 0
        assert len(tracker.lf_geoms) > 0
        assert len(tracker.rf_geoms) > 0

    def test_tracker_feed_lifecycle(self, scene_model):
        tracker = StanceTracker(scene_model)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        tracker.start_feed(0, data, 0.0)
        assert tracker.current_feed_id == 0
        assert tracker.feed_start_time == 0.0
        assert tracker._feed_init_state["pelvis"].shape == (3,)

        # Step physics and observe
        for _ in range(10):
            mujoco.mj_step(scene_model, data)
            tracker.observe_step(data, data.time)

        summary = tracker.finish_feed(data, data.time)
        assert isinstance(summary, FeedStanceSummary)
        assert summary.feed_id == 0
        assert summary.end_time > summary.start_time
        assert summary.pelvis_drift.shape == (3,)
        assert summary.torso_drift.shape == (3,)
        assert summary.com_drift.shape == (3,)
        assert summary.left_foot_drift.shape == (3,)
        assert summary.right_foot_drift.shape == (3,)
        assert isinstance(summary.to_dict(), dict)

    def test_tracker_slip_and_lift_detection(self, scene_model):
        tracker = StanceTracker(scene_model)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        tracker.start_feed(0, data, 0.0)
        # Settle robot onto ground (contacts occur around step 100)
        for _ in range(150):
            mujoco.mj_step(scene_model, data)
            tracker.observe_step(data, data.time)

        summary = tracker.finish_feed(data, data.time)
        assert summary.feed_id == 0
        assert summary.max_lf_tangential_force >= 0.0
        assert summary.max_rf_tangential_force >= 0.0
        assert summary.left_foot_slip_cm >= 0.0
        assert summary.right_foot_slip_cm >= 0.0


class TestStanceRecoveryController:
    def test_recovery_controller_initialization(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model)
        assert ctrl.target_yaw == 0.0
        assert ctrl.target_pelvis_x == -1.800
        assert ctrl.kp_yaw == 4.0
        assert ctrl.ki_yaw == 1.0
        assert ctrl.torso_body_id is not None
        assert ctrl.pelvis_body_id is not None

    def test_inactive_outside_recovery_window(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model, recovery_start=1.05, recovery_end=2.95)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ref_orig = ref.copy()

        # Before recovery window (t=0.5s)
        ctrl.apply_recovery(data, ref, feed_elapsed=0.50)
        np.testing.assert_array_equal(ref, ref_orig)

        # After recovery window (t=3.0s)
        ctrl.apply_recovery(data, ref, feed_elapsed=3.00)
        np.testing.assert_array_equal(ref, ref_orig)

    def test_active_inside_recovery_window_yaw(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model, recovery_start=1.05, recovery_end=2.95)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        # Introduce yaw error: rotate pelvis around z by -5 degrees (-0.087 rad)
        angle = np.radians(-5.0)
        data.qpos[3:7] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=1.50)

        # Yaw angular velocity in torso body frame (ref[6] for upright robot) should command positive counter-rotation
        assert ref[6] > 0.1
        assert ref[6] <= ctrl.max_yaw_rate

    def test_active_inside_recovery_window_linear(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model, kp_x=3.5, recovery_start=1.05, recovery_end=2.95)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        # Move 3 cm rearward of target
        data.qpos[0] = -1.830
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=2.00)  # midpoint of window where Hann weight is high

        # Linear forward velocity (ref[1]) should command positive forward motion
        assert ref[1] > 0.05
        assert ref[1] <= ctrl.max_lin_vel

    def test_angle_wrapping_across_pi(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model, target_yaw=3.10)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)

        # Set torso near -3.10 rad (difference is 0.08 rad across -pi/pi boundary)
        angle = -3.10
        data.qpos[3:7] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=1.50)

        # Should command a small, bounded rate across the wrap boundary, not a huge rate across 2pi
        assert abs(ref[6]) < 0.6

    def test_torso_tilted_projection(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)

        # Pitch robot forward by 30 degrees and yaw by -5 degrees
        pitch = np.radians(30.0)
        yaw = np.radians(-5.0)
        # Pitch around Y, then yaw around Z
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        data.qpos[3:7] = [cy * cp, -sy * sp, cy * sp, sy * cp]
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=1.50)

        # For a pitched torso, world Z-rotation projects onto both body X and Z axes (ref[4:7])
        R_torso = data.xmat[ctrl.torso_body_id].reshape(3, 3)
        omega_body = ref[4:7]
        omega_world_reconstructed = R_torso @ omega_body
        # World X and Y angular velocities should be near zero, while world Z is positive
        assert abs(omega_world_reconstructed[0]) < 1e-4
        assert abs(omega_world_reconstructed[1]) < 1e-4
        assert omega_world_reconstructed[2] > 0.1

    def test_reset_feed_clears_integral(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(model=scene_model)
        ctrl.yaw_err_integral = 0.45
        ctrl.reset_feed(1)
        assert ctrl.yaw_err_integral == 0.0
        assert ctrl.feed_id == 1

    def test_run_continuous_stance_recovery_mode(self):
        from g1_ping_pong_lab.continuous_feeds import run_continuous
        report, frames = run_continuous(
            "assets/scene.xml", "assets/wbc",
            feeds=2, interval=2.0, tail=1.2,
            controller_mode="stance-recovery"
        )
        assert len(report["feeds"]) == 2
        assert report["full_session_stance_accepted"] is True

    def test_run_continuous_with_diagnostics(self, scene_model):
        from g1_ping_pong_lab.continuous_feeds import run_continuous
        tracker = StanceTracker(scene_model)
        report, _ = run_continuous(
            "assets/scene.xml", "assets/wbc",
            feeds=2, interval=2.0, tail=1.2,
            controller_mode="stance-recovery",
            diagnostics=tracker
        )
        assert "stance_diagnostics" in report
        assert len(report["stance_diagnostics"]) == 2
        assert len(tracker.summaries) == 2
        assert report["stance_diagnostics"][0]["feed_id"] == 0
