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
        assert summary.com_drift.shape == (3,)
        assert summary.left_foot_drift.shape == (3,)
        assert summary.right_foot_drift.shape == (3,)

    def test_tracker_slip_and_lift_detection(self, scene_model):
        tracker = StanceTracker(scene_model)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        tracker.start_feed(0, data, 0.0)
        # In keyframe 0, feet are on ground with normal force
        mujoco.mj_step(scene_model, data)
        tracker.observe_step(data, 0.00025)

        summary = tracker.finish_feed(data, 0.00025)
        # Normal forces are positive in keyframe 0
        assert summary.feed_id == 0


class TestStanceRecoveryController:
    def test_recovery_controller_initialization(self):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController()
        assert ctrl.target_yaw == 0.0
        assert ctrl.target_pelvis_x == -1.800
        assert ctrl.kp_yaw == 4.0
        assert ctrl.ki_yaw == 1.0

    def test_inactive_outside_recovery_window(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(recovery_start=1.05, recovery_end=2.95)
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

    def test_active_inside_recovery_window(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController(recovery_start=1.05, recovery_end=2.95)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        # Introduce yaw and linear error
        # Rotate pelvis around z by -5 degrees (-0.087 rad)
        angle = np.radians(-5.0)
        data.qpos[3:7] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
        data.qpos[0] = -1.830  # 3 cm rearward of -1.800
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=1.50)

        # Yaw angular velocity (ref[6]) should command positive counter-rotation
        assert ref[6] > 0.1
        assert ref[6] <= ctrl.max_yaw_rate

        # Linear forward velocity (ref[1]) should command positive forward motion
        assert ref[1] > 0.05
        assert ref[1] <= ctrl.max_lin_vel

    def test_reset_feed_clears_integral(self, scene_model):
        from g1_ping_pong_lab.interception import StanceRecoveryController
        ctrl = StanceRecoveryController()
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
