"""Unit and integration tests for stance tracking, drift diagnostics, and closed-loop stance recovery."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from g1_ping_pong_lab.continuous_feeds import run_continuous
from g1_ping_pong_lab.diagnostics import StanceTracker, FeedStanceSummary
from g1_ping_pong_lab.interception import StanceRecoveryController
from g1_ping_pong_lab.scoring import ReturnAttempt, TableGeometry


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

    def test_tracker_bind_model_lifecycle(self, scene_model):
        """Verify that tracker can re-bind safely to an active model instance."""
        tracker = StanceTracker(scene_model)
        fresh_model = mujoco.MjModel.from_xml_path("assets/scene.xml")
        tracker.bind_model(fresh_model)
        assert tracker.model is fresh_model
        assert tracker.pelvis_id == fresh_model.body("robot/pelvis").id
        assert tracker.torso_id == fresh_model.body("robot/torso_link").id

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

        # Settle robot onto ground from initial keyframe hover (contacts establish around step 100)
        for _ in range(120):
            mujoco.mj_step(scene_model, data)

        # Now start tracking during standing stance
        tracker.start_feed(0, data, data.time)
        for _ in range(100):
            mujoco.mj_step(scene_model, data)
            tracker.observe_step(data, data.time)

        summary = tracker.finish_feed(data, data.time)
        assert summary.feed_id == 0
        assert summary.max_lf_tangential_force >= 0.0
        assert summary.max_rf_tangential_force >= 0.0
        assert summary.left_foot_loaded_displacement_cm >= 0.0
        assert summary.right_foot_loaded_displacement_cm >= 0.0
        # While standing firmly planted, foot_lifted is False
        assert summary.left_foot_lifted is False
        assert summary.right_foot_lifted is False
        assert summary.min_lf_normal_force > 10.0
        assert summary.min_rf_normal_force > 10.0

    def test_phase_sampling_deterministic_capture(self, scene_model):
        tracker = StanceTracker(scene_model)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        rec = tracker.sample_phase(data, "pre_stroke", 0, 0.0, target_yaw=0.0)
        assert rec["feed_id"] == 0
        assert rec["phase"] == "pre_stroke"
        assert len(rec["pelvis_pos"]) == 3
        assert len(rec["torso_pos"]) == 3
        assert abs(rec["pelvis_yaw_deg"]) < 1.0


class TestScoringSemantics:
    def test_valid_returns_counted_independent_of_session_stance_acceptance(self):
        """Verify that valid returns are preserved even when the session safety check fails."""
        report, _ = run_continuous(
            "assets/scene.xml", "assets/wbc",
            feeds=2, interval=2.0, tail=1.2,
            controller_mode="scripted"
        )
        assert "valid_returns" in report
        assert "racket_contacts" in report
        assert "return_rate" in report
        assert "contact_rate" in report
        assert "full_session_stance_accepted" in report
        assert report["return_rate"] == report["valid_returns"] / 2.0
        assert report["contact_rate"] == report["racket_contacts"] / 2.0
        # If session passes, accepted_returns equals valid_returns
        if report["full_session_stance_accepted"]:
            assert report["accepted_returns"] == report["valid_returns"]

    def test_racket_contact_without_valid_return_is_distinct(self):
        """Verify that a racket contact without valid table bounce does not count as a valid return."""
        geom = TableGeometry(
            net_x=0.0, table_x_min=-1.37, table_x_max=1.37,
            table_y_min=-0.762, table_y_max=0.762,
            net_top_z=0.9125, ball_radius=0.02
        )
        attempt = ReturnAttempt(geom)
        # Ball bounces on own side, then hits racket, but goes out of bounds
        attempt.observe(time=0.1, ball_position=np.array([-0.5, 0.0, 0.8]), contacts={"table"}, fallen=False)
        attempt.observe(time=0.3, ball_position=np.array([-0.8, 0.0, 0.8]), contacts={"racket"}, fallen=False)
        attempt.observe(time=0.6, ball_position=np.array([-1.5, 0.0, 0.0]), contacts={"floor"}, fallen=False)
        result = attempt.finish()

        assert result["contacted"] is True
        assert result["success"] is False
        assert result["reason"] == "floor_contact"


class TestStanceRecoveryController:
    def test_recovery_controller_initialization(self, scene_model):
        ctrl = StanceRecoveryController(model=scene_model)
        assert ctrl.target_yaw == 0.0
        assert ctrl.target_pelvis_x == -1.800
        assert ctrl.kp_yaw == 4.0
        assert ctrl.ki_yaw == 1.0
        assert ctrl.torso_body_id is not None
        assert ctrl.pelvis_body_id is not None

    def test_inactive_outside_recovery_window(self, scene_model):
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
        ctrl = StanceRecoveryController(model=scene_model, kp_x=3.5, recovery_start=1.05, recovery_end=2.95)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)
        mujoco.mj_forward(scene_model, data)

        # Move 3 cm rearward of target
        data.qpos[0] = -1.830
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=2.00)

        # Linear forward velocity (ref[1]) should command positive forward motion
        assert ref[1] > 0.05
        assert ref[1] <= ctrl.max_lin_vel

    def test_angle_wrapping_across_pi(self, scene_model):
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
        ctrl = StanceRecoveryController(model=scene_model)
        data = mujoco.MjData(scene_model)
        mujoco.mj_resetDataKeyframe(scene_model, data, 0)

        pitch = np.radians(30.0)
        yaw = np.radians(-5.0)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        data.qpos[3:7] = [cy * cp, -sy * sp, cy * sp, sy * cp]
        mujoco.mj_forward(scene_model, data)

        ref = np.zeros(39, dtype=np.float32)
        ctrl.apply_recovery(data, ref, feed_elapsed=1.50)

        R_torso = data.xmat[ctrl.torso_body_id].reshape(3, 3)
        omega_body = ref[4:7]
        omega_world_reconstructed = R_torso @ omega_body
        assert abs(omega_world_reconstructed[0]) < 1e-4
        assert abs(omega_world_reconstructed[1]) < 1e-4
        assert omega_world_reconstructed[2] > 0.1

    def test_reset_feed_clears_integral(self, scene_model):
        ctrl = StanceRecoveryController(model=scene_model)
        ctrl.yaw_err_integral = 0.45
        ctrl.reset_feed(1)
        assert ctrl.yaw_err_integral == 0.0
        assert ctrl.feed_id == 1


class TestClosedLoopPhysicalRecovery:
    def test_closed_loop_yaw_recovery_reduces_physical_error(self):
        """Simulate closed-loop recovery and verify physical heading error is actively reduced."""
        # Run 1 feed with stance recovery and diagnostics
        report_rec, _ = run_continuous(
            "assets/scene.xml", "assets/wbc",
            feeds=1, interval=3.0, tail=3.0,
            controller_mode="stance-recovery",
            diagnostics=True
        )
        assert "recovery_metrics" in report_rec
        assert len(report_rec["recovery_metrics"]) == 1
        m = report_rec["recovery_metrics"][0]
        # In stance recovery mode, absolute yaw error decreases between entry (t=1.05s) and exit (t=2.95s)
        assert m["absolute_yaw_error_reduction_deg"] > 0.0
        assert abs(m["recovery_exit_yaw_err_deg"]) < abs(m["recovery_entry_yaw_err_deg"])

    def test_closed_loop_baseline_has_no_recovery_reduction(self):
        """In predictive-timing alone (no stance recovery), heading error does not systematically reduce during idle window."""
        report_base, _ = run_continuous(
            "assets/scene.xml", "assets/wbc",
            feeds=1, interval=3.0, tail=3.0,
            controller_mode="predictive-timing",
            diagnostics=True
        )
        m_base = report_base["recovery_metrics"][0]
        # In baseline, without active recovery, error reduction is near zero or negative
        assert m_base["absolute_yaw_error_reduction_deg"] < 0.2

    def test_scripted_baseline_compatibility(self):
        """Empirically verify default scripted execution matches baseline criteria."""
        report, _ = run_continuous(
            "assets/scene.xml", "assets/wbc",
            feeds=2, interval=3.0, tail=3.0,
            controller_mode="scripted"
        )
        assert report["full_session_stance_accepted"] is True
        assert report["valid_returns"] == 2
        assert report["maximum_force_cap_ratio"] <= 1.000001
        assert report["fallen"] is False
