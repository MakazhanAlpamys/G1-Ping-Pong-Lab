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
        assert summary.max_lf_tangential_force >= 0.0
        assert summary.max_rf_tangential_force >= 0.0
