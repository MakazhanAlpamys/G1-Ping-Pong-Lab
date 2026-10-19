"""Unit tests for the predictive ball-flight timing subsystem."""
import mujoco
import numpy as np
import pytest

from g1_ping_pong_lab.interception import (
    BallTrajectoryPredictor,
    DescendingInterceptionPlanner,
    PredictiveTimingInterceptor,
)
from g1_ping_pong_lab.runtime import Controller, stroke_at


@pytest.fixture(scope="module")
def scene_and_controller():
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    ctrl = Controller(model, "assets/wbc")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, ctrl, data


class TestBallTrajectoryPredictor:
    def test_nominal_feed_prediction_accuracy(self):
        predictor = BallTrajectoryPredictor()
        pos = np.array([-0.65, -0.25, 1.0])
        vel = np.array([-1.8, 0.0, 0.5])

        pred = predictor.predict(pos, vel, current_time=0.0)

        # Ground truth values from MuJoCo 4000 Hz simulation
        # Bounce: t=0.2690 s, x=-1.1342 m
        # Impact: t=0.4890 s, x=-1.4716 m, y=-0.2500 m, z=0.8535 m
        assert pred.bounce_time is not None
        assert abs(pred.bounce_time - 0.2690) < 0.002
        assert abs(pred.bounce_pos[0] - (-1.1342)) < 0.005
        assert abs(pred.impact_time - 0.4890) < 0.002
        assert abs(pred.impact_pos[0] - (-1.47158)) < 1e-4
        assert abs(pred.impact_pos[1] - (-0.25)) < 1e-4
        assert abs(pred.impact_pos[2] - 0.8535) < 0.005
        assert not pred.is_post_bounce

    def test_post_bounce_continuity(self):
        predictor = BallTrajectoryPredictor()
        # Initial feed
        p0 = np.array([-0.65, -0.25, 1.0])
        v0 = np.array([-1.8, 0.0, 0.5])
        pred_pre = predictor.predict(p0, v0, current_time=0.0)

        # State after bounce at t = 0.35 s
        dt_post = 0.35 - pred_pre.bounce_time
        v_post = np.array([
            v0[0] * predictor.tangential_restitution,
            0.0,
            -predictor.restitution_z * (v0[2] - 9.81 * pred_pre.bounce_time),
        ])
        p_post = np.array([
            pred_pre.bounce_pos[0] + v_post[0] * dt_post,
            -0.25,
            predictor.z_contact + v_post[2] * dt_post - 0.5 * 9.81 * dt_post**2,
        ])
        v_curr = np.array([v_post[0], 0.0, v_post[2] - 9.81 * dt_post])

        pred_post = predictor.predict(p_post, v_curr, current_time=0.35)
        assert pred_post.is_post_bounce
        assert abs(pred_post.impact_time - pred_pre.impact_time) < 0.001
        assert np.allclose(pred_post.impact_pos, pred_pre.impact_pos, atol=1e-3)

    def test_ball_moving_away(self):
        predictor = BallTrajectoryPredictor()
        pos = np.array([-0.65, -0.25, 1.0])
        vel = np.array([1.8, 0.0, 0.5])  # Moving in +x away from robot
        pred = predictor.predict(pos, vel, current_time=0.0)
        assert pred.impact_time > 5.0

    def test_zero_x_velocity_stationary(self):
        predictor = BallTrajectoryPredictor()
        pos = np.array([-0.65, -0.25, 1.0])
        vel = np.array([0.0, 0.0, 0.0])
        pred = predictor.predict(pos, vel, current_time=0.0)
        assert pred.impact_time >= 10.0
        assert np.array_equal(pred.impact_pos, pos)

    def test_ball_already_past_target(self):
        predictor = BallTrajectoryPredictor()
        pos = np.array([-1.50, -0.25, 0.85])  # already past target_x = -1.47158
        vel = np.array([-1.8, 0.0, 0.5])
        pred = predictor.predict(pos, vel, current_time=0.5)
        # dt should clamp to 0, returning immediate time
        assert abs(pred.impact_time - 0.5) < 1e-5


class TestDescendingInterceptionPlanner:
    def test_planning_time_alignment(self):
        planner = DescendingInterceptionPlanner(nominal_impact_time=0.4890)
        p = np.array([-0.65, -0.25, 1.0])

        # Faster ball: arrival earlier
        v_fast = np.array([-2.07, 0.0, 0.5])  # +15% speed
        plan_fast = planner.plan(p, v_fast, feed_elapsed=0.0)
        assert plan_fast.target_time < 0.4890
        assert plan_fast.time_offset < 0.0

        # Slower ball: arrival later
        v_slow = np.array([-1.53, 0.0, 0.5])  # -15% speed
        plan_slow = planner.plan(p, v_slow, feed_elapsed=0.0)
        assert plan_slow.target_time > 0.4890
        assert plan_slow.time_offset > 0.0

    def test_effective_stroke_time_clamped_at_zero(self):
        planner = DescendingInterceptionPlanner(nominal_impact_time=0.4890)
        p = np.array([-0.65, -0.25, 1.0])
        v_slow = np.array([-1.53, 0.0, 0.5])  # arrives ~0.56s (time_offset ~ +0.07s)
        plan = planner.plan(p, v_slow, feed_elapsed=0.0)

        # Before time_offset, effective time is clamped to 0
        assert planner.effective_stroke_time(0.0, plan) == 0.0
        assert planner.effective_stroke_time(0.02, plan) == 0.0
        assert planner.effective_stroke_time(0.10, plan) > 0.0


class TestPredictiveTimingInterceptor:
    def test_scripted_mode_exact_match(self, scene_and_controller):
        model, ctrl, data = scene_and_controller
        interceptor = PredictiveTimingInterceptor(model, ctrl, mode="scripted")
        interceptor.reset_feed(0, data)

        for t in [0.0, 0.1, 0.25, 0.35, 0.489, 0.6, 0.8]:
            out = interceptor.step(data, t)
            expected = stroke_at(t)
            assert np.array_equal(out, expected)

    def test_predictive_timing_mode_bounded(self, scene_and_controller):
        model, ctrl, data = scene_and_controller
        interceptor = PredictiveTimingInterceptor(model, ctrl, mode="predictive-timing")
        interceptor.reset_feed(0, data)

        for t in np.linspace(0.0, 1.0, 21):
            out = interceptor.step(data, float(t))
            assert out.shape == (7,)
            assert np.all(out >= -1.0) and np.all(out <= 1.0)
            assert np.all(np.isfinite(out))

    def test_plan_latching_monotonicity(self, scene_and_controller):
        """Verify that post-impact ball rebound does not snap effective stroke time back to zero."""
        model, ctrl, data = scene_and_controller
        interceptor = PredictiveTimingInterceptor(model, ctrl, mode="predictive-timing")
        interceptor.reset_feed(0, data)

        ball_q = interceptor.ball_q
        ball_v = interceptor.ball_v

        # Ball approaching: feed_elapsed = 0.15s
        data.qpos[ball_q : ball_q + 3] = [-1.0, -0.25, 0.85]
        data.qvel[ball_v : ball_v + 3] = [-1.8, 0.0, 0.2]
        interceptor.step(data, 0.15)
        tau_pre = interceptor.planner.effective_stroke_time(0.15, interceptor.current_plan)

        # After impact: feed_elapsed = 0.50s, ball rebounding with vx > 0
        data.qpos[ball_q : ball_q + 3] = [-1.2, -0.25, 0.90]
        data.qvel[ball_v : ball_v + 3] = [5.0, 0.0, 0.5]
        interceptor.step(data, 0.50)
        tau_post = interceptor.planner.effective_stroke_time(0.50, interceptor.current_plan)

        assert tau_post > tau_pre, f"Stroke phase reversed! tau_pre={tau_pre}, tau_post={tau_post}"
        assert abs((tau_post - tau_pre) - (0.50 - 0.15)) < 1e-4, f"Plan latch failed: delta_tau={tau_post - tau_pre}"

    def test_tau_monotonic_progression(self, scene_and_controller):
        """Verify that effective stroke time tau advances monotonically throughout a full stroke."""
        model, ctrl, data = scene_and_controller
        interceptor = PredictiveTimingInterceptor(model, ctrl, mode="predictive-timing")
        interceptor.reset_feed(0, data)

        tau_prev = -1.0
        for t in np.linspace(0.0, 1.2, 61):
            interceptor.step(data, float(t))
            tau = interceptor.planner.effective_stroke_time(float(t), interceptor.current_plan)
            assert tau >= tau_prev, f"Tau decreased from {tau_prev} to {tau} at t={t}"
            tau_prev = tau

    def test_infeasible_feed_safe_recovery(self, scene_and_controller):
        """Verify that an infeasible feed (e.g. ball moving away) does not produce NaN or break WBC."""
        model, ctrl, data = scene_and_controller
        interceptor = PredictiveTimingInterceptor(model, ctrl, mode="predictive-timing")
        interceptor.reset_feed(0, data)

        # Set ball moving away
        ball_q = interceptor.ball_q
        ball_v = interceptor.ball_v
        data.qpos[ball_q : ball_q + 3] = [-0.65, -0.25, 1.0]
        data.qvel[ball_v : ball_v + 3] = [2.0, 0.0, 0.5]  # moving in +x away

        out = interceptor.step(data, 0.1)
        assert out.shape == (7,)
        assert np.all(np.isfinite(out))
        assert np.all(np.abs(out) <= 1.0)
