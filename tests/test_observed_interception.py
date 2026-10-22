import importlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from g1_ping_pong_lab.ball_observations import BallObservationStream, ObservationConfig
from g1_ping_pong_lab.continuous_feeds import run_continuous
from g1_ping_pong_lab.runtime import Controller

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    try:
        return importlib.import_module("g1_ping_pong_lab.observed_interception").ObservedTimingInterceptor
    except ModuleNotFoundError:
        pytest.fail("The observed-state timing adapter is not implemented")


@pytest.fixture(scope="module")
def native():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scene.xml"))
    return model, Controller(model, ROOT / "assets/wbc")


def interceptor(native, *, timestamp_aware=False):
    model, controller = native
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    bq = int(model.joint("ball/ball_freejoint").qposadr[0])
    bv = int(model.joint("ball/ball_freejoint").dofadr[0])
    data.qpos[bq:bq+3] = [-.65, -.25, 1.]
    data.qvel[bv:bv+3] = [-1.8, 0., .5]
    sensor = BallObservationStream(ObservationConfig(latency_s=.04, timestamp_aware=timestamp_aware))
    result = adapter()(model, controller, observation_stream=sensor)
    result.reset_feed(0, data)
    return result, data


def test_startup_does_not_leak_perfect_ball_state(native):
    observed, data = interceptor(native)
    assert observed.current_plan is None
    before = data.qpos.copy(), data.qvel.copy()
    np.testing.assert_array_equal(observed.step(data, 0.), np.zeros(7))
    assert observed.current_plan is None
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])


@pytest.mark.parametrize("aware,expected", [(False, .52902717094), (True, .48902717094)])
def test_capture_timestamp_controls_whether_delay_bias_is_compensated(native, aware, expected):
    observed, data = interceptor(native, timestamp_aware=aware)
    for time in [0., .02, .04]:
        observed.step(data, time)
    assert observed.current_plan.target_time == pytest.approx(expected, abs=1.e-8)
    assert observed.observation_stream.trace[-1]["captured_at"] == 0.


def test_sensor_report_counts_phase_reversals_without_counting_feed_resets():
    sensor = BallObservationStream(ObservationConfig())
    assert hasattr(sensor, "record_plan"), "Plan telemetry is missing"
    assert hasattr(sensor, "summary"), "Sensor metrics are missing"
    for time, phase in [(0., 0.), (.02, .1), (.04, .05)]:
        sensor.observe(time, [0., 0., 1.], [-2., 0., 0.])
        sensor.record_plan(phase)
    sensor.reset_feed(1)
    sensor.observe(0., [0., 0., 1.], [-2., 0., 0.])
    sensor.record_plan(0.)
    assert sensor.summary()["phase_reversals"] == 1
    assert sensor.summary()["samples_captured"] == 4


def test_phase_reversal_magnitude_ignores_normal_feed_clock_reset():
    sensor = BallObservationStream(ObservationConfig())
    for time, phase in [(0., .1), (.02, .08), (.04, .07)]:
        sensor.observe(time, [0., 0., 1.], [-2., 0., 0.])
        sensor.record_plan(phase)
    sensor.reset_feed(1)
    sensor.observe(0., [0., 0., 1.], [-2., 0., 0.])
    sensor.record_plan(0.)
    assert sensor.summary().get("largest_phase_reversal_seconds") == pytest.approx(.02)


@pytest.mark.parametrize("mode", ["scripted", "predictive-timing"])
def test_clean_sensor_preserves_native_trajectory(mode):
    import inspect
    assert "observation_config" in inspect.signature(run_continuous).parameters, "Sensor hook is missing"
    kwargs = dict(feeds=1, tail=1.2, controller_mode=mode)
    _, exact = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc", **kwargs)
    report, measured = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc",
                                     observation_config=ObservationConfig(), **kwargs)
    assert report["accepted_returns"] == 1
    for a, b in zip(exact, measured):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_scripted_arm_is_independent_of_noisy_delayed_ball_measurements():
    import inspect
    assert "observation_config" in inspect.signature(run_continuous).parameters, "Sensor hook is missing"
    kwargs = dict(feeds=1, tail=1.2)
    _, exact = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc", **kwargs)
    report, measured = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc",
        observation_config=ObservationConfig(position_std_m=.1, velocity_std_mps=1., latency_s=.06), **kwargs)
    assert report["ball_observations"]["ticks_without_measurement"] == 3
    for a, b in zip(exact, measured):
        np.testing.assert_array_equal(a["qpos"], b["qpos"])


@pytest.mark.parametrize("mode,recovery_flag", [
    ("stance-recovery", False),
    ("predictive-timing-with-recovery", False),
    ("predictive-timing", True),
])
def test_clean_sensor_preserves_stance_recovery_trajectory(mode, recovery_flag):
    """Recovery must remain active when timing consumes measured ball states."""
    kwargs = dict(feeds=1, tail=3., controller_mode=mode, stance_recovery=recovery_flag)
    _, exact = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc", **kwargs)
    report, measured = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc",
                                     observation_config=ObservationConfig(), **kwargs)
    assert report["accepted_returns"] == 1
    assert report["ball_observations"]["samples_captured"] == 150
    assert len(exact) == len(measured) == 150
    assert any(np.linalg.norm(f["reference39"][4:7]) > .01 for f in measured)
    for a, b in zip(exact, measured):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_recovery_mode_keeps_delayed_startup_causal():
    report, frames = run_continuous(ROOT / "assets/scene.xml", ROOT / "assets/wbc",
        feeds=1, tail=3., controller_mode="stance-recovery", diagnostics=True,
        observation_config=ObservationConfig(latency_s=.08, timestamp_aware=True))
    assert report["full_session_stance_accepted"] is True
    assert report["ball_observations"]["ticks_without_measurement"] == 4
    assert len(report["recovery_metrics"]) == 1
    assert all(row["captured_at"] is None for row in report["observation_trace"][:4])
    for frame in frames[:4]:
        np.testing.assert_array_equal(frame["raw7"], np.zeros(7))
    assert any(np.linalg.norm(f["reference39"][4:7]) > .01 for f in frames)
