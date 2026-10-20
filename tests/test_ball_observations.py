"""Sensor-channel contracts: causality, capture-time noise and feed isolation."""
import importlib

import numpy as np
import pytest


def feature():
    try:
        return importlib.import_module("g1_ping_pong_lab.ball_observations")
    except ModuleNotFoundError:
        pytest.fail("The causal ball-observation channel is not implemented")


def stream(**kwargs):
    module = feature()
    result = module.BallObservationStream(module.ObservationConfig(**kwargs))
    result.reset_feed(0)
    return result


def test_delayed_channel_never_exposes_a_future_sample():
    channel = stream(latency_s=.04)
    assert channel.observe(0., [1., 0., 1.], [-2., 0., 0.]) is None
    assert channel.observe(.02, [2., 0., 1.], [-2., 0., 0.]) is None
    sample = channel.observe(.04, [3., 0., 1.], [-2., 0., 0.])
    assert sample.captured_at == 0.
    np.testing.assert_array_equal(sample.position, [1., 0., 1.])
    sample = channel.observe(.06, [4., 0., 1.], [-2., 0., 0.])
    assert sample.captured_at == .02
    np.testing.assert_array_equal(sample.position, [2., 0., 1.])


def test_nonintegral_latency_reports_actual_sample_age():
    channel = stream(latency_s=.01)
    assert channel.observe(0., [0., 0., 1.], [-2., 0., 0.]) is None
    sample = channel.observe(.02, [1., 0., 1.], [-2., 0., 0.])
    assert sample.captured_at == 0.
    assert channel.trace[-1]["age_seconds"] == .02


def test_zero_noise_and_delay_preserve_inputs_without_aliasing():
    channel = stream()
    position = np.array([-.65, -.25, 1.])
    velocity = np.array([-1.8, 0., .5])
    sample = channel.observe(0., position, velocity)
    np.testing.assert_array_equal(sample.position, position)
    np.testing.assert_array_equal(sample.velocity, velocity)
    position[0] = 123.
    assert sample.position[0] == -.65
    np.testing.assert_array_equal(velocity, [-1.8, 0., .5])


def test_noise_is_repeatable_and_sampled_at_capture_not_delivery():
    now = stream(position_std_m=.005, velocity_std_mps=.05, seed=7)
    delayed = stream(position_std_m=.005, velocity_std_mps=.05, seed=7, latency_s=.04)
    first = now.observe(0., [0., 0., 1.], [-2., 0., 0.])
    assert delayed.observe(0., [0., 0., 1.], [-2., 0., 0.]) is None
    assert delayed.observe(.02, [0., 0., 1.], [-2., 0., 0.]) is None
    delivered = delayed.observe(.04, [0., 0., 1.], [-2., 0., 0.])
    np.testing.assert_array_equal(delivered.position, first.position)
    np.testing.assert_array_equal(delivered.velocity, first.velocity)
    other = stream(position_std_m=.005, velocity_std_mps=.05, seed=8)
    assert not np.array_equal(other.observe(0., [0., 0., 1.], [-2., 0., 0.]).position, first.position)


def test_feed_reset_discards_old_ball_measurements():
    channel = stream(latency_s=.04)
    channel.observe(0., [100., 0., 1.], [-2., 0., 0.])
    channel.observe(.04, [200., 0., 1.], [-2., 0., 0.])
    channel.reset_feed(1)
    assert channel.observe(0., [300., 0., 1.], [-2., 0., 0.]) is None
    assert channel.trace[-1]["feed_id"] == 1


@pytest.mark.parametrize("kwargs", [
    {"latency_s": -1.}, {"position_std_m": float("nan")},
    {"velocity_std_mps": -.01}, {"latency_s": float("inf")}, {"seed": -1},
])
def test_invalid_noise_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        stream(**kwargs)


def test_nonmonotonic_timestamps_and_nonfinite_states_are_rejected():
    channel = stream()
    channel.observe(.02, [0., 0., 1.], [-2., 0., 0.])
    with pytest.raises(ValueError):
        channel.observe(.01, [0., 0., 1.], [-2., 0., 0.])
    with pytest.raises(ValueError):
        channel.observe(.04, [float("nan"), 0., 1.], [-2., 0., 0.])
