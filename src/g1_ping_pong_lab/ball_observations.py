"""Causal 50 Hz synthetic ball measurements for simulation experiments.

Noise is independent Gaussian noise per axis, applied once at capture. The
channel delays measurements, never the physical ball or WBC observations.
"""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationConfig:
    position_std_m: float = 0.
    velocity_std_mps: float = 0.
    latency_s: float = 0.
    seed: int = 0
    timestamp_aware: bool = False

    def __post_init__(self):
        for value in (self.position_std_m, self.velocity_std_mps, self.latency_s):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Noise standard deviations and latency must be finite and nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if not isinstance(self.timestamp_aware, bool):
            raise ValueError("timestamp_aware must be boolean")


@dataclass(frozen=True)
class BallObservation:
    captured_at: float
    position: np.ndarray
    velocity: np.ndarray


class BallObservationStream:
    def __init__(self, config: ObservationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.trace = []
        self.reset_feed(0)

    def reset_feed(self, feed_id):
        """A replacement ball cannot inherit measurements of the preceding ball."""
        self.feed_id = feed_id
        self.pending = deque()
        self.latest = None
        self.previous_time = None

    def observe(self, time, position, velocity):
        """Capture now and deliver the newest sample whose delay has elapsed."""
        time = float(time)
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if (not np.isfinite(time) or time < 0
                or (self.previous_time is not None and time <= self.previous_time)
                or position.shape != (3,) or velocity.shape != (3,)
                or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
            raise ValueError("Finite 3D states and strictly increasing nonnegative feed times required")
        self.previous_time = time
        # Draw the same six random variates in every scenario for paired seeds.
        noise = self.rng.standard_normal(6)
        measured_position = position + self.config.position_std_m * noise[:3]
        measured_velocity = velocity + self.config.velocity_std_mps * noise[3:]
        measured_position.setflags(write=False)
        measured_velocity.setflags(write=False)
        self.pending.append(BallObservation(time, measured_position, measured_velocity))
        # Numerical tolerance is one nanosecond, much smaller than a physics step.
        cutoff = time - self.config.latency_s + 1.e-9
        while self.pending and self.pending[0].captured_at <= cutoff:
            self.latest = self.pending.popleft()
        sample = self.latest
        self.trace.append({
            "feed_id": self.feed_id, "time": time,
            "captured_at": None if sample is None else sample.captured_at,
            "age_seconds": None if sample is None else max(0., time - sample.captured_at),
            "truth_position": position.tolist(), "truth_velocity": velocity.tolist(),
            "position": None if sample is None else sample.position.tolist(),
            "velocity": None if sample is None else sample.velocity.tolist(),
            "phase": None, "predicted_impact_time": None, "replanned": False,
        })
        return sample

    def record_plan(self, phase, predicted_impact_time=None, *, replanned=False):
        self.trace[-1].update(phase=float(phase),
                             predicted_impact_time=None if predicted_impact_time is None
                             else float(predicted_impact_time), replanned=bool(replanned))

    def summary(self):
        ages = [row["age_seconds"] for row in self.trace if row["age_seconds"] is not None]
        reversals = sum(b["phase"] < a["phase"] - 1.e-9
                        for a, b in zip(self.trace, self.trace[1:])
                        if a["feed_id"] == b["feed_id"]
                        and a["phase"] is not None and b["phase"] is not None)
        return {"config": asdict(self.config), "capture_hz": 50,
                "samples_captured": len(self.trace),
                "ticks_without_measurement": sum(r["captured_at"] is None for r in self.trace),
                "mean_age_seconds": float(np.mean(ages)) if ages else None,
                "maximum_age_seconds": max(ages) if ages else None,
                "phase_reversals": reversals}
