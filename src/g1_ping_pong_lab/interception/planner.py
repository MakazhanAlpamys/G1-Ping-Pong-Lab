"""Descending interception planner for humanoid table tennis.

Translates predicted ball interception conditions into phase-aligned
stroke execution timing.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .predictor import BallTrajectoryPredictor, PredictionResult


@dataclass
class InterceptionPlan:
    """Interception task plan."""
    target_time: float
    time_offset: float
    target_position: np.ndarray
    prediction: PredictionResult


class DescendingInterceptionPlanner:
    """Computes task-space interception parameters from ball kinematics."""

    def __init__(
        self,
        predictor: BallTrajectoryPredictor | None = None,
        *,
        nominal_impact_time: float = 0.4890,
        nominal_impact_pos: tuple[float, float, float] = (-1.47158, -0.2500, 0.85346),
    ):
        self.predictor = predictor if predictor is not None else BallTrajectoryPredictor()
        self.nom_t = float(nominal_impact_time)
        self.nom_pos = np.asarray(nominal_impact_pos, dtype=np.float64)

    def plan(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        feed_elapsed: float,
        target_x: float = -1.47158,
    ) -> InterceptionPlan:
        """Generate interception plan for current feed.

        Args:
            ball_pos: current ball position [x, y, z].
            ball_vel: current ball velocity [vx, vy, vz].
            feed_elapsed: elapsed time since feed launch in seconds.
            target_x: interception plane x coordinate.

        Returns:
            InterceptionPlan with time offset and target position.
        """
        pred = self.predictor.predict(ball_pos, ball_vel, current_time=feed_elapsed, target_x=target_x)
        time_offset = pred.impact_time - self.nom_t

        return InterceptionPlan(
            target_time=pred.impact_time,
            time_offset=time_offset,
            target_position=pred.impact_pos,
            prediction=pred,
        )

    def effective_stroke_time(self, feed_elapsed: float, plan: InterceptionPlan) -> float:
        """Compute the phase-aligned time to feed into stroke_at().

        Args:
            feed_elapsed: elapsed time since feed launch in seconds.
            plan: active InterceptionPlan.

        Returns:
            Phase-adjusted time tau for stroke generation.
        """
        # tau aligns peak forward stroke with predicted impact time
        tau = feed_elapsed - plan.time_offset
        return max(0.0, tau)
