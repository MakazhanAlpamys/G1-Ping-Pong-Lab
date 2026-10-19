"""Predictive timing interceptor for humanoid table tennis.

Modulates the temporal execution phase of the open-loop CEM arm stroke
based on analytical ball trajectory and table bounce prediction.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..runtime import stroke_at
from .planner import DescendingInterceptionPlanner, InterceptionPlan
from .predictor import BallTrajectoryPredictor


class PredictiveTimingInterceptor:
    """Predictive impact timing controller for humanoid table tennis."""

    def __init__(
        self,
        model: mujoco.MjModel,
        controller,
        *,
        mode: str = "predictive-timing",
        target_x: float = -1.47158,
    ):
        self.model = model
        self.controller = controller
        self.mode = mode.lower()
        if self.mode not in ("scripted", "predictive-timing", "timing-only"):
            raise ValueError(f"Unknown controller mode: {self.mode}")

        self.ball_q = int(model.joint("ball/ball_freejoint").qposadr[0])
        self.ball_v = int(model.joint("ball/ball_freejoint").dofadr[0])
        self.target_x = float(target_x)

        self.predictor = BallTrajectoryPredictor(target_x=self.target_x)
        self.planner = DescendingInterceptionPlanner(self.predictor)

        self.current_plan: InterceptionPlan | None = None
        self.feed_id: int = -1

    def reset_feed(self, feed_id: int, data: mujoco.MjData):
        """Reset state at feed launch."""
        self.feed_id = feed_id
        ball_pos = data.qpos[self.ball_q : self.ball_q + 3].copy()
        ball_vel = data.qvel[self.ball_v : self.ball_v + 3].copy()
        self.current_plan = self.planner.plan(ball_pos, ball_vel, feed_elapsed=0.0, target_x=self.target_x)

    def step(self, data: mujoco.MjData, feed_elapsed: float) -> np.ndarray:
        """Compute the 7-joint raw7 offset command for the current 50 Hz control tick.

        Args:
            data: current MjData simulation state.
            feed_elapsed: elapsed time since current feed launch (seconds).

        Returns:
            raw7: 7-element bounded arm setpoint offsets.
        """
        if self.mode == "scripted":
            return stroke_at(feed_elapsed)

        # Update interception plan while ball is approaching before forward stroke commits
        ball_pos = data.qpos[self.ball_q : self.ball_q + 3]
        ball_vel = data.qvel[self.ball_v : self.ball_v + 3]
        can_replan = (ball_vel[0] < -0.1) and (ball_pos[0] > -1.25) and (feed_elapsed < 0.24)
        if self.current_plan is None or can_replan:
            self.current_plan = self.planner.plan(
                ball_pos, ball_vel, feed_elapsed=feed_elapsed, target_x=self.target_x
            )

        tau = self.planner.effective_stroke_time(feed_elapsed, self.current_plan)
        return stroke_at(tau)
