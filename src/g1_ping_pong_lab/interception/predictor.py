"""Ball trajectory predictor for table tennis interception.

Analytically propagates the ball's ballistic flight under gravity and
models table bounce using restitution parameters calibrated to MuJoCo physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PredictionResult:
    """Predicted ball interception parameters."""
    impact_time: float
    impact_pos: np.ndarray
    bounce_time: Optional[float]
    bounce_pos: Optional[np.ndarray]
    is_post_bounce: bool


class BallTrajectoryPredictor:
    """Predicts ball impact time and 3D position at a target interception plane."""

    def __init__(
        self,
        *,
        gravity: float = 9.81,
        table_z: float = 0.760,
        ball_radius: float = 0.020,
        restitution_z: float = 0.65895,
        tangential_restitution: float = 0.8520,
        target_x: float = -1.47158,
    ):
        self.g = float(gravity)
        self.z_table = float(table_z)
        self.ball_radius = float(ball_radius)
        self.z_contact = self.z_table + self.ball_radius
        self.restitution_z = float(restitution_z)
        self.tangential_restitution = float(tangential_restitution)
        self.target_x = float(target_x)

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        current_time: float = 0.0,
        target_x: Optional[float] = None,
    ) -> PredictionResult:
        """Predict interception time and coordinates given current ball state.

        Args:
            pos: 3D position [x, y, z] of the ball at current_time.
            vel: 3D velocity [vx, vy, vz] of the ball at current_time.
            current_time: current simulation time in seconds.
            target_x: interception plane x coordinate (defaults to self.target_x).

        Returns:
            PredictionResult with impact time, 3D position, and bounce info.
        """
        p = np.asarray(pos, dtype=np.float64)
        v = np.asarray(vel, dtype=np.float64)
        t_curr = float(current_time)
        tx = self.target_x if target_x is None else float(target_x)

        if v[0] >= -1e-6:
            # Ball is moving away from the robot or stationary in x
            return PredictionResult(
                impact_time=t_curr + 10.0,
                impact_pos=p.copy(),
                bounce_time=None,
                bounce_pos=None,
                is_post_bounce=False,
            )

        # Check if the ball has already completed its table bounce
        is_post_bounce = (p[0] < -1.134) or (p[2] >= self.z_contact and v[2] > 0.05 and p[0] < -0.8)

        if is_post_bounce:
            # Direct ballistic propagation to target_x
            dt = max(0.0, (tx - p[0]) / v[0])
            t_impact = t_curr + dt
            p_impact = np.array([
                tx,
                p[1] + v[1] * dt,
                p[2] + v[2] * dt - 0.5 * self.g * dt * dt,
            ])
            return PredictionResult(
                impact_time=t_impact,
                impact_pos=p_impact,
                bounce_time=None,
                bounce_pos=None,
                is_post_bounce=True,
            )

        # Pre-bounce: calculate time to table contact
        # 0.5 * g * dt_b^2 - v_z * dt_b + (z_contact - z_p) = 0
        disc = v[2] * v[2] + 2.0 * self.g * (p[2] - self.z_contact)
        if disc < 0:
            # Cannot reach table contact; fallback to direct flight
            dt = max(0.0, (tx - p[0]) / v[0])
            t_impact = t_curr + dt
            p_impact = np.array([
                tx,
                p[1] + v[1] * dt,
                p[2] + v[2] * dt - 0.5 * self.g * dt * dt,
            ])
            return PredictionResult(
                impact_time=t_impact,
                impact_pos=p_impact,
                bounce_time=None,
                bounce_pos=None,
                is_post_bounce=False,
            )

        dt_b = (v[2] + np.sqrt(disc)) / self.g
        t_bounce = t_curr + dt_b
        p_bounce = np.array([
            p[0] + v[0] * dt_b,
            p[1] + v[1] * dt_b,
            self.z_contact,
        ])
        vz_pre = v[2] - self.g * dt_b

        # Apply table contact restitution
        # kz: vertical coefficient of restitution (-vz_post / vz_pre)
        # kx: tangential velocity retention ratio (vx_post / vx_pre)
        v_post = np.array([
            v[0] * self.tangential_restitution,
            v[1] * self.tangential_restitution,
            -self.restitution_z * vz_pre,
        ])

        # Propagate from bounce to target_x
        dt_post = max(0.0, (tx - p_bounce[0]) / v_post[0])
        t_impact = t_bounce + dt_post
        p_impact = np.array([
            tx,
            p_bounce[1] + v_post[1] * dt_post,
            self.z_contact + v_post[2] * dt_post - 0.5 * self.g * dt_post * dt_post,
        ])

        return PredictionResult(
            impact_time=t_impact,
            impact_pos=p_impact,
            bounce_time=t_bounce,
            bounce_pos=p_bounce,
            is_post_bounce=False,
        )
