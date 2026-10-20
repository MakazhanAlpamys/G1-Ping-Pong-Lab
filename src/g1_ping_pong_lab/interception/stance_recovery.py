"""Active stance recovery controller for humanoid table tennis endurance.

Regulates pelvis yaw orientation and forward position during the post-stroke
recovery window (after arm returns to READY state) to eliminate cumulative stance drift.
"""
from __future__ import annotations

import mujoco
import numpy as np


class StanceRecoveryController:
    """Inter-feed stance recovery controller for multi-feed endurance.

    During the idle observation window between balls (default 1.05 s <= t <= 2.95 s),
    this controller computes corrective references for the pretrained WBC:
    - Base yaw angular velocity (`ref[6]`) to eliminate clockwise yaw recoil from arm strokes.
    - Base forward linear velocity (`ref[1]`) to eliminate rearward displacement without lifting feet.
    """

    def __init__(
        self,
        *,
        target_yaw: float = 0.0,
        target_pelvis_x: float = -1.800,
        kp_yaw: float = 4.0,
        ki_yaw: float = 1.0,
        kp_x: float = 3.5,
        max_yaw_rate: float = 1.0,
        max_lin_vel: float = 0.35,
        recovery_start: float = 1.05,
        recovery_end: float = 2.95,
        dt_control: float = 0.02,
    ):
        self.target_yaw = float(target_yaw)
        self.target_pelvis_x = float(target_pelvis_x)
        self.kp_yaw = float(kp_yaw)
        self.ki_yaw = float(ki_yaw)
        self.kp_x = float(kp_x)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_lin_vel = float(max_lin_vel)
        self.recovery_start = float(recovery_start)
        self.recovery_end = float(recovery_end)
        self.dt = float(dt_control)

        self.yaw_err_integral: float = 0.0
        self.feed_id: int = -1

    def reset_feed(self, feed_id: int):
        """Reset state at feed boundary."""
        self.feed_id = feed_id
        self.yaw_err_integral = 0.0

    def apply_recovery(self, data: mujoco.MjData, ref: np.ndarray, feed_elapsed: float) -> None:
        """Apply corrective reference bias during the inter-feed idle recovery window.

        Args:
            data: Current simulation state.
            ref: Mutable 39-dimensional WBC reference command vector.
            feed_elapsed: Elapsed time since current feed launch (seconds).
        """
        if not (self.recovery_start <= feed_elapsed <= self.recovery_end):
            return

        # 1. Pelvis yaw regulation
        w, x, y, z = data.qpos[3:7]
        current_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_err = current_yaw - self.target_yaw
        self.yaw_err_integral += yaw_err * self.dt
        self.yaw_err_integral = float(np.clip(self.yaw_err_integral, -0.5, 0.5))

        cmd_yaw = -self.kp_yaw * yaw_err - self.ki_yaw * self.yaw_err_integral
        ref[6] = float(np.clip(cmd_yaw, -self.max_yaw_rate, self.max_yaw_rate))

        # 2. Pelvis forward position regulation (Case B lean correction)
        pelvis_x = data.qpos[0]
        pelvis_x_err = pelvis_x - self.target_pelvis_x  # negative if drifted rearward
        cmd_lin = -self.kp_x * pelvis_x_err
        ref[1] = float(np.clip(cmd_lin, 0.0, self.max_lin_vel))
