"""Active stance recovery controller for humanoid table tennis endurance.

Regulates torso heading orientation during the post-stroke recovery window
(after arm returns to READY state) to eliminate cumulative stance drift.
"""
from __future__ import annotations

import mujoco
import numpy as np


class StanceRecoveryController:
    """Inter-feed stance recovery controller for multi-feed endurance.

    During the idle observation window between balls (default 1.05 s <= t <= 2.95 s),
    this controller computes corrective references for the pretrained WBC:
    - Base yaw angular velocity (projected to the torso body frame `ref[4:7]`)
      to eliminate clockwise yaw recoil from arm strokes without destabilizing balance.
    - Optional forward linear velocity (`ref[1:4]`) using smooth Hann-window weighting.
    """

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        *,
        target_yaw: float = 0.0,
        target_pelvis_x: float = -1.800,
        kp_yaw: float = 4.0,
        ki_yaw: float = 1.0,
        kp_x: float = 0.0,
        max_yaw_rate: float = 1.0,
        max_lin_vel: float = 0.35,
        recovery_start: float = 1.05,
        recovery_end: float = 2.95,
        dt_control: float = 0.02,
    ):
        self.model = model
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

        self.torso_body_id: int | None = None
        self.pelvis_body_id: int | None = None
        if model is not None:
            self._cache_body_ids(model)

    def _cache_body_ids(self, model: mujoco.MjModel) -> None:
        self.model = model
        try:
            self.torso_body_id = model.body("robot/torso_link").id
        except KeyError:
            self.torso_body_id = None
        try:
            self.pelvis_body_id = model.body("robot/pelvis").id
        except KeyError:
            self.pelvis_body_id = None

    def reset_feed(self, feed_id: int):
        """Reset state at feed boundary."""
        self.feed_id = feed_id
        self.yaw_err_integral = 0.0

    def apply_recovery(
        self,
        data: mujoco.MjData,
        ref: np.ndarray,
        feed_elapsed: float,
        model: mujoco.MjModel | None = None,
    ) -> None:
        """Apply corrective reference bias during the inter-feed idle recovery window.

        Args:
            data: Current simulation state.
            ref: Mutable 39-dimensional WBC reference command vector.
            feed_elapsed: Elapsed time since current feed launch (seconds).
            model: Optional MjModel if not provided at initialization.
        """
        if not (self.recovery_start <= feed_elapsed <= self.recovery_end):
            return

        if self.torso_body_id is None and model is not None:
            self._cache_body_ids(model)

        # 1. Determine orientation & rotation matrix
        if self.torso_body_id is not None and hasattr(data, "xmat"):
            R_torso = data.xmat[self.torso_body_id].reshape(3, 3)
            # Torso heading in world frame from forward axis (column 0)
            current_yaw = np.arctan2(R_torso[1, 0], R_torso[0, 0])
        elif hasattr(data, "qpos") and len(data.qpos) >= 7:
            w, x, y, z = data.qpos[3:7]
            current_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            R_torso = np.eye(3)
        else:
            return

        # 2. Heading yaw regulation with [-pi, pi] angle wrapping
        raw_yaw_err = current_yaw - self.target_yaw
        yaw_err = (raw_yaw_err + np.pi) % (2.0 * np.pi) - np.pi

        self.yaw_err_integral += yaw_err * self.dt
        self.yaw_err_integral = float(np.clip(self.yaw_err_integral, -0.5, 0.5))

        cmd_yaw_world = -self.kp_yaw * yaw_err - self.ki_yaw * self.yaw_err_integral
        cmd_yaw_world = float(np.clip(cmd_yaw_world, -self.max_yaw_rate, self.max_yaw_rate))

        # Project world angular velocity [0, 0, cmd_yaw_world] to torso body frame: ref[4:7]
        omega_world = np.array([0.0, 0.0, cmd_yaw_world])
        ref[4:7] = R_torso.T @ omega_world

        # 3. Optional linear position regulation with smooth Hann-window weighting
        if self.kp_x > 0.0:
            pelvis_x = data.xpos[self.pelvis_body_id, 0] if (self.pelvis_body_id is not None and hasattr(data, "xpos")) else data.qpos[0]
            pelvis_x_err = pelvis_x - self.target_pelvis_x

            # Smooth Hann-window factor to avoid abrupt step changes and elastic recoil
            window_len = self.recovery_end - self.recovery_start
            tau = (feed_elapsed - self.recovery_start) / window_len if window_len > 0 else 1.0
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))

            cmd_lin_world_x = -self.kp_x * pelvis_x_err * window
            cmd_lin_world_x = float(np.clip(cmd_lin_world_x, -self.max_lin_vel, self.max_lin_vel))

            # Project world linear velocity [cmd_lin_world_x, 0, 0] to torso body frame: ref[1:4]
            v_world = np.array([cmd_lin_world_x, 0.0, 0.0])
            ref[1:4] = R_torso.T @ v_world
