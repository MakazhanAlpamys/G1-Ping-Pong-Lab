"""Diagnostics module for tracking robot stance, foot contact dynamics, and drift."""
from __future__ import annotations

from dataclasses import dataclass, field
import mujoco
import numpy as np


@dataclass
class FeedStanceSummary:
    feed_id: int
    start_time: float
    end_time: float
    pelvis_start: np.ndarray
    pelvis_end: np.ndarray
    pelvis_drift: np.ndarray
    com_start: np.ndarray
    com_end: np.ndarray
    com_drift: np.ndarray
    left_foot_start: np.ndarray
    left_foot_end: np.ndarray
    left_foot_drift: np.ndarray
    right_foot_start: np.ndarray
    right_foot_end: np.ndarray
    right_foot_drift: np.ndarray
    left_foot_lifted: bool
    right_foot_lifted: bool
    max_lf_tangential_force: float
    max_rf_tangential_force: float
    # Enhanced diagnostics
    left_foot_slip_cm: float = 0.0
    right_foot_slip_cm: float = 0.0
    left_foot_airborne_cm: float = 0.0
    right_foot_airborne_cm: float = 0.0
    pelvis_yaw_drift_deg: float = 0.0
    torso_drift: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torso_yaw_drift_deg: float = 0.0
    min_lf_normal_force: float = 0.0
    min_rf_normal_force: float = 0.0

    def to_dict(self) -> dict:
        """Serialize stance summary into JSON-compatible dictionary."""
        return {
            "feed_id": self.feed_id,
            "start_time": round(float(self.start_time), 4),
            "end_time": round(float(self.end_time), 4),
            "pelvis_drift_cm": (self.pelvis_drift * 100.0).round(2).tolist(),
            "pelvis_yaw_drift_deg": round(float(self.pelvis_yaw_drift_deg), 2),
            "torso_drift_cm": (self.torso_drift * 100.0).round(2).tolist(),
            "torso_yaw_drift_deg": round(float(self.torso_yaw_drift_deg), 2),
            "com_drift_cm": (self.com_drift * 100.0).round(2).tolist(),
            "left_foot_drift_cm": (self.left_foot_drift * 100.0).round(2).tolist(),
            "right_foot_drift_cm": (self.right_foot_drift * 100.0).round(2).tolist(),
            "left_foot_slip_cm": round(float(self.left_foot_slip_cm), 3),
            "right_foot_slip_cm": round(float(self.right_foot_slip_cm), 3),
            "left_foot_airborne_cm": round(float(self.left_foot_airborne_cm), 3),
            "right_foot_airborne_cm": round(float(self.right_foot_airborne_cm), 3),
            "left_foot_lifted": bool(self.left_foot_lifted),
            "right_foot_lifted": bool(self.right_foot_lifted),
            "max_lf_tangential_force_N": round(float(self.max_lf_tangential_force), 2),
            "max_rf_tangential_force_N": round(float(self.max_rf_tangential_force), 2),
            "min_lf_normal_force_N": round(float(self.min_lf_normal_force), 2),
            "min_rf_normal_force_N": round(float(self.min_rf_normal_force), 2),
        }


class StanceTracker:
    """Continuously audits foot contact forces, slip, and body drift during simulation."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.pelvis_id = model.body("robot/pelvis").id
        self.torso_id = model.body("robot/torso_link").id
        self.lf_id = model.body("robot/left_ankle_roll_link").id
        self.rf_id = model.body("robot/right_ankle_roll_link").id

        # Include roll and pitch links for complete foot contact tracking
        lf_bodies = {self.lf_id, model.body("robot/left_ankle_pitch_link").id}
        rf_bodies = {self.rf_id, model.body("robot/right_ankle_pitch_link").id}
        self.lf_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] in lf_bodies}
        self.rf_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] in rf_bodies}

        self.current_feed_id = -1
        self.feed_start_time = 0.0
        self.history: list[dict] = []
        self.summaries: list[FeedStanceSummary] = []

        self._feed_init_state: dict = {}
        self._lf_min_fn = float("inf")
        self._rf_min_fn = float("inf")
        self._lf_max_ft = 0.0
        self._rf_max_ft = 0.0

        self._last_lf_pos: np.ndarray | None = None
        self._last_rf_pos: np.ndarray | None = None
        self._lf_slip_dist = 0.0
        self._rf_slip_dist = 0.0
        self._lf_airborne_dist = 0.0
        self._rf_airborne_dist = 0.0

    def start_feed(self, feed_id: int, data: mujoco.MjData, time: float):
        """Record baseline posture at feed start."""
        mujoco.mj_forward(self.model, data)
        self.current_feed_id = feed_id
        self.feed_start_time = float(time)
        self._lf_min_fn = float("inf")
        self._rf_min_fn = float("inf")
        self._lf_max_ft = 0.0
        self._rf_max_ft = 0.0

        self._lf_slip_dist = 0.0
        self._rf_slip_dist = 0.0
        self._lf_airborne_dist = 0.0
        self._rf_airborne_dist = 0.0

        lf_pos = data.xpos[self.lf_id][:2].copy()
        rf_pos = data.xpos[self.rf_id][:2].copy()
        self._last_lf_pos = lf_pos
        self._last_rf_pos = rf_pos

        w_p, x_p, y_p, z_p = data.qpos[3:7]
        pelvis_yaw = float(np.arctan2(2.0 * (w_p * z_p + x_p * y_p), 1.0 - 2.0 * (y_p * y_p + z_p * z_p)))

        R_torso = data.xmat[self.torso_id].reshape(3, 3)
        torso_yaw = float(np.arctan2(R_torso[1, 0], R_torso[0, 0]))

        self._feed_init_state = {
            "pelvis": data.xpos[self.pelvis_id].copy(),
            "pelvis_yaw": pelvis_yaw,
            "torso": data.xpos[self.torso_id].copy(),
            "torso_yaw": torso_yaw,
            "com": data.subtree_com[self.pelvis_id].copy(),
            "lf": data.xpos[self.lf_id].copy(),
            "rf": data.xpos[self.rf_id].copy(),
        }

    def observe_step(self, data: mujoco.MjData, time: float):
        """Observe contact forces and state on each physics substep or control tick."""
        lf_fn = 0.0
        rf_fn = 0.0
        lf_ft = 0.0
        rf_ft = 0.0

        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0 or c.dist > 0:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, f)
            fn = max(0.0, float(f[0]))
            ft = float(np.hypot(f[1], f[2]))

            if c.geom1 in self.lf_geoms or c.geom2 in self.lf_geoms:
                lf_fn += fn
                lf_ft += ft
            if c.geom1 in self.rf_geoms or c.geom2 in self.rf_geoms:
                rf_fn += fn
                rf_ft += ft

        self._lf_min_fn = min(self._lf_min_fn, lf_fn)
        self._rf_min_fn = min(self._rf_min_fn, rf_fn)
        self._lf_max_ft = max(self._lf_max_ft, lf_ft)
        self._rf_max_ft = max(self._rf_max_ft, rf_ft)

        # Foot displacement: classify as contact slip vs airborne step
        lf_pos = data.xpos[self.lf_id][:2]
        rf_pos = data.xpos[self.rf_id][:2]

        if self._last_lf_pos is not None:
            dlf = float(np.linalg.norm(lf_pos - self._last_lf_pos))
            if lf_fn > 1.0:
                self._lf_slip_dist += dlf
            else:
                self._lf_airborne_dist += dlf

        if self._last_rf_pos is not None:
            drf = float(np.linalg.norm(rf_pos - self._last_rf_pos))
            if rf_fn > 1.0:
                self._rf_slip_dist += drf
            else:
                self._rf_airborne_dist += drf

        self._last_lf_pos = lf_pos.copy()
        self._last_rf_pos = rf_pos.copy()

    def finish_feed(self, data: mujoco.MjData, time: float) -> FeedStanceSummary:
        """Compute net drift and contact metrics for the completed feed."""
        mujoco.mj_forward(self.model, data)
        pelvis_end = data.xpos[self.pelvis_id].copy()
        torso_end = data.xpos[self.torso_id].copy()
        com_end = data.subtree_com[self.pelvis_id].copy()
        lf_end = data.xpos[self.lf_id].copy()
        rf_end = data.xpos[self.rf_id].copy()

        w_p, x_p, y_p, z_p = data.qpos[3:7]
        pelvis_yaw_end = float(np.arctan2(2.0 * (w_p * z_p + x_p * y_p), 1.0 - 2.0 * (y_p * y_p + z_p * z_p)))
        pelvis_yaw_err = pelvis_yaw_end - self._feed_init_state.get("pelvis_yaw", 0.0)
        pelvis_yaw_drift_deg = float(np.degrees((pelvis_yaw_err + np.pi) % (2.0 * np.pi) - np.pi))

        R_torso = data.xmat[self.torso_id].reshape(3, 3)
        torso_yaw_end = float(np.arctan2(R_torso[1, 0], R_torso[0, 0]))
        torso_yaw_err = torso_yaw_end - self._feed_init_state.get("torso_yaw", 0.0)
        torso_yaw_drift_deg = float(np.degrees((torso_yaw_err + np.pi) % (2.0 * np.pi) - np.pi))

        summary = FeedStanceSummary(
            feed_id=self.current_feed_id,
            start_time=self.feed_start_time,
            end_time=float(time),
            pelvis_start=self._feed_init_state["pelvis"],
            pelvis_end=pelvis_end,
            pelvis_drift=pelvis_end - self._feed_init_state["pelvis"],
            com_start=self._feed_init_state["com"],
            com_end=com_end,
            com_drift=com_end - self._feed_init_state["com"],
            left_foot_start=self._feed_init_state["lf"],
            left_foot_end=lf_end,
            left_foot_drift=lf_end - self._feed_init_state["lf"],
            right_foot_start=self._feed_init_state["rf"],
            right_foot_end=rf_end,
            right_foot_drift=rf_end - self._feed_init_state["rf"],
            left_foot_lifted=bool(self._lf_min_fn < 5.0),
            right_foot_lifted=bool(self._rf_min_fn < 5.0),
            max_lf_tangential_force=self._lf_max_ft,
            max_rf_tangential_force=self._rf_max_ft,
            left_foot_slip_cm=self._lf_slip_dist * 100.0,
            right_foot_slip_cm=self._rf_slip_dist * 100.0,
            left_foot_airborne_cm=self._lf_airborne_dist * 100.0,
            right_foot_airborne_cm=self._rf_airborne_dist * 100.0,
            pelvis_yaw_drift_deg=pelvis_yaw_drift_deg,
            torso_drift=torso_end - self._feed_init_state["torso"],
            torso_yaw_drift_deg=torso_yaw_drift_deg,
            min_lf_normal_force=0.0 if np.isinf(self._lf_min_fn) else self._lf_min_fn,
            min_rf_normal_force=0.0 if np.isinf(self._rf_min_fn) else self._rf_min_fn,
        )
        self.summaries.append(summary)
        return summary
