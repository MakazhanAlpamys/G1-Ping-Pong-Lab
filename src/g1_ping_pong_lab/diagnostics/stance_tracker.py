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


class StanceTracker:
    """Continuously audits foot contact forces, slip, and body drift during simulation."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.pelvis_id = model.body("robot/pelvis").id
        self.lf_id = model.body("robot/left_ankle_roll_link").id
        self.rf_id = model.body("robot/right_ankle_roll_link").id

        self.lf_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == self.lf_id}
        self.rf_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == self.rf_id}

        self.current_feed_id = -1
        self.feed_start_time = 0.0
        self.history: list[dict] = []
        self.summaries: list[FeedStanceSummary] = []

        self._feed_init_state: dict = {}
        self._lf_min_fn = float("inf")
        self._rf_min_fn = float("inf")
        self._lf_max_ft = 0.0
        self._rf_max_ft = 0.0

    def start_feed(self, feed_id: int, data: mujoco.MjData, time: float):
        """Record baseline posture at feed start."""
        mujoco.mj_forward(self.model, data)
        self.current_feed_id = feed_id
        self.feed_start_time = float(time)
        self._lf_min_fn = float("inf")
        self._rf_min_fn = float("inf")
        self._lf_max_ft = 0.0
        self._rf_max_ft = 0.0

        self._feed_init_state = {
            "pelvis": data.xpos[self.pelvis_id].copy(),
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

        for c in data.contact[:data.ncon]:
            if c.efc_address < 0 or c.dist > 0:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, c.efc_address, f)
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

    def finish_feed(self, data: mujoco.MjData, time: float) -> FeedStanceSummary:
        """Compute net drift and contact metrics for the completed feed."""
        mujoco.mj_forward(self.model, data)
        pelvis_end = data.xpos[self.pelvis_id].copy()
        com_end = data.subtree_com[self.pelvis_id].copy()
        lf_end = data.xpos[self.lf_id].copy()
        rf_end = data.xpos[self.rf_id].copy()

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
        )
        self.summaries.append(summary)
        return summary
