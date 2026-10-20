"""Tool for auditing robot stance drift, contact slip, and ground reaction forces.

Executes continuous feed simulations and generates a structured JSON/CSV report
decomposing Case A (foot slip) vs Case B (postural lean).
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from g1_ping_pong_lab.continuous_feeds import run_continuous
from g1_ping_pong_lab.diagnostics import StanceTracker


def audit_stance_drift(scene_path: Path, wbc_path: Path, feeds: int = 4) -> dict:
    """Run continuous simulation and audit stance dynamics across all feeds."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    tracker = StanceTracker(model)

    print(f"Running stance drift audit over {feeds} feeds...")
    report, frames = run_continuous(scene_path, wbc_path, feeds=feeds, interval=3.0, tail=12.0)

    pelvis_id = model.body("robot/pelvis").id
    lf_id = model.body("robot/left_ankle_roll_link").id
    rf_id = model.body("robot/right_ankle_roll_link").id

    data = mujoco.MjData(model)
    feed_diagnostics = []

    for fid in range(feeds):
        feed_frames = [f for f in frames if f["trial_id"] == fid]
        if not feed_frames:
            continue
        start_f = feed_frames[0]
        end_f = feed_frames[-1]

        # Evaluate kinematics at start and end
        data.qpos[:] = start_f["qpos"]
        mujoco.mj_forward(model, data)
        p_start = data.xpos[pelvis_id].copy()
        lf_start = data.xpos[lf_id].copy()
        rf_start = data.xpos[rf_id].copy()
        com_start = data.subtree_com[pelvis_id].copy()

        data.qpos[:] = end_f["qpos"]
        mujoco.mj_forward(model, data)
        p_end = data.xpos[pelvis_id].copy()
        lf_end = data.xpos[lf_id].copy()
        rf_end = data.xpos[rf_id].copy()
        com_end = data.subtree_com[pelvis_id].copy()

        ball_outcome = report["feeds"][fid]["ball_outcome"]

        diag = {
            "feed_id": fid,
            "success": bool(ball_outcome["success"]),
            "reason": ball_outcome["reason"],
            "contacted": bool(ball_outcome.get("contacted", False)),
            "pelvis_x_start": float(p_start[0]),
            "pelvis_x_end": float(p_end[0]),
            "pelvis_drift_x_cm": float((p_end[0] - p_start[0]) * 100.0),
            "left_foot_drift_x_cm": float((lf_end[0] - lf_start[0]) * 100.0),
            "right_foot_drift_x_cm": float((rf_end[0] - rf_start[0]) * 100.0),
            "com_drift_x_cm": float((com_end[0] - com_start[0]) * 100.0),
            "case_a_foot_slip_cm": float((lf_end[0] - lf_start[0]) * 100.0),
            "case_b_postural_lean_cm": float(((p_end[0] - p_start[0]) - (lf_end[0] - lf_start[0])) * 100.0),
        }
        feed_diagnostics.append(diag)
        print(f"Feed {fid}: {diag['reason']:18s} | Pelvis drift: {diag['pelvis_drift_x_cm']:+5.2f} cm | "
              f"LF slip: {diag['left_foot_drift_x_cm']:+5.2f} cm | RF slip: {diag['right_foot_drift_x_cm']:+5.2f} cm")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feeds": feeds,
        "accepted_returns": report["accepted_returns"],
        "full_session_stance_accepted": report["full_session_stance_accepted"],
        "feed_diagnostics": feed_diagnostics,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--feeds", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/stance_diagnostics"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = audit_stance_drift(args.scene, args.wbc, feeds=args.feeds)

    out_file = args.output / f"stance_drift_{args.feeds}feeds.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved stance audit report to {out_file}")


if __name__ == "__main__":
    main()
