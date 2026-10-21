"""Tool for auditing robot stance drift, contact dynamics, and ground reaction forces.

Executes continuous feed simulations with live StanceTracker diagnostics and
generates structured JSON and CSV reports with defensible metric definitions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from g1_ping_pong_lab.continuous_feeds import run_continuous
from g1_ping_pong_lab.diagnostics import StanceTracker


def audit_stance_drift(
    scene_path: Path,
    wbc_path: Path,
    feeds: int = 4,
    controller_mode: str = "scripted",
) -> dict:
    """Run continuous simulation and audit stance dynamics across all feeds."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    tracker = StanceTracker(model)

    print(f"Running stance drift audit over {feeds} feeds (controller: {controller_mode})...")
    report, _ = run_continuous(
        scene_path,
        wbc_path,
        feeds=feeds,
        interval=3.0,
        tail=3.0,
        controller_mode=controller_mode,
        diagnostics=tracker,
    )

    rec_metrics = {m["feed_id"]: m for m in report.get("recovery_metrics", [])}
    feed_diagnostics = []

    for s in tracker.summaries:
        fid = s.feed_id
        ball_outcome = report["feeds"][fid]["ball_outcome"] if fid < len(report["feeds"]) else {}
        rm = rec_metrics.get(fid, {})
        diag = {
            "feed_id": fid,
            "success": bool(ball_outcome.get("success", False)),
            "reason": ball_outcome.get("reason", "unknown"),
            "contacted": bool(ball_outcome.get("contacted", False)),
            "pelvis_drift_x_cm": float(s.pelvis_drift[0] * 100.0),
            "pelvis_drift_y_cm": float(s.pelvis_drift[1] * 100.0),
            "pelvis_drift_z_cm": float(s.pelvis_drift[2] * 100.0),
            "pelvis_yaw_drift_deg": float(s.pelvis_yaw_drift_deg),
            "torso_drift_x_cm": float(s.torso_drift[0] * 100.0),
            "torso_drift_y_cm": float(s.torso_drift[1] * 100.0),
            "torso_yaw_drift_deg": float(s.torso_yaw_drift_deg),
            "com_drift_x_cm": float(s.com_drift[0] * 100.0),
            "left_foot_drift_x_cm": float(s.left_foot_drift[0] * 100.0),
            "right_foot_drift_x_cm": float(s.right_foot_drift[0] * 100.0),
            "left_foot_loaded_displacement_cm": float(s.left_foot_loaded_displacement_cm),
            "right_foot_loaded_displacement_cm": float(s.right_foot_loaded_displacement_cm),
            "left_foot_airborne_displacement_cm": float(s.left_foot_airborne_displacement_cm),
            "right_foot_airborne_displacement_cm": float(s.right_foot_airborne_displacement_cm),
            "recovery_entry_yaw_err_deg": rm.get("recovery_entry_yaw_err_deg"),
            "recovery_exit_yaw_err_deg": rm.get("recovery_exit_yaw_err_exit_deg", rm.get("recovery_exit_yaw_err_deg")),
            "absolute_yaw_error_reduction_deg": rm.get("absolute_yaw_error_reduction_deg"),
            "signed_yaw_correction_deg": rm.get("signed_yaw_correction_deg"),
            "left_foot_lifted": bool(s.left_foot_lifted),
            "right_foot_lifted": bool(s.right_foot_lifted),
            "left_foot_low_load": bool(s.left_foot_low_load),
            "right_foot_low_load": bool(s.right_foot_low_load),
            "max_lf_tangential_force_N": float(s.max_lf_tangential_force),
            "max_rf_tangential_force_N": float(s.max_rf_tangential_force),
            "min_lf_normal_force_N": float(s.min_lf_normal_force),
            "min_rf_normal_force_N": float(s.min_rf_normal_force),
            "max_lf_friction_ratio": float(s.max_lf_friction_ratio),
            "max_rf_friction_ratio": float(s.max_rf_friction_ratio),
        }
        feed_diagnostics.append(diag)
        entry_str = f"{diag['recovery_entry_yaw_err_deg']:+5.2f}°" if diag['recovery_entry_yaw_err_deg'] is not None else "  N/A "
        exit_str = f"{diag['recovery_exit_yaw_err_deg']:+5.2f}°" if diag['recovery_exit_yaw_err_deg'] is not None else "  N/A "
        print(
            f"Feed {fid}: {diag['reason']:18s} | Pelvis drift: {diag['pelvis_drift_x_cm']:+5.2f} cm | "
            f"Torso yaw: {diag['torso_yaw_drift_deg']:+5.2f}° | "
            f"Entry: {entry_str} -> Exit: {exit_str} | "
            f"LF load-disp: {diag['left_foot_loaded_displacement_cm']:5.3f} cm | RF: {diag['right_foot_loaded_displacement_cm']:5.3f} cm"
        )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feeds": feeds,
        "controller_mode": controller_mode,
        "valid_returns": report["valid_returns"],
        "racket_contacts": report["racket_contacts"],
        "return_rate": report["return_rate"],
        "contact_rate": report["contact_rate"],
        "accepted_returns": report["accepted_returns"],
        "full_session_stance_accepted": report["full_session_stance_accepted"],
        "feed_diagnostics": feed_diagnostics,
        "recovery_phase_metrics": report.get("recovery_metrics", []),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--feeds", type=int, default=4)
    parser.add_argument(
        "--controller",
        choices=["scripted", "predictive-timing", "stance-recovery"],
        default="scripted",
        help="Controller mode to audit",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/stance_diagnostics"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = audit_stance_drift(args.scene, args.wbc, feeds=args.feeds, controller_mode=args.controller)

    json_file = args.output / f"stance_drift_{args.controller}_{args.feeds}feeds.json"
    json_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved stance audit JSON to {json_file}")

    csv_file = args.output / f"stance_drift_{args.controller}_{args.feeds}feeds.csv"
    if summary["feed_diagnostics"]:
        fieldnames = list(summary["feed_diagnostics"][0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary["feed_diagnostics"])
        print(f"Saved stance audit CSV to {csv_file}")


if __name__ == "__main__":
    main()
