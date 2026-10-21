"""Comparative benchmark for multi-feed endurance and stance drift recovery.

Compares:
1. Scripted Baseline
2. Predictive Timing Alone
3. Active Stance Recovery + Predictive Timing

across 4-feed and 8-feed continuous sessions, measuring:
- Valid return rate
- Ball contact rate
- Stance acceptance (no falls, force limits <= 1.0)
- Net base translation and yaw drift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from g1_ping_pong_lab.continuous_feeds import run_continuous
from g1_ping_pong_lab.diagnostics import StanceTracker


def benchmark_stance_recovery(
    scene_path: Path,
    wbc_path: Path,
    feed_counts: list[int] | None = None,
) -> dict:
    """Execute comparative benchmarks across all 3 controllers."""
    if feed_counts is None:
        feed_counts = [4, 8]
    modes = [
        ("scripted", "Scripted Baseline"),
        ("predictive-timing", "Predictive Timing Alone"),
        ("stance-recovery", "Stance Recovery + Predictive Timing"),
    ]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    torso_id = model.body("robot/torso_link").id

    benchmark_results = []

    print("\n" + "=" * 82)
    print("STANCE RECOVERY & MULTI-FEED ENDURANCE BENCHMARK")
    print("=" * 82)

    for feeds in feed_counts:
        print(f"\n--- Running {feeds}-Feed Continuous Trials ---")
        for mode_key, mode_name in modes:
            print(f"Testing: {mode_name:<42} ...", end=" ", flush=True)
            tracker = StanceTracker(model)
            t0 = time.time()
            report, frames = run_continuous(
                scene_path,
                wbc_path,
                feeds=feeds,
                interval=3.0,
                tail=3.0,
                controller_mode=mode_key,
                diagnostics=tracker,
            )
            elapsed = time.time() - t0

            # Calculate drift from frames
            start_f = frames[0]
            end_f = frames[-1]
            pelvis_x_start = float(start_f["qpos"][0])
            pelvis_x_end = float(end_f["qpos"][0])
            pelvis_drift_cm = (pelvis_x_end - pelvis_x_start) * 100.0

            # Yaw calculation with wrap
            w_end, x_end, y_end, z_end = end_f["qpos"][3:7]
            end_yaw_rad = np.arctan2(2.0 * (w_end * z_end + x_end * y_end), 1.0 - 2.0 * (y_end**2 + z_end**2))
            w_0, x_0, y_0, z_0 = start_f["qpos"][3:7]
            start_yaw_rad = np.arctan2(2.0 * (w_0 * z_0 + x_0 * y_0), 1.0 - 2.0 * (y_0**2 + z_0**2))
            yaw_diff_rad = (end_yaw_rad - start_yaw_rad + np.pi) % (2.0 * np.pi) - np.pi
            end_yaw_deg = float(np.degrees(yaw_diff_rad))

            reasons = [f["ball_outcome"]["reason"] for f in report["feeds"]]
            returns = report["accepted_returns"]
            contacts = sum(bool(f["ball_outcome"].get("contacted", False)) for f in report["feeds"])

            # Cumulative foot slip from tracker
            total_lf_slip_cm = sum(s.left_foot_slip_cm for s in tracker.summaries)
            total_rf_slip_cm = sum(s.right_foot_slip_cm for s in tracker.summaries)

            entry = {
                "feeds": feeds,
                "mode_key": mode_key,
                "mode_name": mode_name,
                "accepted_returns": returns,
                "return_rate_pct": round((returns / feeds) * 100.0, 1),
                "contacts": contacts,
                "contact_rate_pct": round((contacts / feeds) * 100.0, 1),
                "pelvis_drift_cm": round(pelvis_drift_cm, 2),
                "final_yaw_deg": round(end_yaw_deg, 2),
                "avg_foot_slip_cm": round((total_lf_slip_cm + total_rf_slip_cm) / 2.0, 3),
                "stance_accepted": report["full_session_stance_accepted"],
                "max_force_cap_ratio": round(report["maximum_force_cap_ratio"], 4),
                "outcomes": reasons,
                "wall_time_s": round(elapsed, 2),
            }
            benchmark_results.append(entry)
            print(
                f"Returns: {returns}/{feeds} | Contacts: {contacts}/{feeds} | "
                f"Drift: {pelvis_drift_cm:+.2f}cm | Yaw: {end_yaw_deg:+.1f}° | Time: {elapsed:.1f}s"
            )

    output_dir = Path("outputs/stance_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "benchmark_comparison.json"
    with open(out_file, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "results": benchmark_results}, f, indent=2)
    print(f"\nSaved benchmark comparison to {out_file}")

    return {"results": benchmark_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--feeds", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()

    benchmark_stance_recovery(args.scene, args.wbc, feed_counts=args.feeds)


if __name__ == "__main__":
    main()
