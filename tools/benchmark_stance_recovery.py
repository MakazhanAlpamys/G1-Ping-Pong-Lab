"""Comparative benchmark for multi-feed endurance and stance drift recovery.

Executes a multi-trial evaluation matrix comparing:
1. Scripted Baseline
2. Predictive Timing Alone
3. Active Stance Recovery + Predictive Timing

across 4-feed and 8-feed continuous sessions under nominal and perturbed conditions, measuring:
- Raw valid returns and return rate
- Raw racket contacts and contact rate
- Stance safety acceptance (no falls, force limits <= 1.0)
- Per-feed recovery phase metrics (entry yaw error, exit yaw error, error reduction)
- Net base translation and final heading yaw drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from g1_ping_pong_lab.continuous_feeds import run_continuous
from g1_ping_pong_lab.diagnostics import StanceTracker


def file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def benchmark_stance_recovery(
    scene_path: Path,
    wbc_path: Path,
    feed_counts: list[int] | None = None,
    trials: list[tuple[str, list[float] | None]] | None = None,
) -> dict:
    """Execute comparative multi-trial benchmarks across all 3 controllers."""
    if feed_counts is None:
        feed_counts = [4, 8]
    if trials is None:
        trials = [
            ("nominal", None),
            ("fast_feed", [-1.9, 0.0, 0.52]),
            ("slow_feed", [-1.7, 0.0, 0.48]),
        ]

    modes = [
        ("scripted", "Scripted Baseline"),
        ("predictive-timing", "Predictive Timing Alone"),
        ("stance-recovery", "Stance Recovery + Predictive Timing"),
    ]
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    # Environment metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(model.opt.timestep),
        "dt_control_s": 0.02,
        "scene_sha256": file_sha256(scene_path),
        "wbc_policy_sha256": file_sha256(wbc_path / "policy.onnx"),
        "wbc_config_sha256": file_sha256(wbc_path / "config.yaml"),
    }

    benchmark_results = []

    print("\n" + "=" * 84)
    print("STANCE RECOVERY & MULTI-FEED ENDURANCE BENCHMARK")
    print(f"Timestamp: {metadata['timestamp']} | Timestep: {metadata['timestep_s']}s")
    print("=" * 84)

    for feeds in feed_counts:
        print(f"\n==================== {feeds}-FEED CONTINUOUS TRIALS ====================")
        for trial_name, trial_vel in trials:
            vel_str = f"custom_velocity={trial_vel}" if trial_vel else "nominal velocity"
            print(f"\n--- Condition: {trial_name} ({vel_str}) ---")
            for mode_key, mode_name in modes:
                print(f"Testing: {mode_name:<40} ...", end=" ", flush=True)
                tracker = StanceTracker(model)
                t0 = time.time()
                report, frames = run_continuous(
                    scene_path,
                    wbc_path,
                    feeds=feeds,
                    interval=3.0,
                    tail=3.0,
                    controller_mode=mode_key,
                    custom_velocity=trial_vel,
                    diagnostics=tracker,
                )
                elapsed = time.time() - t0

                # Calculate session endpoints
                start_f = frames[0]
                end_f = frames[-1]
                pelvis_x_start = float(start_f["qpos"][0])
                pelvis_x_end = float(end_f["qpos"][0])
                pelvis_drift_cm = (pelvis_x_end - pelvis_x_start) * 100.0

                w_end, x_end, y_end, z_end = end_f["qpos"][3:7]
                end_yaw_rad = np.arctan2(2.0 * (w_end * z_end + x_end * y_end), 1.0 - 2.0 * (y_end**2 + z_end**2))
                w_0, x_0, y_0, z_0 = start_f["qpos"][3:7]
                start_yaw_rad = np.arctan2(2.0 * (w_0 * z_0 + x_0 * y_0), 1.0 - 2.0 * (y_0**2 + z_0**2))
                yaw_diff_rad = (end_yaw_rad - start_yaw_rad + np.pi) % (2.0 * np.pi) - np.pi
                end_yaw_deg = float(np.degrees(yaw_diff_rad))

                reasons = [f["ball_outcome"]["reason"] for f in report["feeds"]]
                valid_returns = report["valid_returns"]
                contacts = report["racket_contacts"]

                # Recovery phase metrics
                rec_metrics = report.get("recovery_metrics", [])
                avg_abs_reduction = (
                    float(np.mean([m["absolute_yaw_error_reduction_deg"] for m in rec_metrics]))
                    if rec_metrics else 0.0
                )
                total_lf_disp = sum(s.left_foot_loaded_displacement_cm for s in tracker.summaries)
                total_rf_disp = sum(s.right_foot_loaded_displacement_cm for s in tracker.summaries)

                entry = {
                    "feeds": feeds,
                    "condition": trial_name,
                    "custom_velocity": trial_vel,
                    "mode_key": mode_key,
                    "mode_name": mode_name,
                    "valid_returns": valid_returns,
                    "return_rate_pct": round((valid_returns / feeds) * 100.0, 1),
                    "racket_contacts": contacts,
                    "contact_rate_pct": round((contacts / feeds) * 100.0, 1),
                    "accepted_returns": report["accepted_returns"],
                    "pelvis_drift_cm": round(pelvis_drift_cm, 2),
                    "final_yaw_deg": round(end_yaw_deg, 2),
                    "mean_recovery_abs_reduction_deg": round(avg_abs_reduction, 3),
                    "avg_foot_loaded_disp_cm": round((total_lf_disp + total_rf_disp) / 2.0, 3),
                    "stance_accepted": report["full_session_stance_accepted"],
                    "max_force_cap_ratio": round(report["maximum_force_cap_ratio"], 4),
                    "outcomes": reasons,
                    "recovery_phase_metrics": rec_metrics,
                    "wall_time_s": round(elapsed, 2),
                }
                benchmark_results.append(entry)
                print(
                    f"Returns: {valid_returns}/{feeds} ({entry['return_rate_pct']}%) | "
                    f"Contacts: {contacts}/{feeds} ({entry['contact_rate_pct']}%) | "
                    f"Drift: {pelvis_drift_cm:+.2f}cm | Yaw: {end_yaw_deg:+.1f}° | Time: {elapsed:.1f}s"
                )

    # Compute aggregate summaries per (feeds, mode_key)
    aggregates = []
    for feeds in feed_counts:
        for mode_key, mode_name in modes:
            matched = [r for r in benchmark_results if r["feeds"] == feeds and r["mode_key"] == mode_key]
            if not matched:
                continue
            ret_rates = [r["return_rate_pct"] for r in matched]
            cnt_rates = [r["contact_rate_pct"] for r in matched]
            yaws = [abs(r["final_yaw_deg"]) for r in matched]
            drifts = [r["pelvis_drift_cm"] for r in matched]
            aggregates.append({
                "feeds": feeds,
                "mode_key": mode_key,
                "mode_name": mode_name,
                "trials_count": len(matched),
                "mean_return_rate_pct": round(float(np.mean(ret_rates)), 1),
                "min_return_rate_pct": round(float(np.min(ret_rates)), 1),
                "max_return_rate_pct": round(float(np.max(ret_rates)), 1),
                "mean_contact_rate_pct": round(float(np.mean(cnt_rates)), 1),
                "min_contact_rate_pct": round(float(np.min(cnt_rates)), 1),
                "max_contact_rate_pct": round(float(np.max(cnt_rates)), 1),
                "mean_abs_final_yaw_deg": round(float(np.mean(yaws)), 2),
                "mean_pelvis_drift_cm": round(float(np.mean(drifts)), 2),
                "all_stance_accepted": all(r["stance_accepted"] for r in matched),
            })

    output_dir = Path("outputs/stance_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "benchmark_comparison.json"
    full_output = {
        "metadata": metadata,
        "aggregates": aggregates,
        "results": benchmark_results,
    }
    with open(out_file, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved structured benchmark comparison to {out_file}")

    return full_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--feeds", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--single-trial", action="store_true", help="Run only nominal trial instead of full matrix")
    args = parser.parse_args()

    trials = [("nominal", None)] if args.single_trial else None
    benchmark_stance_recovery(args.scene, args.wbc, feed_counts=args.feeds, trials=trials)


if __name__ == "__main__":
    main()
