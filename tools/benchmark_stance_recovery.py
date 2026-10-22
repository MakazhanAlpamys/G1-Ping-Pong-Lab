"""Benchmark tool comparing continuous feed endurance across controllers.

Evaluates Scripted, Predictive Timing, and Stance Recovery controllers across
multi-feed scenarios and feed velocity conditions. Isolates post-stroke recovery
window metrics from whole-session endpoints and supports repeated trials per condition.
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
    repetitions: int = 1,
) -> dict:
    """Execute comparative multi-trial benchmarks across all 3 controllers.

    Each condition is tested across the requested number of repetitions from
    identical initial conditions, and results are aggregated strictly per condition
    rather than pooled across different feed velocities.
    """
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
        "repetitions_per_condition": repetitions,
    }

    benchmark_results = []

    print("\n" + "=" * 90)
    print("STANCE RECOVERY & MULTI-FEED ENDURANCE BENCHMARK")
    print(f"Timestamp: {metadata['timestamp']} | Timestep: {metadata['timestep_s']}s | Repetitions/Cond: {repetitions}")
    print("=" * 90)

    for feeds in feed_counts:
        print(f"\n==================== {feeds}-FEED CONTINUOUS TRIALS ====================")
        for trial_name, trial_vel in trials:
            vel_str = f"custom_velocity={trial_vel}" if trial_vel else "nominal velocity"
            print(f"\n--- Condition: {trial_name} ({vel_str}) ---")
            for mode_key, mode_name in modes:
                for rep in range(repetitions):
                    rep_prefix = f"[Rep {rep+1}/{repetitions}] " if repetitions > 1 else ""
                    print(f"{rep_prefix}Testing: {mode_name:<38} ...", end=" ", flush=True)
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

                    # Calculate session-wide endpoints (secondary outcome)
                    if frames and len(frames) > 0:
                        start_f = frames[0]
                        end_f = frames[-1]
                        pelvis_x_start = float(start_f["qpos"][0])
                        pelvis_x_end = float(end_f["qpos"][0])
                        session_pelvis_drift_cm = round((pelvis_x_end - pelvis_x_start) * 100.0, 2)

                        w_end, x_end, y_end, z_end = end_f["qpos"][3:7]
                        end_yaw_rad = np.arctan2(2.0 * (w_end * z_end + x_end * y_end), 1.0 - 2.0 * (y_end**2 + z_end**2))
                        w_0, x_0, y_0, z_0 = start_f["qpos"][3:7]
                        start_yaw_rad = np.arctan2(2.0 * (w_0 * z_0 + x_0 * y_0), 1.0 - 2.0 * (y_0**2 + z_0**2))
                        yaw_diff_rad = (end_yaw_rad - start_yaw_rad + np.pi) % (2.0 * np.pi) - np.pi
                        session_final_yaw_deg = round(float(np.degrees(yaw_diff_rad)), 2)
                    else:
                        session_pelvis_drift_cm = None
                        session_final_yaw_deg = None

                    reasons = [f["ball_outcome"]["reason"] for f in report["feeds"]]
                    valid_returns = report["valid_returns"]
                    contacts = report["racket_contacts"]

                    # Primary recovery-window metrics (isolated post-stroke phase [1.05s, 2.95s])
                    rec_metrics = report.get("recovery_metrics", [])
                    if rec_metrics:
                        mean_rec_entry_yaw_err_deg = round(float(np.mean([m["recovery_entry_yaw_err_deg"] for m in rec_metrics])), 3)
                        mean_rec_exit_yaw_err_deg = round(float(np.mean([m["recovery_exit_yaw_err_deg"] for m in rec_metrics])), 3)
                        mean_rec_abs_reduction_deg = round(float(np.mean([m["absolute_yaw_error_reduction_deg"] for m in rec_metrics])), 3)
                        final_rec_exit_yaw_err_deg = round(float(rec_metrics[-1]["recovery_exit_yaw_err_deg"]), 3)
                    else:
                        mean_rec_entry_yaw_err_deg = None
                        mean_rec_exit_yaw_err_deg = None
                        mean_rec_abs_reduction_deg = None
                        final_rec_exit_yaw_err_deg = None

                    total_lf_disp = sum(s.left_foot_loaded_displacement_cm for s in tracker.summaries)
                    total_rf_disp = sum(s.right_foot_loaded_displacement_cm for s in tracker.summaries)

                    entry = {
                        "feeds": feeds,
                        "condition": trial_name,
                        "custom_velocity": trial_vel,
                        "repetition": rep,
                        "mode_key": mode_key,
                        "mode_name": mode_name,
                        "valid_returns": valid_returns,
                        "return_rate_pct": round((valid_returns / feeds) * 100.0, 1),
                        "racket_contacts": contacts,
                        "contact_rate_pct": round((contacts / feeds) * 100.0, 1),
                        "accepted_returns": report["accepted_returns"],
                        "mean_recovery_entry_yaw_err_deg": mean_rec_entry_yaw_err_deg,
                        "mean_recovery_exit_yaw_err_deg": mean_rec_exit_yaw_err_deg,
                        "mean_recovery_abs_reduction_deg": mean_rec_abs_reduction_deg,
                        "final_recovery_exit_yaw_err_deg": final_rec_exit_yaw_err_deg,
                        "session_pelvis_drift_cm": session_pelvis_drift_cm,
                        "session_final_yaw_deg": session_final_yaw_deg,
                        "avg_foot_loaded_disp_cm": round((total_lf_disp + total_rf_disp) / 2.0, 3),
                        "stance_accepted": report["full_session_stance_accepted"],
                        "max_force_cap_ratio": round(report["maximum_force_cap_ratio"], 4),
                        "outcomes": reasons,
                        "recovery_phase_metrics": rec_metrics,
                        "wall_time_s": round(elapsed, 2),
                    }
                    benchmark_results.append(entry)

                    exit_yaw_disp = f"{final_rec_exit_yaw_err_deg:+.2f}°" if final_rec_exit_yaw_err_deg is not None else "N/A"
                    red_disp = f"{mean_rec_abs_reduction_deg:+.2f}°" if mean_rec_abs_reduction_deg is not None else "N/A"
                    sess_yaw_disp = f"{session_final_yaw_deg:+.1f}°" if session_final_yaw_deg is not None else "N/A"
                    sess_drift_disp = f"{session_pelvis_drift_cm:+.2f}cm" if session_pelvis_drift_cm is not None else "N/A"

                    print(
                        f"Ret: {valid_returns}/{feeds} ({entry['return_rate_pct']}%) | "
                        f"Hit: {contacts}/{feeds} ({entry['contact_rate_pct']}%) | "
                        f"RecExitYaw: {exit_yaw_disp} (red: {red_disp}) | "
                        f"SessYaw: {sess_yaw_disp} | SessDrift: {sess_drift_disp} | {elapsed:.1f}s"
                    )

    # Compute condition-specific aggregates strictly per (feeds, condition, mode_key)
    # Does NOT pool different velocity conditions together.
    condition_aggregates = []
    for feeds in feed_counts:
        for trial_name, trial_vel in trials:
            for mode_key, mode_name in modes:
                matched = [
                    r for r in benchmark_results
                    if r["feeds"] == feeds and r["condition"] == trial_name and r["mode_key"] == mode_key
                ]
                if not matched:
                    continue
                ret_rates = [r["return_rate_pct"] for r in matched]
                cnt_rates = [r["contact_rate_pct"] for r in matched]
                rec_exit_yaws = [abs(r["final_recovery_exit_yaw_err_deg"]) for r in matched if r["final_recovery_exit_yaw_err_deg"] is not None]
                rec_reductions = [r["mean_recovery_abs_reduction_deg"] for r in matched if r["mean_recovery_abs_reduction_deg"] is not None]
                sess_yaws = [abs(r["session_final_yaw_deg"]) for r in matched if r["session_final_yaw_deg"] is not None]
                sess_drifts = [r["session_pelvis_drift_cm"] for r in matched if r["session_pelvis_drift_cm"] is not None]

                condition_aggregates.append({
                    "feeds": feeds,
                    "condition": trial_name,
                    "custom_velocity": trial_vel,
                    "mode_key": mode_key,
                    "mode_name": mode_name,
                    "repetitions": len(matched),
                    "mean_valid_returns": round(float(np.mean([r["valid_returns"] for r in matched])), 2),
                    "mean_return_rate_pct": round(float(np.mean(ret_rates)), 1),
                    "mean_racket_contacts": round(float(np.mean([r["racket_contacts"] for r in matched])), 2),
                    "mean_contact_rate_pct": round(float(np.mean(cnt_rates)), 1),
                    "mean_final_recovery_exit_yaw_err_deg": round(float(np.mean(rec_exit_yaws)), 2) if rec_exit_yaws else None,
                    "mean_recovery_abs_reduction_deg": round(float(np.mean(rec_reductions)), 3) if rec_reductions else None,
                    "mean_session_final_yaw_deg": round(float(np.mean(sess_yaws)), 2) if sess_yaws else None,
                    "mean_session_pelvis_drift_cm": round(float(np.mean(sess_drifts)), 2) if sess_drifts else None,
                    "all_stance_accepted": all(r["stance_accepted"] for r in matched),
                })

    output_dir = Path("outputs/stance_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "benchmark_comparison.json"
    full_output = {
        "metadata": metadata,
        "condition_aggregates": condition_aggregates,
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
    parser.add_argument("--repeats", "--repetitions", dest="repetitions", type=int, default=1,
                        help="Number of genuine repetitions to run per condition from identical initial state (default: 1)")
    parser.add_argument("--single-trial", action="store_true", help="Run only nominal trial instead of full matrix")
    args = parser.parse_args()

    trials = [("nominal", None)] if args.single_trial else None
    benchmark_stance_recovery(args.scene, args.wbc, feed_counts=args.feeds, trials=trials, repetitions=args.repetitions)


if __name__ == "__main__":
    main()
