"""Benchmark suite for validating the isolated predictive-timing contribution.

Evaluates:
1. Multi-feed continuous endurance (4 feeds and 8 feeds).
2. Feed velocity perturbation sweep (-15% to +15%).
3. Lateral position variations (launch y perturbations).

Outputs structured JSON and CSV metrics to outputs/predictive_timing/.
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


def run_endurance_suite(scene_path: Path, wbc_path: Path) -> list[dict]:
    """Run 4-feed and 8-feed continuous endurance benchmarks."""
    modes = ["scripted", "predictive-timing"]
    feed_counts = [4, 8]
    results = []

    print("\n" + "=" * 70)
    print("SUITE 1: CONTINUOUS MULTI-FEED ENDURANCE BENCHMARK")
    print("=" * 70)

    for feeds in feed_counts:
        for mode in modes:
            print(f"\n[Running] Feeds={feeds} | Controller={mode} ...", flush=True)
            t0 = time.time()
            report, _ = run_continuous(
                scene_path,
                wbc_path,
                feeds=feeds,
                interval=3.0,
                tail=12.0,
                controller_mode=mode,
            )
            elapsed = time.time() - t0

            reasons = [f["ball_outcome"]["reason"] for f in report["feeds"]]
            success_count = report["accepted_returns"]
            contact_count = sum(bool(f["ball_outcome"].get("contacted", False)) for f in report["feeds"])
            success_rate = (success_count / feeds) * 100.0
            contact_rate = (contact_count / feeds) * 100.0

            entry = {
                "suite": "endurance",
                "feeds": feeds,
                "controller_mode": mode,
                "total_feeds": feeds,
                "accepted_returns": success_count,
                "success_rate_pct": round(success_rate, 1),
                "contacted_count": contact_count,
                "contact_rate_pct": round(contact_rate, 1),
                "stance_accepted": report["full_session_stance_accepted"],
                "fallen": report["fallen"],
                "max_force_cap_ratio": round(report["maximum_force_cap_ratio"], 4),
                "feed_outcomes": reasons,
                "wall_time_s": round(elapsed, 2),
            }
            results.append(entry)
            print(f"  -> Returns: {success_count}/{feeds} ({success_rate:.1f}%) | "
                  f"Contacts: {contact_count}/{feeds} | Outcomes: {reasons} | Time: {elapsed:.1f}s")

    return results


def run_velocity_sweep_suite(scene_path: Path, wbc_path: Path) -> list[dict]:
    """Run velocity perturbation sweep (-15% to +15%)."""
    speeds = [-1.53, -1.62, -1.71, -1.80, -1.89, -1.98, -2.07]
    modes = ["scripted", "predictive-timing"]
    results = []

    print("\n" + "=" * 70)
    print("SUITE 2: FEED VELOCITY PERTURBATION SWEEP (-15% to +15%)")
    print("=" * 70)

    for vx in speeds:
        delta_pct = round((vx - (-1.80)) / (-1.80) * 100)
        custom_v = [vx, 0.0, 0.5]

        for mode in modes:
            report, _ = run_continuous(
                scene_path,
                wbc_path,
                feeds=1,
                interval=3.0,
                tail=3.0,
                controller_mode=mode,
                custom_velocity=custom_v,
            )
            f0 = report["feeds"][0]["ball_outcome"]
            contacted = bool(f0.get("contacted", False))
            success = bool(report["accepted_returns"] > 0)
            reason = f0.get("reason", "unknown")

            contact_time = None
            contact_pos = None
            landing_pos = None
            if "trace" in f0:
                for ev in f0["trace"]:
                    if ev["event"] == "racket_contact":
                        contact_time = round(ev["time"], 4)
                        contact_pos = [round(x, 4) for x in ev["position"]]
                    elif ev["event"] in ("far_bounce", "first_landing", "wrong_first_landing"):
                        landing_pos = [round(x, 4) for x in ev["position"]]

            entry = {
                "suite": "velocity_sweep",
                "vx_launch": vx,
                "delta_pct": f"{delta_pct:+d}%",
                "controller_mode": mode,
                "success": success,
                "reason": reason,
                "contacted": contacted,
                "contact_time": contact_time,
                "contact_pos": contact_pos,
                "landing_pos": landing_pos,
            }
            results.append(entry)
            print(f"vx={vx:+.2f} ({delta_pct:+3d}%) | Mode={mode:18s} | "
                  f"Success={str(success):5s} | Reason={reason:18s} | Contacted={contacted}")

    return results


def run_variation_suite(scene_path: Path, wbc_path: Path) -> list[dict]:
    """Run lateral launch perturbations and varied speeds to test repeatability."""
    test_cases = [
        {"name": "nominal", "v": [-1.80, 0.0, 0.5]},
        {"name": "plus_3pct", "v": [-1.854, 0.0, 0.5]},
        {"name": "plus_7pct", "v": [-1.926, 0.0, 0.5]},
        {"name": "plus_12pct", "v": [-2.016, 0.0, 0.5]},
        {"name": "lateral_left", "v": [-1.80, -0.05, 0.5]},
        {"name": "lateral_right", "v": [-1.80, 0.05, 0.5]},
    ]
    modes = ["scripted", "predictive-timing"]
    results = []

    print("\n" + "=" * 70)
    print("SUITE 3: EXPANDED REPEATABILITY & VARIATION TRIALS")
    print("=" * 70)

    for tc in test_cases:
        for mode in modes:
            report, _ = run_continuous(
                scene_path,
                wbc_path,
                feeds=1,
                interval=3.0,
                tail=3.0,
                controller_mode=mode,
                custom_velocity=tc["v"],
            )
            f0 = report["feeds"][0]["ball_outcome"]
            entry = {
                "suite": "variation_trials",
                "case_name": tc["name"],
                "velocity": tc["v"],
                "controller_mode": mode,
                "success": bool(report["accepted_returns"] > 0),
                "reason": f0.get("reason", "unknown"),
                "contacted": bool(f0.get("contacted", False)),
            }
            results.append(entry)
            print(f"Case={tc['name']:14s} | v={tc['v']} | Mode={mode:18s} | "
                  f"Success={str(entry['success']):5s} | Reason={entry['reason']}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictive_timing"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("G1-PING-PONG-LAB: PREDICTIVE TIMING BENCHMARK SUITE")
    print("=" * 70)

    endurance_results = run_endurance_suite(args.scene, args.wbc)
    velocity_results = run_velocity_sweep_suite(args.scene, args.wbc)
    variation_results = run_variation_suite(args.scene, args.wbc)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "g1-ping-pong-lab.predictive-timing-validation-v1",
        "endurance_suite": endurance_results,
        "velocity_sweep_suite": velocity_results,
        "variation_trials": variation_results,
    }
    json_path = args.output / "benchmark-summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[Artifact Generated] JSON Summary: {json_path}")

    csv_path = args.output / "benchmark-results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["--- SUITE 1: ENDURANCE BENCHMARK ---"])
        writer.writerow(["Feeds", "Mode", "Accepted Returns", "Success Rate (%)", "Contacts", "Contact Rate (%)", "Stance Accepted", "Outcomes"])
        for r in endurance_results:
            writer.writerow([
                r["feeds"],
                r["controller_mode"],
                f"{r['accepted_returns']}/{r['total_feeds']}",
                r["success_rate_pct"],
                f"{r['contacted_count']}/{r['total_feeds']}",
                r["contact_rate_pct"],
                r["stance_accepted"],
                " ".join(r["feed_outcomes"]),
            ])

        writer.writerow([])
        writer.writerow(["--- SUITE 2: VELOCITY SWEEP BENCHMARK ---"])
        writer.writerow(["Launch vx", "Speed Delta", "Mode", "Success", "Reason", "Contacted", "Contact Time", "Contact Pos"])
        for r in velocity_results:
            writer.writerow([
                r["vx_launch"],
                r["delta_pct"],
                r["controller_mode"],
                r["success"],
                r["reason"],
                r["contacted"],
                r["contact_time"],
                r["contact_pos"],
            ])

        writer.writerow([])
        writer.writerow(["--- SUITE 3: EXPANDED VARIATIONS ---"])
        writer.writerow(["Case Name", "Launch Velocity", "Mode", "Success", "Reason", "Contacted"])
        for r in variation_results:
            writer.writerow([
                r["case_name"],
                r["velocity"],
                r["controller_mode"],
                r["success"],
                r["reason"],
                r["contacted"],
            ])
    print(f"[Artifact Generated] CSV Table: {csv_path}")


if __name__ == "__main__":
    main()
