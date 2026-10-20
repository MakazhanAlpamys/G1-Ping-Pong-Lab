"""Reproducible native MuJoCo experiments with noisy, delayed ball measurements."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np
import onnxruntime
import yaml

from .ball_observations import ObservationConfig
from .continuous_feeds import run_continuous


PROFILES = {
    "scripted": {},
    "clean": {},
    "position_5mm": {"position_std_m": .005},
    "velocity_0p05": {"velocity_std_mps": .05},
    "delay_40ms": {"latency_s": .04},
    "delay_40ms_timestamped": {"latency_s": .04, "timestamp_aware": True},
    "combined": {"position_std_m": .005, "velocity_std_mps": .05, "latency_s": .04},
    "combined_timestamped": {"position_std_m": .005, "velocity_std_mps": .05,
                             "latency_s": .04, "timestamp_aware": True},
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_cases(*, seeds=(0, 1, 2), velocities=(-1.8, -1.98), profiles=None):
    profiles = list(PROFILES) if profiles is None else list(profiles)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
            or not velocities or len(set(velocities)) != len(velocities)
            or any(not np.isfinite(vx) or vx >= -.1 for vx in velocities)
            or not profiles or len(set(profiles)) != len(profiles)
            or any(name not in PROFILES for name in profiles)):
        raise ValueError("Use distinct nonnegative seeds, incoming finite velocities and known profiles")
    cases = []
    for index, vx in enumerate(velocities):
        for profile in profiles:
            params = PROFILES[profile]
            noisy = params.get("position_std_m", 0.) or params.get("velocity_std_mps", 0.)
            for seed in seeds if noisy else seeds[:1]:
                cases.append({"id": f"v{index:02d}-{profile}-s{seed}", "profile": profile,
                              "vx": float(vx), "seed": seed,
                              "controller_mode": "scripted" if profile == "scripted" else "predictive-timing",
                              "observation_config": asdict(ObservationConfig(seed=seed, **params))})
    return cases


def aggregate_results(rows):
    groups = {}
    for row in rows:
        key = row["vx"], row["profile"]
        group = groups.setdefault(key, {"vx": row["vx"], "profile": row["profile"],
            "trials": 0, "accepted_returns": 0, "stance_accepted_trials": 0, "phase_reversals": 0})
        group["trials"] += 1
        group["accepted_returns"] += row["accepted_returns"]
        group["stance_accepted_trials"] += int(row["stance_accepted"])
        group["phase_reversals"] += row["phase_reversals"]
    return [groups[key] for key in sorted(groups)]


def provenance(scene, assets):
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    return {"git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "source_sha256": {p.relative_to(root).as_posix(): file_sha(p)
                              for p in sorted(package.rglob("*.py"))},
            "scene_sha256": file_sha(scene),
            "wbc_sha256": {name: file_sha(Path(assets)/name) for name in ["policy.onnx", "config.yaml"]},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "mujoco": mujoco.__version__, "numpy": np.__version__,
                            "onnxruntime": onnxruntime.__version__, "pyyaml": yaml.__version__}}


def run_case(case, *, scene, assets, output, seconds=12., keep_states=False):
    folder = Path(output)/case["id"]
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report, frames = run_continuous(scene, assets, feeds=1, interval=3., tail=seconds,
        custom_velocity=[case["vx"], 0., .5], controller_mode=case["controller_mode"],
        observation_config=ObservationConfig(**case["observation_config"]))
    trace = report.pop("observation_trace")
    fields = ["time", "feed_id", "captured_at", "age_seconds", "truth_position", "truth_velocity",
              "position", "velocity", "phase", "predicted_impact_time", "replanned"]
    vectors = {"truth_position", "truth_velocity", "position", "velocity"}
    arrays = {key: np.array([row[key] if row[key] is not None
                            else ([np.nan]*3 if key in vectors else np.nan)
                            for row in trace]) for key in fields}
    np.savez_compressed(folder/"observations.npz", **arrays)
    artifacts = {"observations.npz": file_sha(folder/"observations.npz")}
    if keep_states:
        np.savez_compressed(folder/"session.npz", **{
            key: np.array([frame[key] for frame in frames]) for key in frames[0]})
        artifacts["session.npz"] = file_sha(folder/"session.npz")
    sensor = report["ball_observations"]
    row = {"id": case["id"], "profile": case["profile"], "vx": case["vx"], "seed": case["seed"],
           "accepted_returns": report["accepted_returns"],
           "reason": report["feeds"][0]["ball_outcome"]["reason"],
           "stance_accepted": report["full_session_stance_accepted"], "fallen": report["fallen"],
           "max_force_cap_ratio": report["maximum_force_cap_ratio"],
           "physics_substeps": report["physics_substeps"], "phase_reversals": sensor["phase_reversals"],
           "maximum_age_seconds": sensor["maximum_age_seconds"],
           "ticks_without_measurement": sensor["ticks_without_measurement"]}
    document = {"case": case, "summary": row, "simulation": report,
                "provenance": provenance(scene, assets), "artifacts_sha256": artifacts,
                "elapsed_wall_seconds": time.monotonic()-started}
    (folder/"report.json").write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    return row


def _worker(task):
    case, kwargs = task
    return run_case(case, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--output", type=Path, default=Path("outputs/noise-latency"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--vx", nargs="+", type=float, default=[-1.8, -1.98])
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-states", action="store_true")
    args = parser.parse_args()
    try:
        cases = make_cases(seeds=args.seeds, velocities=args.vx, profiles=args.profiles)
        if (not np.isfinite(args.seconds) or args.seconds < 1.2
                or abs(args.seconds/.02-round(args.seconds/.02)) > 1.e-8
                or not 1 <= args.workers <= 16):
            raise ValueError("Use seconds >=1.2 in 20 ms increments and 1..16 workers")
        source = provenance(args.scene, args.wbc)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    kwargs = dict(scene=args.scene.resolve(), assets=args.wbc.resolve(),
                  output=(args.output/"trials").resolve(), seconds=args.seconds,
                  keep_states=args.keep_states)
    tasks = [(case, kwargs) for case in cases]
    rows = []
    # Fixed seeds and independent simulator/controller instances make ordering irrelevant.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(_worker, task) for task in tasks]):
            row = future.result()
            rows.append(row)
            print(json.dumps({"completed": len(rows), "total": len(cases), **row}), flush=True)
    rows.sort(key=lambda row: row["id"])
    summary = {"protocol": "g1-ping-pong-lab.ball-observation-noise-latency-v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "seconds_per_trial": args.seconds,
               "assumptions": ["Independent reset trials with fixed feeds and three-dimensional Gaussian noise.",
                   "Ball measurements captured at 50 Hz; delivery ages are quantized to control ticks.",
                   "Only predictor inputs are perturbed; WBC uses exact simulator state.",
                   "Startup holds READY until the first delayed measurement arrives.",
                   "Timestamp-aware mode uses capture time; it adds no filtering or future state.",
                   "Small synthetic-noise pilot; no camera model, hardware validation or endurance claim."],
               "provenance": source, "cases": cases, "trials": rows, "groups": aggregate_results(rows),
               "trial_report_sha256": {row["id"]: file_sha(args.output/"trials"/row["id"]/"report.json")
                                       for row in rows}}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"summary": str(args.output/"summary.json"), "trials": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
