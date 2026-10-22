import importlib
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def feature():
    try:
        return importlib.import_module("g1_ping_pong_lab.observation_benchmark")
    except ModuleNotFoundError:
        pytest.fail("The noise/latency benchmark is not implemented")


def test_matrix_pairs_noise_seeds_and_does_not_repeat_deterministic_cases():
    benchmark = feature()
    cases = benchmark.make_cases(seeds=[0, 1, 2], velocities=[-1.8, -1.98])
    assert len(cases) == 32
    assert len({case["id"] for case in cases}) == 32
    for vx in [-1.8, -1.98]:
        subset = [c for c in cases if c["vx"] == vx]
        assert len([c for c in subset if c["profile"] == "clean"]) == 1
        assert {c["seed"] for c in subset if c["profile"] == "combined"} == {0, 1, 2}
        assert {c["seed"] for c in subset if c["profile"] == "combined_timestamped"} == {0, 1, 2}


@pytest.mark.parametrize("kwargs", [
    {"seeds": []}, {"seeds": [-1]}, {"seeds": [0, 0]},
    {"velocities": [float("nan")]}, {"velocities": [1.]},
    {"profiles": ["unknown"]},
])
def test_invalid_matrix_is_rejected_before_starting_simulations(kwargs):
    with pytest.raises(ValueError):
        feature().make_cases(**kwargs)


def test_summary_uses_stance_accepted_returns_not_only_ball_contacts():
    benchmark = feature()
    rows = [dict(profile="combined", vx=-1.8, seed=seed, accepted_returns=accepted,
                 phase_reversals=reversals, stance_accepted=stance)
            for seed, accepted, reversals, stance in [(0, 1, 0, True), (1, 0, 2, False)]]
    group = benchmark.aggregate_results(rows)[0]
    assert group["trials"] == 2
    assert group["accepted_returns"] == 1
    assert group["stance_accepted_trials"] == 1
    assert group["phase_reversals"] == 2


def test_native_trial_writes_auditable_report_and_observation_trace(tmp_path):
    benchmark = feature()
    root = Path(__file__).resolve().parents[1]
    case = benchmark.make_cases(seeds=[0], velocities=[-1.8], profiles=["clean"])[0]
    row = benchmark.run_case(case, scene=root/"assets/scene.xml", assets=root/"assets/wbc",
                             output=tmp_path, seconds=1.2, keep_states=True)
    assert row["accepted_returns"] == 1
    folder = tmp_path / case["id"]
    report = json.loads((folder/"report.json").read_text())
    assert report["simulation"]["physics_substeps"] == 4800
    assert report["simulation"]["full_session_stance_accepted"]
    for name in ["observations.npz", "session.npz"]:
        assert benchmark.file_sha(folder/name) == report["artifacts_sha256"][name]
    with np.load(folder/"observations.npz", allow_pickle=False) as trace:
        assert trace["time"].shape == (60,)
        assert trace["position"].shape == (60, 3)
        np.testing.assert_allclose(trace["captured_at"], trace["time"])


def test_custom_matrix_pairs_seeds_and_retains_exact_sensor_settings():
    benchmark = feature()
    assert "profile_configs" in inspect.signature(benchmark.make_cases).parameters, "Custom matrices are missing"
    configs = {"delay_80ms": {"latency_s": .08, "timestamp_aware": True},
               "position_10mm": {"position_std_m": .01}}
    cases = benchmark.make_cases(seeds=[10, 11], velocities=[-1.8], profile_configs=configs)
    assert len(cases) == 3
    assert {c["seed"] for c in cases if c["profile"] == "position_10mm"} == {10, 11}
    delayed = next(c for c in cases if c["profile"] == "delay_80ms")
    assert delayed["observation_config"]["latency_s"] == .08
    assert delayed["observation_config"]["timestamp_aware"] is True


@pytest.mark.parametrize("configs", [
    {}, {"../outside": {}}, {"profile": {"seed": 12}},
    {"profile": {"latency_s": -.1}}, {"profile": {"position_std_m": "bad"}},
    {"profile": {"unknown": 1}},
])
def test_malformed_custom_profiles_are_rejected(configs):
    benchmark = feature()
    assert "profile_configs" in inspect.signature(benchmark.make_cases).parameters, "Custom matrices are missing"
    with pytest.raises(ValueError):
        benchmark.make_cases(profile_configs=configs)


def test_largest_phase_reversal_is_maximum_not_sum():
    rows = [dict(profile="noise", vx=-1.8, accepted_returns=1, stance_accepted=True,
                 phase_reversals=1, largest_phase_reversal_seconds=value) for value in [.01, .03]]
    group = feature().aggregate_results(rows)[0]
    assert group.get("largest_phase_reversal_seconds") == .03


def test_cli_rejects_bad_matrix_before_creating_output(tmp_path):
    config = tmp_path/"invalid.json"
    config.write_text(json.dumps({"../outside": {}}))
    output = tmp_path/"results"
    result = subprocess.run([sys.executable, "-m", "g1_ping_pong_lab.observation_benchmark",
                             "--matrix", str(config), "--output", str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "profile" in result.stderr.lower()
    assert "unrecognized arguments" not in result.stderr
    assert not output.exists()


def test_cli_runs_custom_matrix_with_capture_timestamps_and_records_input_hash(tmp_path):
    config = tmp_path/"matrix.json"
    config.write_text(json.dumps({"delay_10ms": {"latency_s": .01, "timestamp_aware": True}}))
    output = tmp_path/"results"
    result = subprocess.run([sys.executable, "-m", "g1_ping_pong_lab.observation_benchmark",
        "--matrix", str(config), "--output", str(output), "--seeds", "10", "--vx", "-1.8",
        "--seconds", "1.2", "--workers", "1"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/"summary.json").read_text())
    assert report["matrix_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert len(report["trials"]) == 1
    trial = report["trials"][0]
    assert trial["accepted_returns"] == 1
    assert trial["maximum_age_seconds"] == pytest.approx(.02)
    assert trial["physics_substeps"] == 4800
