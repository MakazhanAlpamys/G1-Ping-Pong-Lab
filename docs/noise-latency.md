# Ball measurement noise and latency

This benchmark extends Pernav's predictive impact timing from
[PR #1](https://github.com/MakazhanAlpamys/G1-Ping-Pong-Lab/pull/1).
It measures sensitivity to imperfect ball inputs without changing the frozen
WBC, physical scene, scorer, predictor equations or existing default execution.
Base drift and state filtering remain separate experiments.

## Run

Use the pinned Python dependencies in the quick start, from the repository root:

```bash
python -m g1_ping_pong_lab.observation_benchmark --workers 4 --keep-states --output outputs/noise-latency
```

The destination must not already exist. The default matrix has 32 independent
trials: nominal `vx=-1.8` and faster `vx=-1.98` m/s, each initialized from the same
robot state. Every trial audits **12 seconds / 48,000 physical substeps**.
Three noisy repetitions use seeds 0, 1 and 2. Deterministic controls run once per
speed; repeating a deterministic run does not supply independent evidence.

For a smaller run, select profiles and seeds explicitly:

```bash
python -m g1_ping_pong_lab.observation_benchmark --profiles clean delay_40ms delay_40ms_timestamped --seeds 0 --vx -1.8 --workers 2 --output outputs/delay-comparison
```

`--seconds` can shorten debugging runs, but the published pilot uses the complete
12-second observation window. Never compare a short run's acceptance directly
with a longer stance audit without identifying that difference.

## Prespecified profiles

| Profile | Position std / axis | Velocity std / axis | Delivery delay | Planning clock |
| --- | --- | --- | --- | --- |
| `scripted` | 0 | 0 | 0 | Fixed arm script; ignores ball measurements |
| `clean` | 0 | 0 | 0 | Current time |
| `position_5mm` | 5 mm | 0 | 0 | Current time |
| `velocity_0p05` | 0 | 0.05 m/s | 0 | Current time |
| `delay_40ms` | 0 | 0 | 40 ms | Current time; deliberately ignores sample age |
| `delay_40ms_timestamped` | 0 | 0 | 40 ms | Capture time |
| `combined` | 5 mm | 0.05 m/s | 40 ms | Current time |
| `combined_timestamped` | 5 mm | 0.05 m/s | 40 ms | Capture time |

Noise is zero-mean Gaussian, independent across axes and capture ticks. The same
six standard normal draws per tick are used for matching seeds across profiles.
It is applied once at capture, before enqueueing; repeated delivery never draws
new noise. These values define a synthetic pilot, not measured camera accuracy.

## Causality and time

Measurements are captured at the existing **50 Hz control ticks**. At time `t`,
only a sample captured no later than `t - latency` is available. Requested delays
that do not align with 20 ms ticks are effectively rounded up at delivery: a
10 ms setting gives a 20 ms sample age. Reports record actual age, not just the
configured delay. A 1 ns comparison tolerance handles floating point clocks.

At each new-ball feed the queue and timing plan are cleared. The RNG continues
across feeds. During startup the controller holds the zero-offset READY arm
reference until a delayed measurement is available; it never initializes a plan
from hidden perfect ball state. The robot and WBC history continue normally.

Uncompensated mode presents the delayed position and velocity as though captured
now. Timestamp-aware mode invokes the same predictor using the sample's capture
time, obtaining an absolute feed-relative impact time, then evaluates stroke
phase on the current clock. This does not add a filter, estimate velocity from
images, or consult future states. It still relies on the calibrated flight model.

Only copies of ball position and velocity enter this channel. Collision physics,
scoring, balance feedback and WBC action history always use true simulator state.
The observer adapter retains the original replanning gate and phase behavior,
including any phase reversals under changing predictions; reversals are measured
rather than silently corrected. Clean-sensor regression tests compare complete
recorded native trajectories against the original execution.

## Evidence

Each trial writes `report.json` with the full physical return event trace, stance
audit, force cap ratio, config, seed, runtime versions and SHA-256 hashes of all
runtime Python sources, scene, WBC/config and generated artifacts.
`observations.npz` contains capture/delivery clocks, truth and measured states,
phase, predicted impact time and replanning flags at every control tick.
NaN values in this NPZ mean no measurement/plan was available; JSON uses null.
`--keep-states` also saves the full native `session.npz`.

`summary.json` contains all cases, grouped accepted returns, stance counts and
phase reversals, with hashes of the per-trial reports. The benchmark refuses to
overwrite an existing destination. Interrupted runs retain completed trial
artifacts, but do not produce a complete summary.

This is an independent-feed, small-seed sensing experiment. Its outcome does not
establish continuous endurance, robustness to correlated camera error, occlusion,
spin, variable capture rate, delay jitter or real-hardware performance. In
particular, the WBC still receives exact simulated robot state.

## Recorded pilot

The [published evidence](experiments/noise-latency-pilot.json) was generated from
runtime commit `18717db`, with Python 3.12.10, MuJoCo 3.11.0, NumPy 2.5.4 and
ONNX Runtime 1.31.0 on Windows. All 32 trials used the full 12-second window.
Counts below are accepted returns / trials for the prespecified seeds:

| Profile | Nominal, -1.8 m/s | Faster +10%, -1.98 m/s |
| --- | --- | --- |
| Scripted control | 1/1 | 0/1 |
| Clean predictive control | 1/1 | 1/1 |
| Position noise, 5 mm per axis | 2/3 | 3/3 |
| Velocity noise, 0.05 m/s per axis | 2/3 | 3/3 |
| Delay 40 ms, sample age ignored | 0/1 | 0/1 |
| Delay 40 ms, capture time used | 1/1 | 1/1 |
| Combined noise + delay, sample age ignored | 0/3 | 0/3 |
| Combined noise + delay, capture time used | 3/3 | 3/3 |

Every trial passed the stance audit: no fall, external table/net support,
non-foot ground contact or native force-cap excess. In total, 1,536,000 physics
substeps were audited. The independently rerun 12-second scripted control
matched Pernav's unmodified `cfe6451` runtime bitwise across all 600 recorded
frames and every state/control array.

Capture timestamps restored the tested delayed returns, including the combined
profiles. This result depends on the known calibrated flight model and does not
establish a general latency bound. The matrix tests one delay magnitude and
one magnitude for each noise component; three seeds are a pilot, not a reliable
success-rate estimate. The different nominal/faster counts should not be read
as evidence that increasing feed speed generally improves noise robustness.

Noisy velocity inputs also produced 29 backward phase steps across the noisy
trials, with a largest single reversal of about 26.14 ms. Some trials still
succeeded despite those reversals. Capture-time compensation alone does not
remove that jitter; the observation traces provide reproducible inputs for
Pernav's filtering work. Compare any future filter with the same profiles,
seeds and full stance windows before widening the speed/noise/delay grid.

The evidence file embeds the full per-trial physical event reports and artifact
hashes. All 32 report hashes, 64 NPZ artifact hashes, causal delivery clocks and
all saved WBC previous-action histories were checked. Large state and sensor
archives remain local under `outputs/noise-latency-pilot-verified/` and can be
regenerated using the command above. Unit/integration verification was
68 passed, 2 skipped (optional EGL graphics), plus 16 passing subtests.
