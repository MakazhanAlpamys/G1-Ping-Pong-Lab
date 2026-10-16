# Results and limitations

The uncut reference demo shows **two accepted returns to scheduled new-ball feeds by one
continuously simulated robot**. The robot and WBC action history are initialized
once. New balls launch at 0 and 3 seconds; neither feed resets the robot.

The underlying native run lasts **15 seconds**, with a 12-second observation tail
after the second launch. The [published original demo](../media/demo.mp4) shows
the first six seconds uninterrupted at normal speed. The full physical window
is included in `data/continuous-session.npz` and can be rendered locally.

## Continuous-feeder validation

| Measurement | Recorded result |
| --- | --- |
| Scheduled feeds | 2, at 0 and 3 seconds |
| Accepted returns | 2 |
| Physical observation window | 15 seconds |
| Physics timestep | 0.25 ms |
| Audited physics substeps | 60,000 |
| Control rate | 50 Hz |
| Robot resets after initialization | 0 |
| WBC history resets after initialization | 0 |
| Ball replacements after the first launch | 1, at 3 seconds |
| Falls | 0 |
| External table/net support | 0 substeps |
| Non-foot ground contact | 0 substeps |
| Largest actuator force / native force cap | 1.0; no cap exceeded |
| Physics failure | None |
| Wall geometry | Removed from the physical scene |

The [continuous report](continuous-report.json) contains the feed boundaries,
physical event traces, source/scene hashes and full-session stance audit.
The [verification receipt](continuous-verification.json) checks its hashes and
all 750 saved frames' WBC action history. The matching recording is
`data/continuous-session.npz`.

The feed position is `[-0.65, -0.25, 1.0]` meters and velocity is
`[-1.8, 0.0, 0.5]` meters per second in fixed world coordinates. The two event
chains occur at these global simulation times:

| Event | First feed | Second feed |
| --- | --- | --- |
| Launch | 0 s | 3 s |
| Bounce on the robot's table half | 0.269 s | 3.269 s |
| Racket contact | 0.489 s | 3.492 s |
| Outgoing net clearance | 0.7675 s | 3.765 s |
| Bounce on the opposite half | 0.89325 s | 3.8945 s |

At the second feed boundary, measured instantaneous changes in robot joint/root
positions and published WBC history are both zero. The ball's position, velocity
and orientation are deliberately initialized for a new feed. There are no ball
state edits between scheduled feeds.

This is **not a same-ball rally**. The feeder replaces a departed ball rather
than observing and returning a ball struck by an opponent. The robot's motion
continues across that replacement.

## Longer attempts exposed a limit

| Probe | Accepted returns | Observation window |
| --- | --- | --- |
| [Four feeds, fixed world frame, 3-second interval](experiments/fixed-world-3s.json) | 2/4 | 21 s |
| [Four feeds, fixed world frame, 4-second interval](experiments/fixed-world-4s.json) | 2/4 | 24 s |
| [Four feeds, simulator-pose-relative feeder, 3-second interval](experiments/robot-relative-3s.json) | 1/4 | 21 s |

All three probes retained stance without external support or non-foot ground
contact, but later feeds missed, hit the body or landed incorrectly. Repositioning new feeds
relative to the simulated robot did not improve the tested result. That feeder
uses exact simulator pose and is not a perception system.

These are small diagnostic runs, not statistical success estimates. The released
short run does not establish an indefinitely repeatable skill or robust play
under varied inputs.

## Separate reset-trial baseline

The earlier baseline produced 8/8 accepted first returns across eight identical
independent trials. **Every trial reset the robot, ball and controller history.**
All eight recorded state/control sequences are identical. Those results show
fixed-feed repeatability from the same initial state, not eight returns without
resetting the robot.

Each baseline trial ran for 12 seconds, totaling 96 seconds and 384,000 physical
substeps. Falls, table/net support and non-foot ground contact were all zero;
no native actuator cap was exceeded. The reference recording took 146.69 seconds
of wall time.

Its [physical report](validation-report.json),
[verification receipt](validation-verification.json) and
`data/demo-session.npz` are separate from the continuous-feeder result.

## What controls the robot

A frozen pretrained whole-body network supplies balance. The project adds a
seven-joint arm-reference script selected using CEM search. Its preparation,
forward stroke and recovery are timed for a known feed. Restarting this arm
schedule for each new ball does not reset the robot's physical state or the
balance controller's action history.

The network is the unchanged Apache-2.0 asset from
[wbc-mjlab/wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy/tree/6dabf86fddc2b7b429b09e74999732fcde3441f9).
The scene uses the
[mjlab G1 description](https://github.com/mujocolab/mjlab/tree/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/asset_zoo/robots/unitree_g1)
and official Unitree meshes. See [third-party notices](../THIRD_PARTY_NOTICES.md)
for source attribution, licenses and hashes.

This release provides the fixed stroke and reproducible runtime. It does not
claim a newly trained general table-tennis policy.

## What the local renderer changes

In the uncut rendering mode, the camera moves smoothly while physical time advances at 1x. There are no
montage cuts, alternate shots, slow motion, titles or music. Lighting, colors
and render-only table/net detail are presentation changes; they do not change
the recorded movement or collision model. Every rendered pose comes from an
actual saved physical state, with no strike or trajectory interpolation.

The uncut video shows the first six seconds. Acceptance is determined from the
full 15-second native run, not the short video or a graphical counter. A legal
return requires actual own-half bounce, racket contact, net clearance and
opposite-half bounce; the scorer is a return benchmark, not a complete match referee.

## Not established

- Sustained same-ball rallies or reliable long sequences of new feeds.
- Robust returns across unknown speeds, spins, positions or timing.
- A stroke policy that adapts to a human opponent.
- Camera ball tracking, robot localization or calibrated table coordinates.
- Hardware transfer, actuator identification or real-world aerodynamic calibration.

The simulator provides exact state. Real-world perception, calibration and
hardware control require separate work. Follow the [quick start](quickstart.md)
to reproduce the short continuous run, inspect the reset-trial baseline or
render the uncut videos.
